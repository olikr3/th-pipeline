"""Seasonal-naive baseline with empirical residual quantiles.

The point forecast for a week is the same week a year earlier. Series with too
little history for that, or for a year of its errors, use the mean of their last
four weeks instead. Quantiles add the empirical quantiles of the method's errors
over the trailing ``residual_weeks`` to the point.

Cold-start series (under 13 weeks of history) have too few errors of their own and
take the error quantiles of their category x location instead, relative to the
forecast level, falling back to the whole tenant.

The spread does not widen with the horizon. That is exact for seasonal naive,
whose forecast for week t uses week t - 52 whatever the origin, and understates
the spread of the four-week mean at long horizons.
"""

from datetime import timedelta

import polars as pl

from thp.core.asof import AsOfStore
from thp.core.config import DemandSettings
from thp.demand.forecast import finish_forecast
from thp.features.demand import NEW_SERIES_WEEKS, SERIES, weekly_demand
from thp.schema import QUANTILE_COLS, QUANTILES

MODEL_VERSION = "seasonal_naive_v1"

SEASON_WEEKS = 52
LEVEL_WEEKS = 4
# A seasonal series needs a full year back plus a cold-start span of errors.
SEASONAL_MIN_WEEKS = SEASON_WEEKS + NEW_SERIES_WEEKS


def seasonal_naive(store: AsOfStore, settings: DemandSettings) -> pl.DataFrame:
    weekly, last_end = weekly_demand(store)
    qty = pl.col("qty")
    weekly = weekly.sort(*SERIES, "week_end").with_columns(
        history_weeks=pl.len().over(SERIES),
        _season=qty.shift(SEASON_WEEKS).over(SERIES),
        # One-step forecast of the four-week mean, for its errors.
        _prior_level=qty.shift(1).rolling_mean(LEVEL_WEEKS, min_samples=1).over(SERIES),
        _level=qty.rolling_mean(LEVEL_WEEKS, min_samples=1).over(SERIES),
    )
    weekly = weekly.with_columns(
        _seasonal=pl.col("history_weeks") >= SEASONAL_MIN_WEEKS,
        is_cold_start=pl.col("history_weeks") < NEW_SERIES_WEEKS,
    )
    category = store.reference("sku").select("sku_id", "category_l1").collect()

    window_start = last_end - timedelta(weeks=settings.residual_weeks)
    fitted = pl.when(pl.col("_seasonal")).then(pl.col("_season")).otherwise(pl.col("_prior_level"))
    errors = (
        weekly.filter(pl.col("week_end") > window_start)
        .with_columns(_fit=fitted)
        .filter(pl.col("_fit").is_not_null())
        .with_columns(_err=qty - pl.col("_fit"))
        .with_columns(_rel=pl.col("_err") / pl.max_horizontal(pl.col("_fit"), pl.lit(1.0)))
        .join(category, on="sku_id", how="left")
    )
    own = errors.group_by(SERIES).agg(_quantiles("_err", "off"))
    pooled_keys = ["category_l1", "location_id"]
    pooled = errors.group_by(pooled_keys).agg(_quantiles("_rel", "rel"))
    tenant = errors.select(_quantiles("_rel", "rel_tenant"))

    horizons = pl.DataFrame(
        {"horizon_week": range(1, settings.horizon_weeks + 1)}, schema={"horizon_week": pl.Int16}
    )
    origin = weekly.filter(pl.col("week_end") == last_end).select(
        *SERIES, "_seasonal", "_level", "is_cold_start"
    )
    year_ago = weekly.select(*SERIES, _src="week_end", _season_point="qty")
    rows = (
        origin.join(horizons, how="cross")
        .with_columns(week_end=pl.lit(last_end) + pl.duration(weeks=pl.col("horizon_week")))
        .with_columns(_src=pl.col("week_end") - pl.duration(weeks=SEASON_WEEKS))
        .join(year_ago, on=[*SERIES, "_src"], how="left")
        .with_columns(
            _point=pl.when(pl.col("_seasonal"))
            .then(pl.col("_season_point"))
            .otherwise(pl.col("_level"))
        )
        .join(category, on="sku_id", how="left")
        .join(own, on=SERIES, how="left")
        .join(pooled, on=pooled_keys, how="left", nulls_equal=True)
        .join(tenant, how="cross")
    )

    point = pl.col("_point")
    scale = pl.max_horizontal(point, pl.lit(1.0))
    quantiles = [
        pl.when(pl.col("is_cold_start"))
        .then(point + pl.coalesce(f"rel_{col}", f"rel_tenant_{col}", pl.lit(0.0)) * scale)
        .otherwise(point + pl.col(f"off_{col}").fill_null(0.0))
        .alias(col)
        for col in QUANTILE_COLS
    ]
    return finish_forecast(rows.with_columns(quantiles), store, MODEL_VERSION)


def _quantiles(col: str, prefix: str) -> list[pl.Expr]:
    return [
        pl.col(col).quantile(q, interpolation="linear").alias(f"{prefix}_{name}")
        for q, name in zip(QUANTILES, QUANTILE_COLS, strict=True)
    ]
