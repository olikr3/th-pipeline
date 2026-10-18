"""Forecasts from another source, re-based onto as_of's horizons.

For each series, the latest forecast ``source`` issued on or before ``as_of``
(and no more than ``external_max_age_days`` earlier) is used; its target weeks
after as_of's last complete week become horizons 1, 2, ... Weeks the source did
not forecast are missing, not filled.

Sources often send a point forecast only, but the simulator needs quantiles. A
missing quantile is the source's centre (q50, else qty) scaled by the baseline's
ratio of that quantile to its median for the same series and horizon, or shifted
by their difference where the baseline median is zero. Quantiles the source gives
are kept. Rows that still miss a quantile (no baseline, i.e. no sales history) are
returned as skipped.
"""

from datetime import timedelta
from typing import NamedTuple

import polars as pl

from thp.core.asof import AsOfStore
from thp.core.config import DemandSettings
from thp.core.weeks import last_complete_week_end
from thp.demand.forecast import finish_forecast
from thp.features.demand import SERIES, tenant_week_start
from thp.schema import EXTERNAL_FORECAST, QUANTILE_COLS


class ExternalForecast(NamedTuple):
    forecast: pl.DataFrame
    # Series and horizons the source forecast but that could not be completed.
    skipped: pl.DataFrame


def model_version(source: str) -> str:
    return f"external_{source}"


def external_forecast(
    store: AsOfStore, source: str, baseline: pl.DataFrame, settings: DemandSettings
) -> ExternalForecast:
    last_end = last_complete_week_end(store.as_of, tenant_week_start(store))
    oldest = store.as_of - timedelta(days=settings.external_max_age_days)
    issued = pl.col("issued_date")
    rows = (
        store.facts(EXTERNAL_FORECAST.name)
        .filter((pl.col("source") == source) & (issued >= oldest))
        .collect()
        # The same key ingested twice: the later batch wins.
        .sort("ingested_at")
        .unique(subset=list(EXTERNAL_FORECAST.key), keep="last")
        .filter(issued == issued.max().over(SERIES))
        .with_columns(
            horizon_week=((pl.col("week_end") - pl.lit(last_end)).dt.total_days() // 7).cast(
                pl.Int16
            )
        )
        .filter(pl.col("horizon_week").is_between(1, settings.horizon_weeks))
    )

    spread = baseline.select(
        *SERIES,
        "horizon_week",
        "is_cold_start",
        *(pl.col(col).alias(f"b_{col}") for col in QUANTILE_COLS),
    )
    rows = rows.join(spread, on=[*SERIES, "horizon_week"], how="left")
    center = pl.coalesce("q50", "qty")
    b50 = pl.col("b_q50")
    rows = rows.with_columns(
        pl.coalesce(
            col,
            pl.when(b50 > 0)
            .then(center * pl.col(f"b_{col}") / b50)
            .otherwise(center + pl.col(f"b_{col}") - b50),
        ).alias(col)
        for col in QUANTILE_COLS
    ).with_columns(pl.col("is_cold_start").fill_null(False))

    complete = pl.all_horizontal(pl.col(QUANTILE_COLS).is_not_null())
    skipped = rows.filter(~complete).select(*SERIES, "horizon_week", "issued_date")
    forecast = finish_forecast(rows.filter(complete), store, model_version(source))
    return ExternalForecast(forecast, skipped.sort(*SERIES, "horizon_week"))
