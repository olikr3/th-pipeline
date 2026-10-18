"""Weekly demand features per SKU x location.

``build_demand_features`` returns a panel: one row per series and complete week
up to ``as_of``, where every value on a row uses only data dated on or before
that row's ``week_end``. The last week is the scoring snapshot; the whole panel
is the training set for rolling-origin demand models.

Quantities are kept in units. Scaling by the series level is the model's job.
"""

import math
from datetime import date
from pathlib import Path

import polars as pl

from thp.core.asof import AsOfStore
from thp.core.weeks import last_complete_week_end, week_end

FEATURE_VERSION = "demand_v1"

SERIES = ("sku_id", "location_id")
LAGS = (1, 2, 3, 4, 8, 13, 26, 52)
WINDOWS = (4, 13, 26, 52)
INTERMITTENCY_WEEKS = 52
TREND_WEEKS = 13
PRICE_WEEKS = 13
NEW_SERIES_WEEKS = 13
HIERARCHY_WINDOWS = (4, 13)

# Syntetos-Boylan cut-offs.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49

DEFAULT_WEEK_START = 1  # ISO Monday, when the tenant row does not say


def syntetos_boylan(adi: pl.Expr, cv2: pl.Expr) -> pl.Expr:
    """Demand class from ADI and CV²; null when there is no demand to classify."""
    frequent = adi < ADI_CUTOFF
    stable = cv2 < CV2_CUTOFF
    return (
        pl.when(adi.is_null() | cv2.is_null())
        .then(None)
        .when(frequent & stable)
        .then(pl.lit("smooth"))
        .when(frequent)
        .then(pl.lit("erratic"))
        .when(stable)
        .then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy"))
    )


def build_demand_features(store: AsOfStore) -> pl.DataFrame:
    panel, _ = weekly_demand(store)
    panel = _with_prices(panel, store)
    panel = panel.sort(*SERIES, "week_end")
    panel = _with_series_features(panel)
    panel = _with_reference(panel, store)
    panel = _with_hierarchy(panel).with_columns(tenant_id=pl.lit(store.tenant_id))
    return panel.select(_output_columns()).sort(*SERIES, "week_end")


def write_demand_features(store: AsOfStore, out_root: Path, *, latest_only: bool = False) -> Path:
    """Write the panel (or only its last week) to Parquet and return the path."""
    frame = build_demand_features(store)
    name = "panel"
    if latest_only:
        frame = frame.filter(pl.col("week_end") == pl.col("week_end").max())
        name = "snapshot"
    path = out_root / "demand" / FEATURE_VERSION / store.tenant_id / f"as_of={store.as_of}"
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / f"{name}.parquet")
    return path / f"{name}.parquet"


def weekly_demand(store: AsOfStore) -> tuple[pl.DataFrame, date]:
    """Dense weekly sales per series up to the last complete week, and that week's end."""
    week_start = tenant_week_start(store)
    last_end = last_complete_week_end(store.as_of, week_start)
    return _dense_weeks(_weekly_sales(store, week_start, last_end), last_end), last_end


def tenant_week_start(store: AsOfStore) -> int:
    rows = store.reference("tenant").select("week_start").collect()["week_start"]
    value = rows[0] if len(rows) else None
    return DEFAULT_WEEK_START if value is None else int(value)


def _weekly_sales(store: AsOfStore, week_start: int, last_end: date) -> pl.DataFrame:
    discount = pl.col("discount")
    return (
        store.facts("sales")
        .filter(pl.col("date") <= last_end)
        .with_columns(week_end=week_end(pl.col("date"), week_start))
        .group_by(*SERIES, "week_end")
        .agg(
            qty=pl.col("qty").sum(),
            net_revenue=pl.col("net_revenue").sum(),
            # Null when the source has no discount data, rather than a fake 0.
            discount=pl.when(discount.is_not_null().any()).then(discount.sum()),
        )
        .collect()
    )


def _dense_weeks(weekly: pl.DataFrame, last_end: date) -> pl.DataFrame:
    """One row per series and week from its first sale on; weeks without sales are 0."""
    grid = (
        weekly.group_by(SERIES)
        .agg(first=pl.col("week_end").min())
        .select(
            *SERIES,
            week_end=pl.date_ranges("first", pl.lit(last_end), interval="1w"),
        )
        .explode("week_end")
    )
    return grid.join(weekly, on=[*SERIES, "week_end"], how="left").with_columns(
        pl.col("qty", "net_revenue").fill_null(0.0)
    )


def _with_prices(panel: pl.DataFrame, store: AsOfStore) -> pl.DataFrame:
    prices = (
        store.facts("price")
        .select(*SERIES, "valid_from", price="list_price")
        .collect()
        .sort("valid_from")
    )
    return panel.sort("week_end").join_asof(
        prices, left_on="week_end", right_on="valid_from", by=SERIES, check_sortedness=False
    )


def _roll_sum(col: str, window: int, min_samples: int = 1) -> pl.Expr:
    return pl.col(col).rolling_sum(window, min_samples=min_samples).over(SERIES)


def _with_series_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Lags, rolling stats, intermittency, trend, price and calendar. Needs sorted series."""
    qty = pl.col("qty")
    panel = panel.with_columns(
        _i=pl.int_range(pl.len()).over(SERIES).cast(pl.Float64),
        _nonzero=(qty > 0).cast(pl.Float64),
        _zero=(qty == 0).cast(pl.Float64),
        _qty_sq=qty * qty,
    ).with_columns(_iq=pl.col("_i") * qty, _i_sq=pl.col("_i") * pl.col("_i"))

    lags = [qty.shift(lag - 1).over(SERIES).alias(f"qty_lag_{lag}") for lag in LAGS]
    rolling = [
        expr
        for w in WINDOWS
        for expr in (
            qty.rolling_mean(w, min_samples=1).over(SERIES).alias(f"qty_mean_{w}"),
            qty.rolling_std(w, min_samples=2).over(SERIES).alias(f"qty_std_{w}"),
            qty.rolling_median(w, min_samples=1).over(SERIES).alias(f"qty_median_{w}"),
            qty.rolling_max(w, min_samples=1).over(SERIES).alias(f"qty_max_{w}"),
            pl.col("_zero").rolling_mean(w, min_samples=1).over(SERIES).alias(f"zero_share_{w}"),
        )
    ]

    # Intermittency over the window, from rolling sums: zero weeks add nothing to
    # the sums of demand sizes, so these are the moments of the non-zero weeks.
    w = INTERMITTENCY_WEEKS
    n_nonzero = _roll_sum("_nonzero", w)
    span = pl.min_horizontal(pl.col("_i") + 1, pl.lit(float(w)))
    size_mean = _roll_sum("qty", w) / n_nonzero
    size_var = _roll_sum("_qty_sq", w) / n_nonzero - size_mean * size_mean
    has_demand = n_nonzero > 0
    intermittency = [
        pl.when(has_demand).then(span / n_nonzero).alias("adi"),
        pl.when(has_demand)
        .then(pl.max_horizontal(size_var, pl.lit(0.0)) / (size_mean * size_mean))
        .alias("cv2"),
    ]

    # Least-squares slope over the trailing window, against the week index.
    t = TREND_WEEKS
    n = pl.min_horizontal(pl.col("_i") + 1, pl.lit(float(t)))
    sx, sy = _roll_sum("_i", t), _roll_sum("qty", t)
    sxy, sxx = _roll_sum("_iq", t), _roll_sum("_i_sq", t)
    denom = n * sxx - sx * sx
    trend = pl.when(denom > 0).then((n * sxy - sx * sy) / denom).alias(f"trend_slope_{t}")

    price = pl.col("price")
    price_mean = price.rolling_mean(PRICE_WEEKS, min_samples=1).over(SERIES)
    revenue_gross = pl.col("net_revenue") + pl.col("discount")
    angle = 2 * math.pi * pl.col("week_end").dt.ordinal_day() / 365.25
    other = [
        (price / price_mean).alias(f"price_ratio_{PRICE_WEEKS}"),
        (pl.col("week_end") - pl.col("valid_from"))
        .dt.total_days()
        .alias("days_since_price_change"),
        pl.when(revenue_gross > 0).then(pl.col("discount") / revenue_gross).alias("discount_depth"),
        angle.sin().alias("woy_sin"),
        angle.cos().alias("woy_cos"),
        (pl.col("_i").cast(pl.Int32) + 1).alias("history_weeks"),
    ]

    panel = panel.with_columns(*lags, *rolling, *intermittency, trend, *other)
    return panel.with_columns(
        syntetos_boylan(pl.col("adi"), pl.col("cv2")).alias("demand_class"),
        (pl.col("qty_mean_13") / pl.col("qty_mean_52")).alias("mean_ratio_13_52"),
        (pl.col("history_weeks") < NEW_SERIES_WEEKS).alias("is_new"),
    )


def _with_reference(panel: pl.DataFrame, store: AsOfStore) -> pl.DataFrame:
    # Only static attributes: reference tables hold current state, so a field like
    # lifecycle_status ("discontinued") would leak the future into old rows.
    sku = store.reference("sku").select("sku_id", "category_l1", "launch_date").collect()
    return panel.join(sku, on="sku_id", how="left").with_columns(
        weeks_since_launch=(pl.col("week_end") - pl.col("launch_date")).dt.total_days() // 7
    )


def _with_hierarchy(panel: pl.DataFrame) -> pl.DataFrame:
    """Rolling means of the average series in the category x location and the tenant."""
    levels = {"cat": ("category_l1", "location_id"), "tenant": ()}
    for prefix, keys in levels.items():
        weekly = (
            panel.group_by(*keys, "week_end").agg(_avg=pl.col("qty").mean()).sort(*keys, "week_end")
        )
        rolled = [
            (
                pl.col("_avg").rolling_mean(w, min_samples=1).over(keys)
                if keys
                else pl.col("_avg").rolling_mean(w, min_samples=1)
            ).alias(f"{prefix}_qty_mean_{w}")
            for w in HIERARCHY_WINDOWS
        ]
        weekly = weekly.with_columns(rolled).drop("_avg")
        panel = panel.join(weekly, on=[*keys, "week_end"], how="left", nulls_equal=True)
    return panel


def _output_columns() -> list[str]:
    rolling = [
        f"{stat}_{w}"
        for w in WINDOWS
        for stat in ("qty_mean", "qty_std", "qty_median", "qty_max", "zero_share")
    ]
    hierarchy = [f"{p}_qty_mean_{w}" for p in ("cat", "tenant") for w in HIERARCHY_WINDOWS]
    return [
        "tenant_id",
        *SERIES,
        "week_end",
        *(f"qty_lag_{lag}" for lag in LAGS),
        *rolling,
        "adi",
        "cv2",
        "demand_class",
        f"trend_slope_{TREND_WEEKS}",
        "mean_ratio_13_52",
        "woy_sin",
        "woy_cos",
        "price",
        f"price_ratio_{PRICE_WEEKS}",
        "days_since_price_change",
        "discount_depth",
        "history_weeks",
        "is_new",
        "weeks_since_launch",
        "category_l1",
        *hierarchy,
    ]
