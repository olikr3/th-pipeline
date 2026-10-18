"""The forecast table every demand source produces, and where it is written."""

from pathlib import Path

import polars as pl

from thp.core.asof import AsOfStore
from thp.schema import FORECAST, QUANTILE_COLS


def finish_forecast(frame: pl.DataFrame, store: AsOfStore, model_version: str) -> pl.DataFrame:
    """Clip quantiles at zero, sort them so they cannot cross, and conform to ``FORECAST``.

    ``frame`` needs the series, ``horizon_week``, ``week_end``, the quantile columns
    and ``is_cold_start``.
    """
    ordered = pl.concat_list(pl.col(col).clip(lower_bound=0.0) for col in QUANTILE_COLS).list.sort()
    frame = frame.with_columns(
        ordered.list.to_struct(fields=list(QUANTILE_COLS)).alias("_q"),
        tenant_id=pl.lit(store.tenant_id),
        model_version=pl.lit(model_version),
        as_of=pl.lit(store.as_of),
    ).drop(QUANTILE_COLS)
    frame = FORECAST.conform(frame.unnest("_q"))
    FORECAST.validate(frame)
    return frame.sort("sku_id", "location_id", "horizon_week")


def write_forecast(frame: pl.DataFrame, out_root: Path) -> Path:
    """Write one model version's forecast for one tenant and as_of; return the path."""
    (model_version,) = frame["model_version"].unique()
    (tenant,) = frame["tenant_id"].unique()
    (as_of,) = frame["as_of"].unique()
    path = out_root / FORECAST.name / str(model_version) / str(tenant) / f"as_of={as_of}"
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / "forecast.parquet")
    return path / "forecast.parquet"
