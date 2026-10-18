"""Forecasts from another source -> canonical ``external_forecast`` Parquet.

The input is a CSV or Parquet file with one row per series and target week:

- ``sku_id``, ``location_id``
- ``week``: any date in the target week; stored as that week's end in the tenant's weeks
- ``qty`` (the source's point forecast) and/or quantile columns ``q05`` ... ``q95``
- ``issued_date`` (optional): when the source made the forecast; otherwise the
  ``issued_date`` argument applies to the whole file

Rejects the file when a row has neither ``qty`` nor ``q50``, a value is negative,
the given quantiles cross, or a series repeats a target week.
"""

import re
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from thp.core.weeks import week_end
from thp.schema import EXTERNAL_FORECAST, QUANTILE_COLS

SOURCE_NAME = re.compile(r"[a-z0-9_]+")
VALUE_COLS = ("qty", *QUANTILE_COLS)


def read_external_forecasts(
    path: Path,
    *,
    tenant_id: str,
    source: str,
    week_start: int,
    issued_date: date | None = None,
    ingested_at: datetime | None = None,
) -> pl.DataFrame:
    if not SOURCE_NAME.fullmatch(source):
        raise ValueError(f"source must match {SOURCE_NAME.pattern}, got {source!r}")
    if path.suffix == ".parquet":
        raw = pl.read_parquet(path)
    else:
        raw = pl.read_csv(path, try_parse_dates=True)

    missing = [col for col in ("sku_id", "location_id", "week") if col not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    values = [col for col in VALUE_COLS if col in raw.columns]
    if "qty" not in values and "q50" not in values:
        raise ValueError(f"{path.name}: needs a qty or q50 column")
    if "issued_date" not in raw.columns:
        if issued_date is None:
            raise ValueError(f"{path.name}: no issued_date column, so pass issued_date")
        raw = raw.with_columns(issued_date=pl.lit(issued_date))

    ingested_at = ingested_at or datetime.now(UTC).replace(tzinfo=None)
    frame = raw.with_columns(
        pl.col("sku_id", "location_id").cast(pl.String),
        pl.col("issued_date").cast(pl.Date),
        pl.col(values).cast(pl.Float64),
        week_end=week_end(pl.col("week").cast(pl.Date), week_start),
        tenant_id=pl.lit(tenant_id),
        source=pl.lit(source),
        ingested_at=pl.lit(ingested_at),
    )
    frame = EXTERNAL_FORECAST.conform(frame)
    EXTERNAL_FORECAST.validate(frame)
    _check_values(frame, path.name)
    return frame


def write_external_forecasts(frame: pl.DataFrame, out_root: Path) -> Path:
    """Append one ingest batch; on read, a later batch wins for the same key."""
    (tenant,) = frame["tenant_id"].unique()
    (source,) = frame["source"].unique()
    (ingested_at,) = frame["ingested_at"].unique()
    path = out_root / EXTERNAL_FORECAST.name / str(tenant) / str(source)
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{ingested_at:%Y%m%dT%H%M%S%f}.parquet"
    frame.write_parquet(path)
    return path


def _check_values(frame: pl.DataFrame, name: str) -> None:
    no_center = frame.filter(pl.col("qty").is_null() & pl.col("q50").is_null()).height
    if no_center:
        raise ValueError(f"{name}: {no_center} rows have neither qty nor q50")
    negative = frame.filter(pl.any_horizontal(pl.col(VALUE_COLS) < 0)).height
    if negative:
        raise ValueError(f"{name}: {negative} rows have negative forecasts")
    given = pl.concat_list(QUANTILE_COLS).list.drop_nulls()
    crossing = frame.filter(given != given.list.sort()).height
    if crossing:
        raise ValueError(f"{name}: {crossing} rows have quantiles that decrease")
