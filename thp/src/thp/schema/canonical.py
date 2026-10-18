"""Canonical tables: column types and keys every source maps into."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: dict[str, pl.DataType]
    key: tuple[str, ...]

    def conform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Cast to the canonical types and column order.

        Key columns must be present; other missing columns become null. Extra
        columns are dropped.
        """
        missing_keys = [col for col in self.key if col not in frame.columns]
        if missing_keys:
            raise ValueError(f"{self.name}: missing key columns {missing_keys}")
        return frame.select(
            pl.col(col).cast(dtype)
            if col in frame.columns
            else pl.lit(None, dtype=dtype).alias(col)
            for col, dtype in self.columns.items()
        )

    def validate(self, frame: pl.DataFrame) -> None:
        """Raise if any key column is null or any key occurs twice."""
        keys = frame.select(self.key)
        null_keys = [col for col in self.key if keys[col].null_count() > 0]
        if null_keys:
            raise ValueError(f"{self.name}: null values in key columns {null_keys}")
        duplicates = keys.height - keys.n_unique()
        if duplicates:
            raise ValueError(f"{self.name}: {duplicates} rows repeat an existing key {self.key}")


_STR = pl.String()
_DATE = pl.Date()
_NUM = pl.Float64()
_TS = pl.Datetime("us")

TENANT = TableSchema(
    name="tenant",
    columns={
        "tenant_id": _STR,
        "name": _STR,
        "currency": _STR,
        "timezone": _STR,
        "week_start": pl.Int8(),  # ISO weekday, 1 = Monday
        "fiscal_year_start": _DATE,
        "created_at": _TS,
    },
    key=("tenant_id",),
)

LOCATION = TableSchema(
    name="location",
    columns={
        "tenant_id": _STR,
        "location_id": _STR,
        "name": _STR,
        "type": _STR,
        "country": _STR,
    },
    key=("tenant_id", "location_id"),
)

SKU = TableSchema(
    name="sku",
    columns={
        "tenant_id": _STR,
        "sku_id": _STR,
        "name": _STR,
        "category_l1": _STR,
        "category_l2": _STR,
        "category_l3": _STR,
        "brand": _STR,
        "supplier_id": _STR,
        "uom": _STR,
        "pack_size": _NUM,
        "lifecycle_status": _STR,
        "launch_date": _DATE,
        "is_perishable": pl.Boolean(),
        "shelf_life_days": pl.Int32(),
    },
    key=("tenant_id", "sku_id"),
)

SALES = TableSchema(
    name="sales",
    columns={
        "tenant_id": _STR,
        "sku_id": _STR,
        "location_id": _STR,
        "date": _DATE,
        "channel": _STR,
        "qty": _NUM,
        "net_revenue": _NUM,
        "discount": _NUM,
        "returned_qty": _NUM,
        "customer_type": _STR,
        "ingested_at": _TS,
    },
    key=("tenant_id", "sku_id", "location_id", "date", "channel"),
)

# Keyed by location as well: prices can differ per store.
PRICE = TableSchema(
    name="price",
    columns={
        "tenant_id": _STR,
        "sku_id": _STR,
        "location_id": _STR,
        "valid_from": _DATE,
        "list_price": _NUM,
        "unit_cost": _NUM,
        "currency": _STR,
        "ingested_at": _TS,
    },
    key=("tenant_id", "sku_id", "location_id", "valid_from"),
)

# Demand quantiles every forecast carries, and their column names.
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QUANTILE_COLS = tuple(f"q{round(q * 100):02d}" for q in QUANTILES)

# Forecasts from another source (a merchant's planning tool, an ERP), as issued.
# A fact dated by issued_date: an as_of before the issue cannot see it. Any of
# qty (the source's point forecast) and the quantiles may be null, but each row
# needs qty or q50.
EXTERNAL_FORECAST = TableSchema(
    name="external_forecast",
    columns={
        "tenant_id": _STR,
        "source": _STR,
        "sku_id": _STR,
        "location_id": _STR,
        "issued_date": _DATE,
        "week_end": _DATE,  # last day of the target week, in the tenant's weeks
        "qty": _NUM,
        **{col: _NUM for col in QUANTILE_COLS},
        "ingested_at": _TS,
    },
    key=("tenant_id", "source", "sku_id", "location_id", "issued_date", "week_end"),
)

# Written by the engine: demand quantiles per series for the weeks after as_of's
# last complete week (horizon_week 1 ends 7 days after it).
FORECAST = TableSchema(
    name="forecast",
    columns={
        "tenant_id": _STR,
        "model_version": _STR,
        "sku_id": _STR,
        "location_id": _STR,
        "as_of": _DATE,
        "horizon_week": pl.Int16(),
        "week_end": _DATE,
        **{col: _NUM for col in QUANTILE_COLS},
        "is_cold_start": pl.Boolean(),
    },
    key=("tenant_id", "model_version", "sku_id", "location_id", "as_of", "horizon_week"),
)
