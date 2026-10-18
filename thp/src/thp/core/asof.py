"""As-of data access: the only way the engine reads fact tables.

Every scan is filtered to rows whose event date is on or before ``as_of``, so
features, forecasts and scores cannot see the future by construction.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

INGESTED_COL = "ingested_at"
TENANT_COL = "tenant_id"

# Fact table -> the column that dates the event.
FACT_EVENT_COLS: dict[str, str] = {
    "sales": "date",
    "price": "valid_from",
    "stock_snapshot": "snapshot_date",
    "purchase_order": "order_date",
    "external_forecast": "issued_date",
}

REFERENCE_TABLES: frozenset[str] = frozenset({"tenant", "location", "sku", "supplier"})


@dataclass(frozen=True)
class AsOfStore:
    """Reads canonical tables stored as ``{root}/{table}/**/*.parquet`` for one tenant.

    ``ingest_grace`` bounds how late a row may have been ingested and still be
    visible: rows need ``ingested_at`` no later than the end of ``as_of`` plus the
    grace. ``None`` disables that filter, which is what backfilled history (M5,
    a merchant's first upload) needs, since all of it is ingested "today".
    """

    root: Path
    tenant_id: str
    as_of: date
    ingest_grace: timedelta | None = None

    def facts(self, table: str) -> pl.LazyFrame:
        try:
            event_col = FACT_EVENT_COLS[table]
        except KeyError:
            raise KeyError(f"unknown fact table {table!r}") from None

        visible = pl.col(event_col) <= self.as_of
        if self.ingest_grace is not None:
            cutoff = datetime.combine(self.as_of + timedelta(days=1), time.min) + self.ingest_grace
            visible &= pl.col(INGESTED_COL) < cutoff
        return self._scan(table).filter(visible)

    def reference(self, table: str) -> pl.LazyFrame:
        if table not in REFERENCE_TABLES:
            raise KeyError(f"unknown reference table {table!r}")
        return self._scan(table)

    def _scan(self, table: str) -> pl.LazyFrame:
        frame = pl.scan_parquet(self.root / table / "**" / "*.parquet")
        return frame.filter(pl.col(TENANT_COL) == self.tenant_id)
