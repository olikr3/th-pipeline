"""From a lot request and the canonical tables to a ``Lot`` the engine can simulate.

Everything is read through the as-of store, so a lot scored at a past as_of sees
only what was known then:

- **Price:** the latest list price on or before as_of, per SKU x location.
- **Category:** ``sku.category_l1`` (``"unknown"`` when missing).
- **Return rate:** returned / sold units per category over the 52 weeks to as_of.
- **Stock ahead of the lot:** from the latest stock snapshot (at most
  ``MAX_SNAPSHOT_AGE_DAYS`` old) and open purchase orders. A PO line is open at
  as_of when it was ordered by then and not received by then; its status and a
  receipt after as_of are the future and are ignored. For a purchase-order lot,
  stock ahead is on hand plus open POs expected no later than the lot. For a
  stock-financing lot, the lot is the youngest part of on-hand stock, so stock
  ahead is on hand minus the lot.
- **Forecast:** knots for every week the longest tenor needs; missing weeks
  refuse the lot (``MissingForecastError``).
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from thp.core.asof import AsOfStore
from thp.core.weeks import last_complete_week_end
from thp.features.demand import tenant_week_start
from thp.simulate.engine import DEFAULT_TENORS, Lot, weeks_needed
from thp.simulate.inputs import forecast_knots, series_key

MAX_SNAPSHOT_AGE_DAYS = 14
RETURN_WEEKS = 52
UNKNOWN_CATEGORY = "unknown"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LotLine(_Model):
    sku_id: str
    location_id: str
    qty: float = Field(gt=0)
    unit_cost: float = Field(gt=0)


class LotOptions(_Model):
    n_paths: int | None = Field(default=None, gt=0)
    seed: int = 42
    confidence: float | None = Field(default=None, gt=0, lt=1)


class LotRequest(_Model):
    """A purchase order or stock position to score; mirrors the spec's API request."""

    name: str | None = None
    description: str | None = None
    as_of: date
    # When the money goes out; the tenor counts from here. Defaults to as_of.
    disbursement_date: date | None = None
    # None: the goods are on hand (stock financing).
    expected_receipt_date: date | None = None
    tenor_days: int = Field(default=90, gt=0)
    fee_rate: float = Field(default=0.03, ge=0)
    lines: list[LotLine] = Field(min_length=1)
    options: LotOptions = LotOptions()

    @model_validator(mode="after")
    def _one_line_per_series(self) -> "LotRequest":
        keys = [(line.sku_id, line.location_id) for line in self.lines]
        if len(set(keys)) != len(keys):
            raise ValueError("each SKU x location may appear in only one line")
        return self

    @property
    def disbursed(self) -> date:
        return self.disbursement_date or self.as_of

    @property
    def tenors(self) -> tuple[int, ...]:
        return tuple(sorted({*DEFAULT_TENORS, self.tenor_days}))


def assemble_lot(store: AsOfStore, request: LotRequest, forecast: pl.DataFrame) -> Lot:
    if store.as_of != request.as_of:
        raise ValueError(f"store is as of {store.as_of}, the request as of {request.as_of}")
    pairs = [(line.sku_id, line.location_id) for line in request.lines]
    lines = pl.DataFrame(
        [line.model_dump() for line in request.lines],
        schema={
            "sku_id": pl.String,
            "location_id": pl.String,
            "qty": pl.Float64,
            "unit_cost": pl.Float64,
        },
    )
    horizon_start = last_complete_week_end(request.as_of, tenant_week_start(store)) + timedelta(1)

    lines = lines.join(_prices(store, lines), on=["sku_id", "location_id"], how="left")
    no_price = lines.filter(pl.col("price").is_null())
    if no_price.height:
        raise ValueError(f"no list price on or before {request.as_of} for {_list(no_price)}")
    lines = lines.join(_categories(store), on="sku_id", how="left").with_columns(
        pl.col("category").fill_null(UNKNOWN_CATEGORY)
    )
    lines = lines.join(_return_rates(store), on="category", how="left").with_columns(
        pl.col("return_rate").fill_null(0.0)
    )
    lines = lines.join(
        _stock_ahead(store, lines, request), on=["sku_id", "location_id"], how="left"
    )

    n_weeks = weeks_needed(request.disbursed, horizon_start, max(request.tenors))

    def column(name: str) -> np.ndarray:
        return lines[name].to_numpy().astype(np.float32)

    return Lot(
        as_of=request.as_of,
        horizon_start=horizon_start,
        disbursement_date=request.disbursed,
        series=tuple(series_key(sku, loc) for sku, loc in pairs),
        categories=tuple(lines["category"].to_list()),
        qty=column("qty"),
        unit_cost=column("unit_cost"),
        price=column("price"),
        stock_ahead=column("stock_ahead"),
        return_rate=column("return_rate"),
        knots=forecast_knots(forecast, pairs, n_weeks),
        expected_receipt_date=request.expected_receipt_date,
        tenors_days=request.tenors,
    )


def _prices(store: AsOfStore, lines: pl.DataFrame) -> pl.DataFrame:
    return (
        store.facts("price")
        .join(lines.lazy().select("sku_id", "location_id"), on=["sku_id", "location_id"])
        .sort("valid_from")
        .group_by("sku_id", "location_id")
        .agg(price=pl.col("list_price").last())
        .collect()
    )


def _categories(store: AsOfStore) -> pl.DataFrame:
    return store.reference("sku").select("sku_id", category="category_l1").collect()


def _return_rates(store: AsOfStore) -> pl.DataFrame:
    since = store.as_of - timedelta(weeks=RETURN_WEEKS)
    return (
        store.facts("sales")
        .filter(pl.col("date") > since)
        .join(store.reference("sku").select("sku_id", category="category_l1"), on="sku_id")
        .group_by("category")
        .agg(
            return_rate=pl.when(pl.col("qty").sum() > 0).then(
                pl.col("returned_qty").fill_null(0).sum() / pl.col("qty").sum()
            )
        )
        .collect()
    )


def _stock_ahead(store: AsOfStore, lines: pl.DataFrame, request: LotRequest) -> pl.DataFrame:
    keys = lines.select("sku_id", "location_id")
    snapshots = (
        store.facts("stock_snapshot")
        .join(keys.lazy(), on=["sku_id", "location_id"])
        .sort("snapshot_date")
        .group_by("sku_id", "location_id")
        .agg(pl.col("snapshot_date").last(), on_hand=pl.col("on_hand_qty").last())
        .collect()
    )
    stale = snapshots.filter(
        pl.col("snapshot_date") < request.as_of - timedelta(days=MAX_SNAPSHOT_AGE_DAYS)
    )
    if stale.height:
        raise ValueError(
            f"latest stock snapshot older than {MAX_SNAPSHOT_AGE_DAYS} days for {_list(stale)}"
        )
    stock = keys.join(snapshots, on=["sku_id", "location_id"], how="left").with_columns(
        pl.col("on_hand").fill_null(0.0)
    )

    if request.expected_receipt_date is None:  # stock financing: the lot is on hand
        stock = stock.join(
            lines.select("sku_id", "location_id", "qty"), on=["sku_id", "location_id"]
        )
        short = stock.filter(pl.col("qty") > pl.col("on_hand"))
        if short.height:
            raise ValueError(f"lot qty exceeds stock on hand at {request.as_of} for {_list(short)}")
        return stock.select("sku_id", "location_id", stock_ahead=pl.col("on_hand") - pl.col("qty"))

    received = pl.col("actual_receipt_date")
    open_ahead = (
        store.facts("purchase_order")
        .join(keys.lazy(), on=["sku_id", "location_id"])
        .filter(
            (received.is_null() | (received > request.as_of))
            & (pl.col("expected_receipt_date") <= request.expected_receipt_date)
        )
        .group_by("sku_id", "location_id")
        .agg(on_order=pl.col("qty").sum())
        .collect()
    )
    return (
        stock.join(open_ahead, on=["sku_id", "location_id"], how="left")
        .with_columns(pl.col("on_order").fill_null(0.0))
        .select("sku_id", "location_id", stock_ahead=pl.col("on_hand") + pl.col("on_order"))
    )


def _list(frame: pl.DataFrame) -> str:
    return ", ".join(series_key(s, loc) for s, loc in frame.select("sku_id", "location_id").rows())
