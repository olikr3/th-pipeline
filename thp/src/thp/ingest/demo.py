"""Synthetic demo tenant: every canonical table, from a known generating process.

M5 has no stock or purchase orders, and real merchant data is not available yet.
This tenant fills that gap for demos and integration tests. Day by day, per SKU x
location:

- **Demand** is negative binomial around a level set by the SKU's base rate, its
  location, a yearly seasonal curve per category, a growth trend, price changes
  (elasticity -1.5), and weekly merchant and category shocks (AR(1)), so SKUs
  have good and bad weeks together. Some SKUs launch during the history; some are
  discontinued, after which demand falls to a tenth and ordering stops.
- **Inventory** follows an order-up-to buyer: every two weeks it orders up to
  ``rate x (lead time + review period) x k``, in packs of 6. The rate is the mean
  of the last eight weeks' sales on days the SKU was in stock (so stock-outs do
  not shrink the next order), scaled by the known seasonal curve from now to the
  coming lead time. The bias k depends on the buyer profile (disciplined,
  average, overbuyer), so some SKUs build excess stock. Lead times are lognormal
  per supplier.
- **Sales** are min(demand, stock): stock-outs censor demand, as in real data.
  Returns are binomial per sale at the category's return rate.

It writes ``tenant``, ``location``, ``sku``, ``sales``, ``price``,
``stock_snapshot`` (Sundays) and ``purchase_order``. Receipts after the last day
are unknown: those lines are ``open`` with a null ``actual_receipt_date``.
"""

import math
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thp.core.asof import AsOfStore
from thp.core.seeding import make_rng
from thp.schema import (
    LOCATION,
    PRICE,
    PURCHASE_ORDER,
    SALES,
    SKU,
    STOCK_SNAPSHOT,
    TENANT,
    TableSchema,
)


@dataclass(frozen=True)
class Category:
    peak_day: int  # day of year of the seasonal peak
    amplitude: float  # log-scale seasonal swing
    return_rate: float
    margin: float  # gross margin: unit_cost = price x (1 - margin)
    supplier: str
    lead_days: int  # median lead time
    lead_sigma: float  # lognormal spread of the lead time


CATEGORIES = {
    "garden": Category(165, 0.9, 0.03, 0.45, "SUP-GARDEN", 42, 0.25),
    "kitchen": Category(0, 0.1, 0.05, 0.40, "SUP-KITCHEN", 35, 0.20),
    "toys": Category(340, 1.0, 0.08, 0.50, "SUP-TOYS", 56, 0.30),
}
# name: (description, type, demand factor, sales channel, share of SKUs stocked)
LOCATIONS = {
    "WH1": ("Online warehouse", "warehouse", 1.0, "online", 1.0),
    "ST1": ("City store", "store", 0.5, "store", 0.6),
}
# Order-up-to bias k per buyer profile, and the profile's share of SKUs.
BUYERS = {"disciplined": (1.1, 0.6), "average": (1.4, 0.25), "overbuyer": (2.2, 0.15)}
REVIEW_DAYS = 14
PACK = 6
ELASTICITY = -1.5
SHOCK_SD = 0.15  # weekly log-scale shocks
SHOCK_PERSISTENCE = 0.8
NB_SHAPE = 2.0  # negative-binomial shape: lower is more overdispersed
CURRENCY = "EUR"

# Demo lots, written next to the data.
PO_LEAD_DAYS = 42
LOT_LINES = 12


@dataclass(frozen=True)
class DemoSpec:
    tenant_id: str = "demo"
    end: date = date(2026, 9, 27)  # a Sunday: the last day of history
    weeks: int = 156
    n_skus: int = 40
    seed: int = 7


@dataclass(frozen=True)
class _Sku:
    sku_id: str
    category: str
    base: float  # mean daily demand at the main location, before seasonality
    growth: float  # yearly log growth
    launch: int  # day index of launch (may be <= 0: already selling)
    discontinue: int  # day index of discontinuation (n_days: never)
    price: float  # list price at launch
    buyer: str
    locations: tuple[str, ...]


def build_demo(
    out_root: Path, spec: DemoSpec | None = None, *, ingested_at: datetime | None = None
) -> dict[str, int]:
    """Write the demo tenant's canonical tables; return rows per table.

    Replaces whatever a previous run wrote for this tenant.
    """
    spec = spec or DemoSpec()
    if spec.end.isoweekday() != 7:
        raise ValueError("spec.end must be a Sunday, so history ends on a complete week")
    ingested_at = ingested_at or datetime.now(UTC).replace(tzinfo=None)
    rng = make_rng(spec.seed)
    n_days = spec.weeks * 7
    start = spec.end - timedelta(days=n_days - 1)
    days = np.arange(n_days)
    dates = [start + timedelta(days=int(d)) for d in days]

    skus = _skus(spec, rng, n_days)
    merchant = _daily_shocks(rng, spec.weeks)
    category_shock = {name: _daily_shocks(rng, spec.weeks) for name in CATEGORIES}
    # Past the last day too: the buyer looks ahead by the lead time.
    horizon = n_days + 120
    day_of_year = np.array(
        [(start + timedelta(int(d))).timetuple().tm_yday for d in range(horizon)]
    )

    sales: list[dict[str, Any]] = []
    prices: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    po_lines: list[dict[str, Any]] = []
    for sku in skus:
        cat = CATEGORIES[sku.category]
        price = _price_path(rng, sku, n_days)
        for loc in sku.locations:
            _, _, factor, channel, _ = LOCATIONS[loc]
            season = np.exp(cat.amplitude * np.cos(2 * np.pi * (day_of_year - cat.peak_day) / 365))
            season /= np.exp(cat.amplitude * np.cos(2 * np.pi * np.arange(365) / 365)).mean()
            level = (
                sku.base
                * factor
                * season[:n_days]
                * np.exp(sku.growth * days / 365)
                * (price / sku.price) ** ELASTICITY
                * np.exp(merchant + category_shock[sku.category])
            )
            level = np.where(days < sku.launch, 0.0, level)
            level = np.where(days >= sku.discontinue, 0.1 * level, level)
            demand = rng.poisson(level * rng.gamma(NB_SHAPE, 1 / NB_SHAPE, n_days))
            planned = sku.base * factor  # what the buyer expects from a new SKU

            run = _run_inventory(rng, sku, cat, demand, planned, season, n_days)
            for d, qty, returned in run.sales:
                sales.append(
                    {
                        "sku_id": sku.sku_id,
                        "location_id": loc,
                        "date": dates[d],
                        "channel": channel,
                        "qty": float(qty),
                        "net_revenue": float(qty * price[d]),
                        "returned_qty": float(returned),
                        "customer_type": "b2c",
                    }
                )
            for d, on_hand, on_order, oldest in run.snapshots:
                snapshots.append(
                    {
                        "sku_id": sku.sku_id,
                        "location_id": loc,
                        "snapshot_date": dates[d],
                        "on_hand_qty": float(on_hand),
                        "on_order_qty": float(on_order),
                        "reserved_qty": 0.0,
                        "unit_cost": float(price[d] * (1 - cat.margin)),
                        "oldest_receipt_date": None
                        if oldest is None
                        else start + timedelta(oldest),
                    }
                )
            for ordered, expected, actual, qty in run.orders:
                arrived = actual < n_days
                po_lines.append(
                    {
                        "sku_id": sku.sku_id,
                        "location_id": loc,
                        "supplier_id": cat.supplier,
                        "order_date": start + timedelta(ordered),
                        "expected_receipt_date": start + timedelta(expected),
                        "actual_receipt_date": start + timedelta(actual) if arrived else None,
                        "qty": float(qty),
                        "unit_cost": float(price[max(ordered, 0)] * (1 - cat.margin)),
                        "status": "received" if arrived else "open",
                    }
                )
            changes = np.flatnonzero(np.diff(price, prepend=-1.0))
            prices += [
                {
                    "sku_id": sku.sku_id,
                    "location_id": loc,
                    "valid_from": dates[d],
                    "list_price": float(price[d]),
                    "unit_cost": float(price[d] * (1 - cat.margin)),
                    "currency": CURRENCY,
                }
                for d in changes
            ]

    tables: list[tuple[TableSchema, list[dict[str, Any]] | pl.DataFrame]] = [
        (
            TENANT,
            [
                {
                    "tenant_id": spec.tenant_id,
                    "name": "Demo merchant",
                    "currency": CURRENCY,
                    "timezone": "Europe/Berlin",
                    "week_start": 1,
                    "created_at": ingested_at,
                }
            ],
        ),
        (
            LOCATION,
            [
                {"location_id": loc, "name": name, "type": kind, "country": "DE"}
                for loc, (name, kind, *_) in LOCATIONS.items()
            ],
        ),
        (
            SKU,
            [
                {
                    "sku_id": s.sku_id,
                    "name": f"{s.category.title()} item {s.sku_id[-3:]}",
                    "category_l1": s.category,
                    "supplier_id": CATEGORIES[s.category].supplier,
                    "uom": "piece",
                    "pack_size": float(PACK),
                    "lifecycle_status": "discontinued" if s.discontinue < n_days else "active",
                    "launch_date": start + timedelta(max(s.launch, 0)),
                    "is_perishable": False,
                }
                for s in skus
            ],
        ),
        (SALES, sales),
        (PRICE, prices),
        (STOCK_SNAPSHOT, snapshots),
        (PURCHASE_ORDER, _number_po_lines(po_lines)),
    ]
    counts = {}
    for schema, rows in tables:
        shutil.rmtree(out_root / schema.name / spec.tenant_id, ignore_errors=True)
        frame = (
            rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows, infer_schema_length=None)
        )
        frame = frame.with_columns(tenant_id=pl.lit(spec.tenant_id))
        if "ingested_at" in schema.columns:
            frame = frame.with_columns(ingested_at=pl.lit(ingested_at))
        counts[schema.name] = _write(out_root, schema, spec.tenant_id, frame)
    return counts


@dataclass
class _Run:
    sales: list[tuple[int, int, int]]  # day, qty, returned
    snapshots: list[tuple[int, int, int, int | None]]  # day, on hand, on order, oldest receipt
    orders: list[tuple[int, int, int, int]]  # ordered, expected, actual, qty (day indices)


def _run_inventory(
    rng: np.random.Generator,
    sku: _Sku,
    cat: Category,
    demand: np.ndarray,
    planned: float,
    season: np.ndarray,
    n_days: int,
) -> _Run:
    k = BUYERS[sku.buyer][0]
    run = _Run([], [], [])
    layers: list[list[int]] = []  # FIFO: [receipt day, units left]
    incoming: dict[int, int] = {}  # arrival day -> units
    on_order = 0
    sold = np.zeros(n_days)
    in_stock = np.zeros(n_days, dtype=bool)
    if sku.launch <= 0:  # selling from the start: open with eight weeks of stock
        layers.append([-30, _packs(planned * 56 * k)])
    first_review = max(sku.launch - cat.lead_days, 0)
    review = first_review + int(rng.integers(REVIEW_DAYS))

    for d in range(n_days):
        if d in incoming:
            units = incoming.pop(d)
            layers.append([d, units])
            on_order -= units
        if d == review and d < sku.discontinue:
            review += REVIEW_DAYS
            window = slice(max(d - 56, 0), d)
            stocked_days = in_stock[window]
            if (d - max(sku.launch, 0)) < 28 or stocked_days.sum() < 14:
                rate = planned  # too little in-stock history: the buyer's plan
            else:
                deseasoned = sold[window][stocked_days] / season[window][stocked_days]
                rate = float(deseasoned.mean())
            ahead = season[d + cat.lead_days : d + cat.lead_days + REVIEW_DAYS].mean()
            on_hand = sum(units for _, units in layers)
            target = rate * ahead * (cat.lead_days + REVIEW_DAYS) * k
            if target > on_hand + on_order:
                qty = _packs(target - on_hand - on_order)
                lead = max(
                    1, round(cat.lead_days * math.exp(cat.lead_sigma * rng.standard_normal()))
                )
                run.orders.append((d, d + cat.lead_days, d + lead, qty))
                on_order += qty
                if d + lead < n_days:
                    incoming[d + lead] = incoming.get(d + lead, 0) + qty

        in_stock[d] = bool(layers)
        wanted = int(demand[d])
        left = wanted
        while left and layers:
            take = min(left, layers[0][1])
            layers[0][1] -= take
            left -= take
            if layers[0][1] == 0:
                layers.pop(0)
        qty = wanted - left
        sold[d] = qty
        if qty:
            run.sales.append((d, qty, int(rng.binomial(qty, cat.return_rate))))
        if (d + 1) % 7 == 0:  # history starts on a Monday, so this is a Sunday
            on_hand = sum(units for _, units in layers)
            run.snapshots.append((d, on_hand, on_order, layers[0][0] if layers else None))
    return run


def _skus(spec: DemoSpec, rng: np.random.Generator, n_days: int) -> list[_Sku]:
    names = list(CATEGORIES)
    buyers, shares = zip(*((b, share) for b, (_, share) in BUYERS.items()), strict=True)
    skus = []
    for i in range(spec.n_skus):
        launch = 0
        if rng.random() < 0.25:  # launched during the last two years
            launch = int(rng.integers(n_days - 730, n_days - 84))
        discontinue = n_days
        if rng.random() < 0.1:  # discontinued in the last year, leaving dead stock
            discontinue = int(rng.integers(n_days - 365, n_days - 56))
        stocked = tuple(
            loc for loc, (*_, share) in LOCATIONS.items() if loc == "WH1" or rng.random() < share
        )
        skus.append(
            _Sku(
                sku_id=f"SKU-{i + 1:03d}",
                category=names[i % len(names)],
                base=float(rng.lognormal(0.0, 1.0)),
                growth=float(rng.uniform(-0.3, 0.4)),
                launch=launch,
                discontinue=max(discontinue, launch + 84),
                price=float(round(rng.lognormal(math.log(20), 0.6), 2)),
                buyer=str(rng.choice(buyers, p=shares)),
                locations=stocked,
            )
        )
    return skus


def _daily_shocks(rng: np.random.Generator, weeks: int) -> np.ndarray:
    """Weekly AR(1) log-shocks with standard deviation SHOCK_SD, repeated per day."""
    z = np.empty(weeks)
    z[0] = rng.standard_normal()
    scale = math.sqrt(1 - SHOCK_PERSISTENCE**2)
    for t in range(1, weeks):
        z[t] = SHOCK_PERSISTENCE * z[t - 1] + scale * rng.standard_normal()
    return np.repeat(SHOCK_SD * z, 7)


def _price_path(rng: np.random.Generator, sku: _Sku, n_days: int) -> np.ndarray:
    """List price per day: a change of up to +-8% roughly every half year."""
    price = np.full(n_days, sku.price)
    for d in range(182, n_days, 182):
        change = min(int(d + rng.integers(-30, 30)), n_days - 1)
        price[change:] = round(price[change] * (1 + rng.uniform(-0.08, 0.08)), 2)
    return price


def _packs(units: float) -> int:
    return max(PACK, math.ceil(units / PACK) * PACK)


def _number_po_lines(lines: list[dict[str, Any]]) -> pl.DataFrame:
    """One PO per location, supplier and order date; lines numbered within it."""
    if not lines:
        return pl.DataFrame(schema=PURCHASE_ORDER.columns).drop("tenant_id", "ingested_at")
    frame = pl.DataFrame(lines, infer_schema_length=None)
    po = pl.format(
        "PO-{}-{}-{}",
        pl.col("location_id"),
        pl.col("order_date").dt.strftime("%Y%m%d"),
        pl.col("supplier_id").str.replace("SUP-", ""),
    )
    return (
        frame.sort("order_date", "location_id", "supplier_id", "sku_id")
        .with_columns(po_id=po)
        .with_columns(line_no=pl.int_range(1, pl.len() + 1).over("po_id").cast(pl.Int32))
    )


def _write(out_root: Path, schema: TableSchema, tenant: str, frame: pl.DataFrame) -> int:
    frame = schema.conform(frame)
    schema.validate(frame)
    path = out_root / schema.name / tenant / "demo.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame.height


def demo_lots(canonical_root: Path, spec: DemoSpec | None = None) -> dict[str, dict[str, Any]]:
    """Three lot requests on the demo tenant, as of its last day, in the request format.

    - ``po_disciplined``: a new PO for the best sellers at WH1, eight weeks of their
      recent demand, arriving in six weeks; tenor 90 days.
    - ``po_overbought``: the same lines at 30 weeks of demand.
    - ``stock_slow_movers``: stock financing of the on-hand stock with the most
      cash tied up among series covering over 20 weeks; tenor 60 days.

    A lot with no qualifying lines (possible for a small tenant) is left out.
    """
    spec = spec or DemoSpec()
    as_of = spec.end
    store = AsOfStore(root=canonical_root, tenant_id=spec.tenant_id, as_of=as_of)
    sku = store.reference("sku").collect()
    established = sku.filter(
        (pl.col("lifecycle_status") == "active")
        & (pl.col("launch_date") <= as_of - timedelta(weeks=26))
    )["sku_id"]
    at_wh1 = pl.col("location_id") == "WH1"
    weekly = (
        store.facts("sales")
        .filter(at_wh1 & (pl.col("date") > as_of - timedelta(weeks=8)))
        .group_by("sku_id", "location_id")
        .agg(weekly=pl.col("qty").sum() / 8, revenue=pl.col("net_revenue").sum())
        .collect()
    )
    cost = (
        store.facts("price")
        .filter(at_wh1)
        .sort("valid_from")
        .group_by("sku_id", "location_id")
        .agg(unit_cost=pl.col("unit_cost").last())
        .collect()
    )
    stock = (
        store.facts("stock_snapshot")
        .filter(at_wh1 & (pl.col("snapshot_date") == as_of))
        .select("sku_id", "location_id", "on_hand_qty")
        .collect()
    )
    keys = ["sku_id", "location_id"]

    best = (
        weekly.filter(pl.col("sku_id").is_in(established.implode()) & (pl.col("weekly") > 0))
        .join(cost, on=keys)
        .sort("revenue", descending=True)
        .head(LOT_LINES)
    )

    def po_lines(weeks: int) -> list[dict[str, Any]]:
        return [
            {
                "sku_id": row["sku_id"],
                "location_id": row["location_id"],
                "qty": float(_packs(row["weekly"] * weeks)),
                "unit_cost": round(row["unit_cost"], 2),
            }
            for row in best.iter_rows(named=True)
        ]

    sold_before = store.facts("sales").select(keys).unique().collect()
    slow = (
        stock.filter(pl.col("on_hand_qty") > 0)
        .join(sold_before, on=keys)  # needs sales history, so a forecast exists
        .join(weekly.select(*keys, "weekly"), on=keys, how="left")
        .join(cost, on=keys)
        .with_columns(pl.col("weekly").fill_null(0.0))
        .filter(pl.col("on_hand_qty") > 20 * pl.col("weekly"))
        .sort(pl.col("on_hand_qty") * pl.col("unit_cost"), descending=True)
        .head(LOT_LINES // 2)
    )
    receipt = as_of + timedelta(days=PO_LEAD_DAYS)
    lots = {
        "po_disciplined": {
            "name": "PO, disciplined",
            "description": "New PO for the WH1 best sellers: 8 weeks of recent demand.",
            "as_of": as_of.isoformat(),
            "tenor_days": 90,
            "fee_rate": 0.03,
            "expected_receipt_date": receipt.isoformat(),
            "lines": po_lines(8),
        },
        "po_overbought": {
            "name": "PO, overbought",
            "description": "The same PO at 30 weeks of recent demand.",
            "as_of": as_of.isoformat(),
            "tenor_days": 90,
            "fee_rate": 0.03,
            "expected_receipt_date": receipt.isoformat(),
            "lines": po_lines(30),
        },
        "stock_slow_movers": {
            "name": "Stock, slow movers",
            "description": "Stock financing of on-hand WH1 stock covering over 20 weeks.",
            "as_of": as_of.isoformat(),
            "tenor_days": 60,
            "fee_rate": 0.03,
            "expected_receipt_date": None,
            "lines": [
                {
                    "sku_id": row["sku_id"],
                    "location_id": row["location_id"],
                    "qty": float(row["on_hand_qty"]),
                    "unit_cost": round(row["unit_cost"], 2),
                }
                for row in slow.iter_rows(named=True)
            ],
        },
    }
    return {name: lot for name, lot in lots.items() if lot["lines"]}
