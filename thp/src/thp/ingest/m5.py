"""M5 (Walmart) sandbox connector: the Kaggle CSVs -> canonical Parquet.

M5 has no costs, stock or purchase orders. Unit cost is derived from the selling
price with an assumed gross margin per category; stock and POs are synthesised
later by the pseudo-loan generator.
"""

import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from thp.core.config import M5Settings
from thp.schema import LOCATION, PRICE, SALES, SKU, TENANT, TableSchema

SALES_FILE = "sales_train_evaluation.csv"
PRICES_FILE = "sell_prices.csv"
CALENDAR_FILE = "calendar.csv"

# M5 weeks (wm_yr_wk) run Saturday to Friday.
WEEK_START = 6
CURRENCY = "USD"


def build_m5(
    raw_dir: Path,
    out_root: Path,
    settings: M5Settings,
    *,
    stores: Sequence[str] | None = None,
    ingested_at: datetime | None = None,
) -> dict[str, int]:
    """Write the canonical tables for the M5 tenant and return rows per table.

    Replaces whatever a previous run wrote for this tenant. ``stores`` limits the
    build to a subset of store ids, for fast development runs.
    """
    tenant = settings.tenant_id
    ingested_at = ingested_at or datetime.now(UTC).replace(tzinfo=None)

    calendar = pl.read_csv(
        raw_dir / CALENDAR_FILE,
        columns=["date", "wm_yr_wk", "d"],
        schema_overrides={"date": pl.Date},
    )
    wide = pl.scan_csv(raw_dir / SALES_FILE)
    if stores is not None:
        wide = wide.filter(pl.col("store_id").is_in(stores))
    sales_wide = wide.drop("id", "state_id").collect()
    found = sales_wide["store_id"].unique().sort().to_list()
    if stores is not None and set(stores) != set(found):
        raise ValueError(f"stores not in {SALES_FILE}: {sorted(set(stores) - set(found))}")
    prices = pl.read_csv(raw_dir / PRICES_FILE).filter(pl.col("store_id").is_in(found))

    for schema in (TENANT, LOCATION, SKU, SALES, PRICE):
        shutil.rmtree(out_root / schema.name / tenant, ignore_errors=True)

    counts = {SALES.name: 0}
    launches = []
    for store in found:
        long = (
            sales_wide.filter(pl.col("store_id") == store)
            .unpivot(
                index=["item_id", "dept_id", "cat_id", "store_id"],
                variable_name="d",
                value_name="qty",
            )
            .join(calendar, on="d")
        )
        # Zeros before the first sale mean "not yet on the shelf", not "no demand".
        launch = (
            long.filter(pl.col("qty") > 0)
            .group_by("item_id", "store_id")
            .agg(launch_date=pl.col("date").min())
        )
        launches.append(launch)
        qty = pl.col("qty").cast(pl.Float64)
        sales = (
            long.join(launch, on=["item_id", "store_id"])
            .filter(pl.col("date") >= pl.col("launch_date"))
            .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
            .select(
                tenant_id=pl.lit(tenant),
                sku_id="item_id",
                location_id="store_id",
                date="date",
                channel=pl.lit("store"),
                qty=qty,
                net_revenue=pl.when(qty == 0).then(0.0).otherwise(qty * pl.col("sell_price")),
                returned_qty=pl.lit(0.0),
                customer_type=pl.lit("b2c"),
                ingested_at=pl.lit(ingested_at),
            )
        )
        counts[SALES.name] += _write(out_root, SALES, tenant, store, sales)

    listed = pl.concat(launches)
    items = sales_wide.select("item_id", "dept_id", "cat_id").unique()

    week_start = calendar.group_by("wm_yr_wk").agg(valid_from=pl.col("date").min())
    margin = pl.col("cat_id").replace_strict(
        settings.gross_margin_by_category,
        default=settings.default_gross_margin,
        return_dtype=pl.Float64,
    )
    price = (
        prices.join(listed, on=["item_id", "store_id"], how="semi")
        .sort("store_id", "item_id", "wm_yr_wk")
        # Keep only the weeks where the price changes.
        .filter(
            (
                pl.col("sell_price") != pl.col("sell_price").shift().over("store_id", "item_id")
            ).fill_null(True)
        )
        .join(week_start, on="wm_yr_wk")
        .join(items, on="item_id")
        .select(
            tenant_id=pl.lit(tenant),
            sku_id="item_id",
            location_id="store_id",
            valid_from="valid_from",
            list_price="sell_price",
            unit_cost=pl.col("sell_price") * (1 - margin),
            currency=pl.lit(CURRENCY),
            ingested_at=pl.lit(ingested_at),
        )
    )
    counts[PRICE.name] = _write(out_root, PRICE, tenant, "prices", price)

    sku = (
        listed.group_by("item_id")
        .agg(launch_date=pl.col("launch_date").min())
        .join(items, on="item_id")
        .select(
            tenant_id=pl.lit(tenant),
            sku_id="item_id",
            name="item_id",
            category_l1="cat_id",
            category_l2="dept_id",
            lifecycle_status=pl.lit("active"),
            launch_date="launch_date",
        )
    )
    counts[SKU.name] = _write(out_root, SKU, tenant, "skus", sku)

    location = pl.DataFrame({"location_id": found}).select(
        tenant_id=pl.lit(tenant),
        location_id="location_id",
        name="location_id",
        type=pl.lit("store"),
        country=pl.lit("US"),
    )
    counts[LOCATION.name] = _write(out_root, LOCATION, tenant, "locations", location)

    tenant_row = pl.DataFrame(
        {
            "tenant_id": [tenant],
            "name": ["M5 sandbox"],
            "currency": [CURRENCY],
            "week_start": [WEEK_START],
            "created_at": [ingested_at],
        }
    )
    counts[TENANT.name] = _write(out_root, TENANT, tenant, "tenant", tenant_row)
    return counts


def _write(out_root: Path, schema: TableSchema, tenant: str, part: str, frame: pl.DataFrame) -> int:
    frame = schema.conform(frame)
    schema.validate(frame)
    path = out_root / schema.name / tenant / f"{part}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame.height
