from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from thp.core.asof import AsOfStore
from thp.core.config import M5Settings
from thp.ingest.m5 import CALENDAR_FILE, PRICES_FILE, SALES_FILE, build_m5

START = date(2011, 1, 29)  # a Saturday, like the real calendar
N_DAYS = 21
WEEKS = (11101, 11102, 11103)
INGESTED = datetime(2026, 1, 1)

# (item, store) -> (first selling day index or None, {week: price})
SERIES: dict[tuple[str, str], tuple[int | None, dict[int, float]]] = {
    ("HOBBIES_1_001", "CA_1"): (0, {11101: 10.0, 11102: 10.0, 11103: 12.0}),
    ("FOODS_1_001", "CA_1"): (9, {11102: 2.0, 11103: 2.0}),
    ("HOUSEHOLD_1_001", "CA_1"): (None, {11101: 5.0, 11102: 5.0, 11103: 5.0}),
    ("HOBBIES_1_001", "TX_1"): (2, {11101: 11.0, 11102: 11.0, 11103: 11.0}),
    ("FOODS_1_001", "TX_1"): (0, {11101: 2.5, 11103: 2.5}),  # delisted in week 2
    ("HOUSEHOLD_1_001", "TX_1"): (0, {11101: 5.0, 11102: 5.0, 11103: 4.0}),
}


def _qty(first: int | None, prices: dict[int, float], day: int) -> int:
    if first is None or day < first or WEEKS[day // 7] not in prices:
        return 0
    return 1 + day % 3


@pytest.fixture(scope="module")
def raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw = tmp_path_factory.mktemp("m5_raw")
    pl.DataFrame(
        {
            "date": [START + timedelta(days=i) for i in range(N_DAYS)],
            "wm_yr_wk": [WEEKS[i // 7] for i in range(N_DAYS)],
            "weekday": ["x"] * N_DAYS,
            "d": [f"d_{i + 1}" for i in range(N_DAYS)],
        }
    ).write_csv(raw / CALENDAR_FILE)
    pl.DataFrame(
        [
            {
                "id": f"{item}_{store}_evaluation",
                "item_id": item,
                "dept_id": item.rsplit("_", 1)[0],
                "cat_id": item.split("_")[0],
                "store_id": store,
                "state_id": store.split("_")[0],
                **{f"d_{i + 1}": _qty(first, prices, i) for i in range(N_DAYS)},
            }
            for (item, store), (first, prices) in SERIES.items()
        ]
    ).write_csv(raw / SALES_FILE)
    pl.DataFrame(
        [
            {"store_id": store, "item_id": item, "wm_yr_wk": week, "sell_price": price}
            for (item, store), (_, prices) in SERIES.items()
            for week, price in prices.items()
        ]
    ).write_csv(raw / PRICES_FILE)
    return raw


@pytest.fixture(scope="module")
def built(raw: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, int]]:
    out = tmp_path_factory.mktemp("canonical")
    return out, build_m5(raw, out, M5Settings(), ingested_at=INGESTED)


def _store(root: Path, as_of: date = date(2011, 12, 31)) -> AsOfStore:
    return AsOfStore(root=root, tenant_id="m5", as_of=as_of)


def test_row_counts(built: tuple[Path, dict[str, int]]) -> None:
    assert built[1] == {"sales": 94, "price": 7, "sku": 3, "location": 2, "tenant": 1}


def test_sales_start_at_first_sale_and_skip_never_sold_series(
    built: tuple[Path, dict[str, int]],
) -> None:
    sales = _store(built[0]).facts("sales").collect()
    first = sales.group_by("sku_id", "location_id").agg(pl.col("date").min(), pl.len())
    by_series = {(r["sku_id"], r["location_id"]): (r["date"], r["len"]) for r in first.to_dicts()}

    assert by_series == {
        ("HOBBIES_1_001", "CA_1"): (START, 21),
        ("FOODS_1_001", "CA_1"): (START + timedelta(days=9), 12),
        ("HOBBIES_1_001", "TX_1"): (START + timedelta(days=2), 19),
        ("FOODS_1_001", "TX_1"): (START, 21),
        ("HOUSEHOLD_1_001", "TX_1"): (START, 21),
    }


def test_net_revenue_uses_the_price_of_that_week(built: tuple[Path, dict[str, int]]) -> None:
    sales = _store(built[0]).facts("sales").collect()
    hobbies = sales.filter(sku_id="HOBBIES_1_001", location_id="CA_1").sort("date")
    assert hobbies.row(14, named=True)["qty"] == 3.0
    assert hobbies.row(14, named=True)["net_revenue"] == 36.0  # week 3 price is 12.0

    # No price row in the delisted week: nothing sold, so revenue is 0, not null.
    assert sales["net_revenue"].null_count() == 0
    delisted = sales.filter(sku_id="FOODS_1_001", location_id="TX_1").sort("date")[7:14]
    assert delisted["net_revenue"].sum() == 0.0


def test_price_keeps_change_points_per_location_with_derived_cost(
    built: tuple[Path, dict[str, int]],
) -> None:
    price = _store(built[0]).facts("price").sort("location_id", "sku_id", "valid_from").collect()
    rows = price.select("sku_id", "location_id", "valid_from", "list_price", "unit_cost").rows()

    assert rows == [
        ("FOODS_1_001", "CA_1", START + timedelta(days=7), 2.0, 1.5),
        ("HOBBIES_1_001", "CA_1", START, 10.0, 6.0),
        ("HOBBIES_1_001", "CA_1", START + timedelta(days=14), 12.0, pytest.approx(7.2)),
        ("FOODS_1_001", "TX_1", START, 2.5, 1.875),
        ("HOBBIES_1_001", "TX_1", START, 11.0, pytest.approx(6.6)),
        ("HOUSEHOLD_1_001", "TX_1", START, 5.0, 3.25),
        ("HOUSEHOLD_1_001", "TX_1", START + timedelta(days=14), 4.0, pytest.approx(2.6)),
    ]


def test_reference_tables(built: tuple[Path, dict[str, int]]) -> None:
    store = _store(built[0])
    sku = store.reference("sku").sort("sku_id").collect()
    assert sku["sku_id"].to_list() == ["FOODS_1_001", "HOBBIES_1_001", "HOUSEHOLD_1_001"]
    assert sku["category_l1"].to_list() == ["FOODS", "HOBBIES", "HOUSEHOLD"]
    assert sku["category_l2"].to_list() == ["FOODS_1", "HOBBIES_1", "HOUSEHOLD_1"]
    assert sku["launch_date"].to_list() == [START, START, START]

    assert store.reference("location").sort("location_id").collect()["location_id"].to_list() == [
        "CA_1",
        "TX_1",
    ]
    assert store.reference("tenant").collect()["week_start"].to_list() == [6]


def test_as_of_cuts_sales_and_prices(built: tuple[Path, dict[str, int]]) -> None:
    store = _store(built[0], as_of=START + timedelta(days=6))
    assert store.facts("sales").collect()["date"].max() == START + timedelta(days=6)
    assert store.facts("price").collect()["valid_from"].unique().to_list() == [START]


def test_store_subset_replaces_previous_build(raw: Path, tmp_path: Path) -> None:
    build_m5(raw, tmp_path, M5Settings(), ingested_at=INGESTED)
    counts = build_m5(raw, tmp_path, M5Settings(), stores=["CA_1"], ingested_at=INGESTED)

    assert counts == {"sales": 33, "price": 3, "sku": 2, "location": 1, "tenant": 1}
    store = _store(tmp_path)
    assert store.facts("sales").collect()["location_id"].unique().to_list() == ["CA_1"]
    foods = store.reference("sku").filter(sku_id="FOODS_1_001").collect()
    assert foods["launch_date"].to_list() == [START + timedelta(days=9)]


def test_unknown_store_is_rejected(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NOPE"):
        build_m5(raw, tmp_path, M5Settings(), stores=["CA_1", "NOPE"])
