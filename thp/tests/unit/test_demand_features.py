from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from thp.core.asof import AsOfStore
from thp.core.weeks import last_complete_week_end
from thp.features.demand import (
    build_demand_features,
    syntetos_boylan,
    write_demand_features,
)
from thp.schema import PRICE, SALES, SKU, TENANT, TableSchema

TENANT_ID = "t1"
START = date(2025, 1, 6)  # a Monday; the tenant's weeks start on Monday
N_WEEKS = 22
AS_OF = START + timedelta(weeks=20, days=2)  # mid-week: week 20 is not complete
LAST_END = START + timedelta(weeks=20, days=-1)


def _sales_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(sku: str, loc: str, day: date, qty: float, discount: float | None = None) -> None:
        rows.append(
            {
                "tenant_id": TENANT_ID,
                "sku_id": sku,
                "location_id": loc,
                "date": day,
                "channel": "store",
                "qty": qty,
                "net_revenue": qty * 9 if discount is not None else qty * 10,
                "discount": discount,
            }
        )

    for k in range(N_WEEKS):
        monday = START + timedelta(weeks=k)
        for d in range(7):
            add("A", "S1", monday + timedelta(days=d), 1.0)  # smooth, 7 a week
            add("A", "S2", monday + timedelta(days=d), 2.0, discount=2.0)  # 10% discount
        if k % 4 == 0:
            add("B", "S1", monday, (k + 1) * 3.0)  # every fourth week only
        if k >= 3:
            add("C", "S1", monday + timedelta(days=2), float(k))  # linear trend
    return rows


def _price_rows() -> list[dict[str, Any]]:
    changes = [
        (START, 10.0),
        (START + timedelta(weeks=10), 8.0),
        (START + timedelta(weeks=21), 5.0),
    ]
    return [
        {
            "tenant_id": TENANT_ID,
            "sku_id": "A",
            "location_id": "S1",
            "valid_from": d,
            "list_price": p,
        }
        for d, p in changes
    ]


def _write(root: Path, schema: TableSchema, rows: list[dict[str, Any]]) -> None:
    path = root / schema.name / TENANT_ID
    path.mkdir(parents=True, exist_ok=True)
    schema.conform(pl.DataFrame(rows, infer_schema_length=None)).write_parquet(path / "p.parquet")


def _write_all(root: Path, sales: list[dict[str, Any]]) -> None:
    _write(root, SALES, sales)
    _write(root, PRICE, _price_rows())
    _write(root, TENANT, [{"tenant_id": TENANT_ID, "week_start": 1}])
    _write(
        root,
        SKU,
        [
            {"tenant_id": TENANT_ID, "sku_id": "A", "category_l1": "CAT1"},
            {"tenant_id": TENANT_ID, "sku_id": "B", "category_l1": "CAT1"},
            {
                "tenant_id": TENANT_ID,
                "sku_id": "C",
                "category_l1": "CAT2",
                "launch_date": START + timedelta(weeks=3),
            },
        ],
    )


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("canonical")
    _write_all(root, _sales_rows())
    return root


def _build(root: Path, as_of: date = AS_OF) -> pl.DataFrame:
    return build_demand_features(AsOfStore(root=root, tenant_id=TENANT_ID, as_of=as_of))


@pytest.fixture(scope="module")
def panel(root: Path) -> pl.DataFrame:
    return _build(root)


def _last(panel: pl.DataFrame, sku: str, loc: str = "S1") -> dict[str, Any]:
    rows = panel.filter(sku_id=sku, location_id=loc, week_end=LAST_END)
    assert rows.height == 1
    return rows.row(0, named=True)


@pytest.mark.parametrize(
    ("as_of", "week_start", "expected"),
    [
        (date(2025, 1, 12), 1, date(2025, 1, 12)),  # Sunday ends a Monday week
        (date(2025, 1, 13), 1, date(2025, 1, 12)),
        (date(2025, 1, 18), 1, date(2025, 1, 12)),
        (date(2025, 1, 17), 6, date(2025, 1, 17)),  # Friday ends a Saturday week (M5)
        (date(2025, 1, 16), 6, date(2025, 1, 10)),
    ],
)
def test_last_complete_week_end(as_of: date, week_start: int, expected: date) -> None:
    assert last_complete_week_end(as_of, week_start) == expected


def test_weeks_end_on_tenant_week_and_drop_the_partial_week(panel: pl.DataFrame) -> None:
    assert panel["week_end"].max() == LAST_END
    assert set(panel["week_end"].dt.weekday().unique().to_list()) == {7}
    assert panel["tenant_id"].unique().to_list() == [TENANT_ID]


def test_series_start_at_first_sale_and_gaps_are_zero(panel: pl.DataFrame) -> None:
    weeks = panel.group_by("sku_id", "location_id").agg(pl.len(), pl.col("week_end").min())
    by_series = {
        (r["sku_id"], r["location_id"]): (r["len"], r["week_end"]) for r in weeks.rows(named=True)
    }
    first_end = START + timedelta(days=6)
    assert by_series == {
        ("A", "S1"): (20, first_end),
        ("A", "S2"): (20, first_end),
        ("B", "S1"): (20, first_end),
        ("C", "S1"): (17, first_end + timedelta(weeks=3)),
    }
    b = panel.filter(sku_id="B").sort("week_end")["qty_lag_1"].to_list()
    assert b[:5] == [3.0, 0.0, 0.0, 0.0, 15.0]


def test_lags_count_back_from_the_current_week(panel: pl.DataFrame) -> None:
    c = _last(panel, "C")
    assert (c["qty_lag_1"], c["qty_lag_2"], c["qty_lag_13"]) == (19.0, 18.0, 7.0)
    assert c["qty_lag_26"] is None


def test_rolling_stats(panel: pl.DataFrame) -> None:
    a, b = _last(panel, "A"), _last(panel, "B")
    assert (a["qty_mean_13"], a["qty_std_13"], a["zero_share_13"]) == (7.0, 0.0, 0.0)
    # B over weeks 16..19: 51, 0, 0, 0
    assert b["qty_mean_4"] == pytest.approx(12.75)
    assert (b["qty_median_4"], b["qty_max_4"], b["zero_share_4"]) == (0.0, 51.0, 0.75)
    assert b["qty_max_52"] == 51.0


def test_intermittency(panel: pl.DataFrame) -> None:
    a, b = _last(panel, "A"), _last(panel, "B")
    assert (a["adi"], a["cv2"], a["demand_class"]) == (1.0, 0.0, "smooth")
    # B sells 3, 15, 27, 39, 51 in 5 of 20 weeks: mean 27, population variance 288.
    assert b["adi"] == pytest.approx(4.0)
    assert b["cv2"] == pytest.approx(288 / 27**2)
    assert b["demand_class"] == "intermittent"


def test_syntetos_boylan_quadrants() -> None:
    frame = pl.DataFrame(
        {"adi": [1.0, 1.0, 2.0, 2.0, None], "cv2": [0.1, 0.9, 0.1, 0.9, None]},
        schema={"adi": pl.Float64, "cv2": pl.Float64},
    )
    classes = frame.select(syntetos_boylan(pl.col("adi"), pl.col("cv2")))
    assert classes.to_series().to_list() == ["smooth", "erratic", "intermittent", "lumpy", None]


def test_trend_and_lifecycle(panel: pl.DataFrame) -> None:
    c = _last(panel, "C")
    assert c["trend_slope_13"] == pytest.approx(1.0)
    assert c["mean_ratio_13_52"] == pytest.approx(13 / 11)  # weeks 7..19 vs 3..19
    assert (c["history_weeks"], c["is_new"], c["weeks_since_launch"]) == (17, False, 16)
    assert _last(panel, "A")["weeks_since_launch"] is None

    young = panel.filter(sku_id="C").sort("week_end").row(11, named=True)
    assert (young["history_weeks"], young["is_new"]) == (12, True)


def test_price_features(panel: pl.DataFrame) -> None:
    a = _last(panel, "A")
    # Price 10 in weeks 7..9 and 8 in weeks 10..19; the change in week 21 is not visible.
    assert a["price"] == 8.0
    assert a["price_ratio_13"] == pytest.approx(8 / (110 / 13))
    assert a["days_since_price_change"] == 9 * 7 + 6
    assert _last(panel, "B")["price"] is None

    assert _last(panel, "A", "S2")["discount_depth"] == pytest.approx(0.1)
    assert a["discount_depth"] is None


def test_hierarchy_means(panel: pl.DataFrame) -> None:
    b = _last(panel, "B")
    # CAT1 at S1 is A (7 a week) and B (51, 0, 0, 0) over the last four weeks.
    assert b["cat_qty_mean_4"] == pytest.approx((29 + 3.5 * 3) / 4)
    # All four series per week: 22, 9.5, 9.75, 10.
    assert b["tenant_qty_mean_4"] == pytest.approx(51.25 / 4)
    assert _last(panel, "C")["category_l1"] == "CAT2"


def test_future_perturbation_leaves_features_unchanged(panel: pl.DataFrame, tmp_path: Path) -> None:
    rows = [
        {**row, "qty": row["qty"] + 1000} if row["date"] > AS_OF else row for row in _sales_rows()
    ]
    _write_all(tmp_path, rows)
    assert_frame_equal(panel, _build(tmp_path))


@settings(max_examples=25, deadline=None)
@given(weeks=st.integers(min_value=0, max_value=20), days=st.integers(min_value=0, max_value=6))
def test_rows_do_not_change_when_later_data_arrives(
    root: Path, panel: pl.DataFrame, weeks: int, days: int
) -> None:
    as_of = START + timedelta(weeks=weeks, days=days)
    earlier = _build(root, as_of)
    expected = panel.filter(pl.col("week_end") <= last_complete_week_end(as_of, 1))
    assert_frame_equal(earlier, expected)


def test_write_panel_and_snapshot(root: Path, panel: pl.DataFrame, tmp_path: Path) -> None:
    store = AsOfStore(root=root, tenant_id=TENANT_ID, as_of=AS_OF)
    path = write_demand_features(store, tmp_path)
    assert (
        path == tmp_path / "demand" / "demand_v1" / TENANT_ID / f"as_of={AS_OF}" / "panel.parquet"
    )
    assert_frame_equal(pl.read_parquet(path), panel)

    snapshot = pl.read_parquet(write_demand_features(store, tmp_path, latest_only=True))
    assert snapshot["week_end"].unique().to_list() == [LAST_END]
    assert snapshot.height == 4
