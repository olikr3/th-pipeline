from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thp.core.asof import AsOfStore
from thp.core.config import DemandSettings
from thp.demand.baseline import MODEL_VERSION, seasonal_naive
from thp.demand.external import external_forecast
from thp.demand.forecast import write_forecast
from thp.ingest.forecasts import read_external_forecasts, write_external_forecasts
from thp.schema import QUANTILE_COLS, QUANTILES, SALES, SKU, TENANT, TableSchema

TENANT_ID = "t1"
START = date(2024, 1, 1)  # a Monday; the tenant's weeks start on Monday
N_WEEKS = 96
AS_OF = START + timedelta(weeks=80, days=2)  # mid-week: week 80 is not complete
LAST_END = START + timedelta(weeks=80, days=-1)
SETTINGS = DemandSettings()
H = SETTINGS.horizon_weeks


def _seasonal_qty(k: int) -> float:
    """A yearly pattern plus a three-week wobble, so year-on-year errors are not zero."""
    return float(k % 52 + k % 3)


def _sales_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in range(N_WEEKS):
        monday = START + timedelta(weeks=k)
        qty = {"S": _seasonal_qty(k)}  # 80 weeks to as_of: seasonal
        if k >= 60:
            qty["N"] = 5.0  # 20 weeks: four-week mean
        if k >= 75:
            qty["C"] = [2.0, 4.0][k % 2]  # 5 weeks: cold start
        rows += [
            {
                "tenant_id": TENANT_ID,
                "sku_id": sku,
                "location_id": "L1",
                "date": monday,
                "channel": "store",
                "qty": q,
            }
            for sku, q in qty.items()
        ]
    return rows


def _write(root: Path, schema: TableSchema, rows: list[dict[str, Any]]) -> None:
    path = root / schema.name / TENANT_ID
    path.mkdir(parents=True, exist_ok=True)
    schema.conform(pl.DataFrame(rows, infer_schema_length=None)).write_parquet(path / "p.parquet")


def _write_all(root: Path, sales: list[dict[str, Any]]) -> None:
    _write(root, SALES, sales)
    _write(root, TENANT, [{"tenant_id": TENANT_ID, "week_start": 1}])
    _write(
        root,
        SKU,
        [
            {"tenant_id": TENANT_ID, "sku_id": "S", "category_l1": "CAT2"},
            {"tenant_id": TENANT_ID, "sku_id": "N", "category_l1": "CAT1"},
            {"tenant_id": TENANT_ID, "sku_id": "C", "category_l1": "CAT1"},
        ],
    )


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("canonical")
    _write_all(root, _sales_rows())
    return root


def _store(root: Path, as_of: date = AS_OF) -> AsOfStore:
    return AsOfStore(root=root, tenant_id=TENANT_ID, as_of=as_of)


@pytest.fixture(scope="module")
def baseline(root: Path) -> pl.DataFrame:
    return seasonal_naive(_store(root), SETTINGS)


def _row(frame: pl.DataFrame, sku: str, h: int) -> dict[str, Any]:
    rows = frame.filter(sku_id=sku, horizon_week=h)
    assert rows.height == 1
    return rows.row(0, named=True)


def _quantiles(row: dict[str, Any]) -> list[float]:
    return [row[col] for col in QUANTILE_COLS]


# --- baseline ---------------------------------------------------------------


def test_baseline_covers_every_series_and_horizon(baseline: pl.DataFrame) -> None:
    assert baseline.height == 3 * H
    assert set(baseline["model_version"]) == {MODEL_VERSION}
    assert set(baseline["as_of"]) == {AS_OF}
    weeks = baseline.filter(sku_id="S").sort("horizon_week")["week_end"].to_list()
    assert weeks == [LAST_END + timedelta(weeks=h) for h in range(1, H + 1)]


def test_seasonal_series_uses_last_year_and_its_errors(baseline: pl.DataFrame) -> None:
    # Errors over the 28 weeks that have a year behind them (52..79).
    errors = pl.Series([_seasonal_qty(s) - _seasonal_qty(s - 52) for s in range(52, 80)])
    offsets = [errors.quantile(q, interpolation="linear") or 0.0 for q in QUANTILES]
    for h in (1, 7, H):
        point = _seasonal_qty(79 + h - 52)
        row = _row(baseline, "S", h)
        assert _quantiles(row) == pytest.approx([max(point + o, 0.0) for o in offsets])
        assert row["is_cold_start"] is False


def test_short_series_uses_four_week_mean(baseline: pl.DataFrame) -> None:
    for h in (1, H):
        assert _quantiles(_row(baseline, "N", h)) == [5.0] * len(QUANTILES)


def test_cold_start_pools_errors_of_its_category(baseline: pl.DataFrame) -> None:
    row = _row(baseline, "C", 1)
    assert row["is_cold_start"] is True
    # Mostly N's zero errors in CAT1 x L1, so the median sits on the mean of 4, 2, 4, 2.
    assert row["q50"] == pytest.approx(3.0)
    assert row["q95"] > row["q50"] > row["q05"]


def test_quantiles_are_sorted_and_non_negative(baseline: pl.DataFrame) -> None:
    values = baseline.select(pl.concat_list(QUANTILE_COLS).alias("q"))["q"].to_list()
    assert all(v == sorted(v) and v[0] >= 0 for v in values)


def test_future_perturbation_leaves_baseline_unchanged(
    baseline: pl.DataFrame, tmp_path: Path
) -> None:
    rows = [{**r, "qty": r["qty"] + 1000} if r["date"] > AS_OF else r for r in _sales_rows()]
    _write_all(tmp_path, rows)
    assert_frame_equal(baseline, seasonal_naive(_store(tmp_path), SETTINGS))


def test_write_forecast(baseline: pl.DataFrame, tmp_path: Path) -> None:
    path = write_forecast(baseline, tmp_path)
    expected = tmp_path / "forecast" / MODEL_VERSION / TENANT_ID / f"as_of={AS_OF}"
    assert path == expected / "forecast.parquet"
    assert_frame_equal(pl.read_parquet(path), baseline)


# --- external forecasts -----------------------------------------------------


def _ingest(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    source: str = "planner",
    issued: date | None = AS_OF - timedelta(days=3),
    ingested_at: datetime | None = None,
) -> None:
    csv = root / "upload.csv"
    pl.DataFrame(rows, infer_schema_length=None).write_csv(csv)
    frame = read_external_forecasts(
        csv,
        tenant_id=TENANT_ID,
        source=source,
        week_start=1,
        issued_date=issued,
        ingested_at=ingested_at,
    )
    write_external_forecasts(frame, root)


def _week(h: int) -> date:
    """A Wednesday in horizon week h of AS_OF."""
    return LAST_END + timedelta(weeks=h, days=-4)


@pytest.fixture
def ext_root(tmp_path: Path) -> Path:
    _write_all(tmp_path, _sales_rows())
    return tmp_path


def _external(root: Path, baseline: pl.DataFrame, source: str = "planner") -> Any:
    return external_forecast(_store(root), source, baseline, SETTINGS)


def test_point_forecast_takes_baseline_spread(ext_root: Path, baseline: pl.DataFrame) -> None:
    _ingest(
        ext_root,
        [{"sku_id": "S", "location_id": "L1", "week": _week(h), "qty": 40.0} for h in (1, 7)]
        + [{"sku_id": "N", "location_id": "L1", "week": _week(1), "qty": 8.0}],
    )
    result = _external(ext_root, baseline)
    assert result.skipped.height == 0
    assert set(result.forecast["model_version"]) == {"external_planner"}

    for h in (1, 7):
        b = _row(baseline, "S", h)
        assert _quantiles(_row(result.forecast, "S", h)) == pytest.approx(
            [40.0 * b[col] / b["q50"] for col in QUANTILE_COLS]
        )
    # N's baseline has no spread: every quantile is the point.
    assert _quantiles(_row(result.forecast, "N", 1)) == [8.0] * len(QUANTILES)


def test_horizons_rebase_on_as_of(ext_root: Path, baseline: pl.DataFrame) -> None:
    # Issued four weeks before: its first weeks are history by as_of.
    _ingest(
        ext_root,
        [{"sku_id": "N", "location_id": "L1", "week": _week(h), "qty": 6.0} for h in range(-3, 16)],
        issued=AS_OF - timedelta(weeks=4),
    )
    forecast = _external(ext_root, baseline).forecast
    assert forecast["horizon_week"].to_list() == list(range(1, H + 1))
    assert forecast["week_end"].to_list() == [
        LAST_END + timedelta(weeks=h) for h in range(1, H + 1)
    ]


def test_given_quantiles_are_kept(ext_root: Path, baseline: pl.DataFrame) -> None:
    given = dict(zip(QUANTILE_COLS, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], strict=True))
    _ingest(
        ext_root,
        [
            # A SKU without sales: fine when the source gives every quantile.
            {"sku_id": "NEW", "location_id": "L1", "week": _week(1), **given},
            # Only some quantiles: the rest come from the baseline around q50.
            {"sku_id": "N", "location_id": "L1", "week": _week(1), "q50": 4.0, "q95": 9.0},
        ],
    )
    forecast = _external(ext_root, baseline).forecast
    new = _row(forecast, "NEW", 1)
    assert _quantiles(new) == list(given.values())
    assert new["is_cold_start"] is False
    assert _quantiles(_row(forecast, "N", 1)) == [4.0] * 6 + [9.0]


def test_point_forecast_without_history_is_skipped(ext_root: Path, baseline: pl.DataFrame) -> None:
    _ingest(ext_root, [{"sku_id": "NEW", "location_id": "L1", "week": _week(1), "qty": 3.0}])
    result = _external(ext_root, baseline)
    assert result.forecast.height == 0
    assert result.skipped.select("sku_id", "horizon_week").rows() == [("NEW", 1)]


def test_latest_visible_issue_wins(ext_root: Path, baseline: pl.DataFrame) -> None:
    def issue(qty: float, issued: date) -> None:
        _ingest(
            ext_root,
            [{"sku_id": "N", "location_id": "L1", "week": _week(2), "qty": qty}],
            issued=issued,
        )

    issue(6.0, AS_OF - timedelta(days=60))  # too old
    issue(7.0, AS_OF - timedelta(days=10))
    issue(8.0, AS_OF - timedelta(days=3))  # latest visible
    issue(9.0, AS_OF + timedelta(days=1))  # not issued yet at as_of
    assert _row(_external(ext_root, baseline).forecast, "N", 2)["q50"] == 8.0

    older = _store(ext_root, AS_OF - timedelta(days=5))
    early = external_forecast(older, "planner", seasonal_naive(older, SETTINGS), SETTINGS)
    assert early.forecast["q50"].to_list() == [7.0]


def test_reingested_rows_replace_earlier_ones(ext_root: Path, baseline: pl.DataFrame) -> None:
    for qty, hour in ((6.0, 1), (7.0, 2)):
        _ingest(
            ext_root,
            [{"sku_id": "N", "location_id": "L1", "week": _week(1), "qty": qty}],
            ingested_at=datetime(2025, 7, 1, hour),
        )
    assert _external(ext_root, baseline).forecast["q50"].to_list() == [7.0]


def test_sources_are_separate(ext_root: Path, baseline: pl.DataFrame) -> None:
    _ingest(ext_root, [{"sku_id": "N", "location_id": "L1", "week": _week(1), "qty": 6.0}])
    assert _external(ext_root, baseline, source="erp").forecast.height == 0


# --- ingest validation ------------------------------------------------------


def _read(tmp_path: Path, rows: list[dict[str, Any]], **kwargs: Any) -> pl.DataFrame:
    csv = tmp_path / "upload.csv"
    pl.DataFrame(rows, infer_schema_length=None).write_csv(csv)
    args: dict[str, Any] = {
        "tenant_id": TENANT_ID,
        "source": "planner",
        "week_start": 1,
        "issued_date": AS_OF,
    }
    return read_external_forecasts(csv, **{**args, **kwargs})


def test_ingest_maps_weeks_and_issue_dates(tmp_path: Path) -> None:
    frame = _read(
        tmp_path,
        [
            {"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "qty": 1},
            {"sku_id": "A", "location_id": "L1", "week": "2025-07-07", "qty": 2},
        ],
    )
    assert frame["week_end"].to_list() == [date(2025, 7, 6), date(2025, 7, 13)]
    assert set(frame["issued_date"]) == {AS_OF}
    assert frame["q50"].null_count() == 2

    with_column = _read(
        tmp_path,
        [
            {
                "sku_id": "A",
                "location_id": "L1",
                "week": "2025-07-02",
                "q50": 1,
                "issued_date": "2025-06-30",
            }
        ],
        issued_date=None,
    )
    assert with_column["issued_date"].to_list() == [date(2025, 6, 30)]


@pytest.mark.parametrize(
    ("rows", "kwargs", "message"),
    [
        ([{"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "q90": 1}], {}, "qty or q50"),
        (
            [
                {"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "qty": 1, "q50": None},
                {
                    "sku_id": "A",
                    "location_id": "L1",
                    "week": "2025-07-09",
                    "qty": None,
                    "q50": None,
                },
            ],
            {},
            "neither qty nor q50",
        ),
        ([{"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "qty": -1}], {}, "negative"),
        (
            [{"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "q10": 5, "q50": 3}],
            {},
            "decrease",
        ),
        (
            [
                {"sku_id": "A", "location_id": "L1", "week": "2025-07-01", "qty": 1},
                {"sku_id": "A", "location_id": "L1", "week": "2025-07-03", "qty": 2},
            ],
            {},
            "repeat",
        ),
        (
            [{"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "qty": 1}],
            {"issued_date": None},
            "issued_date",
        ),
        (
            [{"sku_id": "A", "location_id": "L1", "week": "2025-07-02", "qty": 1}],
            {"source": "My Planner"},
            "source",
        ),
        ([{"sku_id": "A", "week": "2025-07-02", "qty": 1}], {}, "location_id"),
    ],
)
def test_ingest_rejects_bad_files(
    tmp_path: Path, rows: list[dict[str, Any]], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, rows, **kwargs)
