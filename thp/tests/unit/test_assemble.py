from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from thp.core.asof import AsOfStore
from thp.schema import (
    PRICE,
    PURCHASE_ORDER,
    QUANTILE_COLS,
    SALES,
    SKU,
    STOCK_SNAPSHOT,
    TENANT,
    TableSchema,
)
from thp.simulate.assemble import LotRequest, assemble_lot
from thp.simulate.inputs import MissingForecastError

T = "t1"
AS_OF = date(2026, 1, 4)  # a Sunday: horizon week 1 starts Monday Jan 5
INGESTED = datetime(2026, 1, 1)


def _write(root: Path, schema: TableSchema, rows: list[dict[str, Any]]) -> None:
    path = root / schema.name / T
    path.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        tenant_id=pl.lit(T), ingested_at=pl.lit(INGESTED), channel=pl.lit("store")
    )
    schema.conform(frame).write_parquet(path / "p.parquet")


def _snap(sku: str, day: date, on_hand: float) -> dict[str, Any]:
    return {"sku_id": sku, "location_id": "L1", "snapshot_date": day, "on_hand_qty": on_hand}


def _po(n: int, ordered: date, expected: date, actual: date | None, qty: float) -> dict[str, Any]:
    return {
        "po_id": f"PO{n}",
        "line_no": 1,
        "sku_id": "A",
        "location_id": "L1",
        "order_date": ordered,
        "expected_receipt_date": expected,
        "actual_receipt_date": actual,
        "qty": qty,
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(tmp_path, TENANT, [{"week_start": 1}])
    _write(
        tmp_path,
        SKU,
        [{"sku_id": "A", "category_l1": "food"}, {"sku_id": "B", "category_l1": "toys"}],
    )
    _write(
        tmp_path,
        PRICE,
        [
            {"sku_id": "A", "location_id": "L1", "valid_from": date(2025, 6, 1), "list_price": 10},
            {"sku_id": "A", "location_id": "L1", "valid_from": date(2025, 12, 1), "list_price": 12},
            {"sku_id": "A", "location_id": "L1", "valid_from": date(2026, 1, 10), "list_price": 99},
            {"sku_id": "B", "location_id": "L1", "valid_from": date(2025, 12, 1), "list_price": 5},
            {"sku_id": "C", "location_id": "L1", "valid_from": date(2025, 12, 1), "list_price": 7},
        ],
    )
    _write(
        tmp_path,
        SALES,
        [
            {
                "sku_id": "A",
                "location_id": "L1",
                "date": date(2025, 11, 1),
                "qty": 100,
                "returned_qty": 5,
            },
            {
                "sku_id": "A",
                "location_id": "L1",
                "date": date(2024, 6, 1),
                "qty": 10,
                "returned_qty": 10,
            },  # over a year ago: not in the return rate
            {
                "sku_id": "B",
                "location_id": "L1",
                "date": date(2025, 11, 1),
                "qty": 50,
                "returned_qty": 0,
            },
        ],
    )
    _write(
        tmp_path,
        STOCK_SNAPSHOT,
        [
            _snap("A", date(2025, 12, 28), 30),
            _snap("A", AS_OF, 20),
            _snap("A", date(2026, 1, 11), 999),  # after as_of: invisible
            _snap("B", AS_OF, 0),
            _snap("C", AS_OF, 4),
        ],
    )
    _write(
        tmp_path,
        PURCHASE_ORDER,
        [
            _po(1, date(2025, 12, 1), date(2026, 1, 20), None, 10),  # open, ahead of the lot
            _po(2, date(2025, 12, 1), date(2025, 12, 20), date(2025, 12, 22), 50),  # on hand
            _po(3, date(2025, 12, 15), date(2026, 1, 10), date(2026, 1, 12), 7),  # open at as_of
            _po(4, date(2026, 1, 2), date(2026, 3, 30), None, 40),  # arrives after the lot
            _po(5, date(2026, 1, 6), date(2026, 1, 20), None, 1000),  # ordered after as_of
        ],
    )
    return tmp_path


def _forecast(weeks: int = 14, skus: tuple[str, ...] = ("A", "B", "C")) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model_version": "m",
                "as_of": AS_OF,
                "sku_id": sku,
                "location_id": "L1",
                "horizon_week": h,
                **dict.fromkeys(QUANTILE_COLS, 1.0),
            }
            for sku in skus
            for h in range(1, weeks + 1)
        ]
    ).with_columns(pl.col("horizon_week").cast(pl.Int16))


def _request(**overrides: Any) -> LotRequest:
    args: dict[str, Any] = {
        "as_of": AS_OF,
        "expected_receipt_date": date(2026, 2, 15),
        "lines": [
            {"sku_id": "A", "location_id": "L1", "qty": 60, "unit_cost": 6},
            {"sku_id": "B", "location_id": "L1", "qty": 30, "unit_cost": 3},
        ],
    }
    return LotRequest.model_validate({**args, **overrides})


def _assemble(root: Path, request: LotRequest, forecast: pl.DataFrame | None = None) -> Any:
    store = AsOfStore(root=root, tenant_id=T, as_of=AS_OF)
    return assemble_lot(store, request, _forecast() if forecast is None else forecast)


def test_po_lot_reads_everything_as_of(root: Path) -> None:
    lot = _assemble(root, _request())
    assert lot.series == ("A|L1", "B|L1")
    assert lot.horizon_start == date(2026, 1, 5)
    assert lot.disbursement_date == AS_OF
    np.testing.assert_allclose(lot.price, [12, 5])  # not the later 99
    assert lot.categories == ("food", "toys")
    np.testing.assert_allclose(lot.return_rate, [0.05, 0.0])  # the 2024 returns are too old
    # On hand 20, plus open POs expected by the lot's receipt: PO1 (10) and PO3 (7,
    # received only after as_of). Not PO2 (already on hand), PO4 (after the lot)
    # or PO5 (ordered after as_of).
    np.testing.assert_allclose(lot.stock_ahead, [37, 0])
    np.testing.assert_allclose(lot.qty, [60, 30])
    assert lot.tenors_days == (30, 60, 90)
    assert lot.knots.shape == (2, 13, 7)


def test_stock_lot_is_the_youngest_on_hand_stock(root: Path) -> None:
    request = _request(
        expected_receipt_date=None,
        lines=[{"sku_id": "A", "location_id": "L1", "qty": 15, "unit_cost": 6}],
    )
    np.testing.assert_allclose(_assemble(root, request).stock_ahead, [5])

    too_much = _request(
        expected_receipt_date=None,
        lines=[{"sku_id": "A", "location_id": "L1", "qty": 25, "unit_cost": 6}],
    )
    with pytest.raises(ValueError, match="exceeds stock on hand"):
        _assemble(root, too_much)


def test_sku_without_reference_row_gets_unknown_category(root: Path) -> None:
    request = _request(lines=[{"sku_id": "C", "location_id": "L1", "qty": 1, "unit_cost": 1}])
    lot = _assemble(root, request)
    assert lot.categories == ("unknown",)
    np.testing.assert_allclose(lot.return_rate, [0.0])


def test_line_without_price_is_refused(root: Path) -> None:
    request = _request(lines=[{"sku_id": "A", "location_id": "L2", "qty": 1, "unit_cost": 1}])
    with pytest.raises(ValueError, match=r"no list price .* A\|L2"):
        _assemble(root, request, _forecast().with_columns(location_id=pl.lit("L2")))


def test_stale_snapshot_is_refused(root: Path) -> None:
    _write(root, STOCK_SNAPSHOT, [_snap("A", date(2025, 12, 14), 20)])
    request = _request(lines=[{"sku_id": "A", "location_id": "L1", "qty": 1, "unit_cost": 1}])
    with pytest.raises(ValueError, match=r"older than 14 days for A\|L1"):
        _assemble(root, request)


def test_missing_forecast_weeks_refuse_the_lot(root: Path) -> None:
    with pytest.raises(MissingForecastError, match="week 11"):
        _assemble(root, _request(), _forecast(weeks=10))


def test_store_must_match_the_request(root: Path) -> None:
    store = AsOfStore(root=root, tenant_id=T, as_of=date(2026, 1, 1))
    with pytest.raises(ValueError, match="as of"):
        assemble_lot(store, _request(), _forecast())


def test_request_defaults_and_validation() -> None:
    request = _request(tenor_days=45)
    assert request.tenors == (30, 45, 60, 90)
    assert request.disbursed == AS_OF
    assert _request(disbursement_date=date(2026, 1, 6)).disbursed == date(2026, 1, 6)

    line = {"sku_id": "A", "location_id": "L1", "qty": 1, "unit_cost": 1}
    with pytest.raises(ValidationError, match="only one line"):
        _request(lines=[line, line])
    with pytest.raises(ValidationError):
        _request(lines=[])
    with pytest.raises(ValidationError):
        _request(lines=[{**line, "qty": 0}])
    with pytest.raises(ValidationError):
        _request(unknown_field=1)
