from datetime import date

import polars as pl
import pytest

from thp.schema import PRICE, SKU


def test_conform_casts_orders_and_fills_optional_columns() -> None:
    frame = pl.DataFrame(
        {"extra": [1], "sku_id": ["A"], "launch_date": ["2025-01-01"], "tenant_id": ["t1"]}
    ).with_columns(pl.col("launch_date").str.to_date())

    out = SKU.conform(frame)

    assert out.columns == list(SKU.columns)
    assert out.schema == pl.Schema(SKU.columns)
    assert out.row(0, named=True)["launch_date"] == date(2025, 1, 1)
    assert out["brand"].null_count() == 1


def test_conform_requires_key_columns() -> None:
    with pytest.raises(ValueError, match="missing key columns"):
        SKU.conform(pl.DataFrame({"tenant_id": ["t1"]}))


def test_validate_rejects_null_and_duplicate_keys() -> None:
    row = {"tenant_id": "t1", "sku_id": "A", "location_id": "L", "valid_from": date(2025, 1, 1)}
    PRICE.validate(PRICE.conform(pl.DataFrame([row])))

    with pytest.raises(ValueError, match="repeat an existing key"):
        PRICE.validate(PRICE.conform(pl.DataFrame([row, row])))
    with pytest.raises(ValueError, match="null values in key columns"):
        PRICE.validate(PRICE.conform(pl.DataFrame([row, {**row, "sku_id": None}])))
