from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from thp.core.asof import AsOfStore

TENANT = "t1"
START = date(2025, 1, 1)
N_DAYS = 120
AS_OF = date(2025, 3, 1)


def _sales() -> pl.DataFrame:
    days = [START + timedelta(days=i) for i in range(N_DAYS)]
    rows = [
        {
            "tenant_id": tenant,
            "sku_id": sku,
            "location_id": "WH1",
            "date": day,
            "qty": float((i * 7 + len(sku)) % 11),
            # Ingested at 02:00 the night after the sale.
            "ingested_at": datetime.combine(day + timedelta(days=1), datetime.min.time())
            + timedelta(hours=2),
        }
        for tenant in (TENANT, "t2")
        for sku in ("A", "BB")
        for i, day in enumerate(days)
    ]
    return pl.DataFrame(rows)


def _write(root: Path, table: str, frame: pl.DataFrame) -> None:
    path = root / table
    path.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path / "part-0.parquet")


def _read(root: Path, as_of: date = AS_OF, grace: timedelta | None = None) -> pl.DataFrame:
    store = AsOfStore(root=root, tenant_id=TENANT, as_of=as_of, ingest_grace=grace)
    return store.facts("sales").sort("sku_id", "date").collect()


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("store")
    _write(root, "sales", _sales())
    return root


def test_future_perturbation_leaves_reads_unchanged(root: Path, tmp_path: Path) -> None:
    perturbed = _sales().with_columns(
        pl.when(pl.col("date") > AS_OF)
        .then(pl.col("qty") + 1000)
        .otherwise(pl.col("qty"))
        .alias("qty")
    )
    _write(tmp_path, "sales", perturbed)

    assert_frame_equal(_read(root), _read(tmp_path))


def test_reads_include_as_of_day_and_nothing_later(root: Path) -> None:
    frame = _read(root)
    assert frame["date"].max() == AS_OF
    assert frame.height == 2 * ((AS_OF - START).days + 1)


def test_reads_only_own_tenant(root: Path) -> None:
    assert _read(root)["tenant_id"].unique().to_list() == [TENANT]


def test_ingest_grace_hides_late_rows(root: Path) -> None:
    # Rows land at 02:00 the next day, so zero grace hides the as_of day itself.
    assert _read(root, grace=timedelta(0))["date"].max() == AS_OF - timedelta(days=1)
    assert _read(root, grace=timedelta(days=1))["date"].max() == AS_OF


def test_unknown_table_is_rejected(root: Path) -> None:
    store = AsOfStore(root=root, tenant_id=TENANT, as_of=AS_OF)
    with pytest.raises(KeyError, match="unknown fact table"):
        store.facts("sku")
    with pytest.raises(KeyError, match="unknown reference table"):
        store.reference("sales")


def test_missing_optional_table_reads_as_empty(root: Path) -> None:
    store = AsOfStore(root=root, tenant_id=TENANT, as_of=AS_OF)
    for table in ("stock_snapshot", "purchase_order", "external_forecast"):
        frame = store.facts(table).collect()
        assert frame.height == 0
        assert frame.columns[0] == "tenant_id"


def test_missing_required_table_names_it(root: Path) -> None:
    store = AsOfStore(root=root, tenant_id=TENANT, as_of=AS_OF)
    with pytest.raises(FileNotFoundError, match="no 'price' table"):
        store.facts("price").collect()
    with pytest.raises(FileNotFoundError, match="no 'sku' table"):
        store.reference("sku")


@given(offset=st.integers(min_value=-30, max_value=N_DAYS + 30))
def test_no_row_is_ever_dated_after_as_of(root: Path, offset: int) -> None:
    as_of = START + timedelta(days=offset)
    frame = _read(root, as_of=as_of)
    assert frame.filter(pl.col("date") > as_of).is_empty()
    assert frame.height == 2 * min(max(offset + 1, 0), N_DAYS)
