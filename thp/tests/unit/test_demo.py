import json
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thp import cli
from thp.core.config import load_settings
from thp.ingest.demo import CATEGORIES, DemoSpec, build_demo, demo_lots
from thp.scoring import score_lot
from thp.simulate.assemble import LotRequest

# Small enough to be quick, long enough for the seasonal baseline (65+ weeks).
SPEC = DemoSpec(n_skus=12, weeks=80, seed=3)
INGESTED = datetime(2026, 10, 1)


def _table(root: Path, name: str) -> pl.DataFrame:
    return pl.read_parquet(root / name / SPEC.tenant_id / "*.parquet")


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("demo")
    build_demo(root, SPEC, ingested_at=INGESTED)
    return root


def test_writes_every_table(root: Path) -> None:
    for table in (
        "tenant",
        "location",
        "sku",
        "sales",
        "price",
        "stock_snapshot",
        "purchase_order",
    ):
        assert _table(root, table).height > 0, table
    sales = _table(root, "sales")
    first = SPEC.end - timedelta(weeks=SPEC.weeks)
    assert sales.filter((pl.col("date") > SPEC.end) | (pl.col("date") <= first)).is_empty()


def test_same_seed_same_data(root: Path, tmp_path: Path) -> None:
    build_demo(tmp_path, SPEC, ingested_at=INGESTED)
    for table in ("sales", "stock_snapshot", "purchase_order"):
        assert_frame_equal(_table(root, table), _table(tmp_path, table))


def test_sales_and_stock_are_consistent(root: Path) -> None:
    sales = _table(root, "sales")
    assert (sales["qty"] > 0).all()
    assert (sales["returned_qty"] <= sales["qty"]).all()

    snapshots = _table(root, "stock_snapshot")
    assert set(snapshots["snapshot_date"].dt.weekday()) == {7}
    assert (snapshots["on_hand_qty"] >= 0).all()
    assert (snapshots["on_order_qty"] >= 0).all()
    # Stock-outs happen, as in real data, but are not the norm.
    stocked_out = snapshots.select((pl.col("on_hand_qty") == 0).mean()).item()
    assert 0 < stocked_out < 0.4


def test_purchase_orders_are_time_honest(root: Path) -> None:
    po = _table(root, "purchase_order")
    received = po.filter(pl.col("status") == "received")
    open_ = po.filter(pl.col("status") == "open")
    assert received["actual_receipt_date"].null_count() == 0
    assert (received["actual_receipt_date"] <= SPEC.end).all()
    assert (received["actual_receipt_date"] > received["order_date"]).all()
    # A receipt after the last day is unknown, not recorded.
    assert open_.height > 0
    assert open_["actual_receipt_date"].null_count() == open_.height
    lead = {c.supplier: c.lead_days for c in CATEGORIES.values()}
    expected_lead = (po["expected_receipt_date"] - po["order_date"]).dt.total_days()
    assert expected_lead.to_list() == [lead[s] for s in po["supplier_id"]]
    assert po.select("po_id", "line_no").n_unique() == po.height


def test_history_must_end_on_a_sunday(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Sunday"):
        build_demo(tmp_path, DemoSpec(end=date(2026, 9, 28)))


def test_demo_lots_are_valid_requests(root: Path) -> None:
    lots = {name: LotRequest.model_validate(lot) for name, lot in demo_lots(root, SPEC).items()}
    assert {"po_disciplined", "po_overbought"} <= set(lots)
    disciplined, overbought = lots["po_disciplined"], lots["po_overbought"]
    assert [line.sku_id for line in disciplined.lines] == [line.sku_id for line in overbought.lines]
    assert all(o.qty > d.qty for d, o in zip(disciplined.lines, overbought.lines, strict=True))
    assert disciplined.expected_receipt_date == SPEC.end + timedelta(days=42)
    if "stock_slow_movers" in lots:
        assert lots["stock_slow_movers"].expected_receipt_date is None


def test_overbuying_lowers_the_advance(root: Path) -> None:
    # The end-to-end check the demo exists for: same SKUs, 30 instead of 8 weeks
    # of demand, so more of the lot is still unsold at day 90.
    settings = load_settings({"simulate": {"n_paths": 400}})
    lots = demo_lots(root, SPEC)
    scores = {
        name: score_lot(root, SPEC.tenant_id, LotRequest.model_validate(lots[name]), settings)
        for name in ("po_disciplined", "po_overbought")
    }
    disciplined, overbought = scores["po_disciplined"], scores["po_overbought"]
    assert overbought["advance_rate"] < disciplined["advance_rate"]
    for score in scores.values():
        assert 0 <= score["advance_amount"] <= 0.8 * score["lot_cost"] + 1e-6
        assert score["versions"]["forecast"] == "seasonal_naive_v2"
        d90 = score["cash_ratio"]["d90"]
        assert d90["q05"] <= d90["q50"] <= d90["q90"]


def test_scores_without_stock_or_purchase_orders(root: Path, tmp_path: Path) -> None:
    # Sources like M5 have neither table; a PO lot still scores, with nothing ahead of it.
    for table in ("tenant", "location", "sku", "sales", "price"):
        shutil.copytree(root / table, tmp_path / table)
    settings = load_settings({"simulate": {"n_paths": 200}})
    lots = demo_lots(root, SPEC)
    request = LotRequest.model_validate(lots["po_disciplined"])
    without = score_lot(tmp_path, SPEC.tenant_id, request, settings)
    with_stock = score_lot(root, SPEC.tenant_id, request, settings)
    assert without["advance_amount"] >= with_stock["advance_amount"]
    if "stock_slow_movers" in lots:
        stock_lot = LotRequest.model_validate(lots["stock_slow_movers"])
        with pytest.raises(ValueError, match="exceeds stock on hand"):
            score_lot(tmp_path, SPEC.tenant_id, stock_lot, settings)


def test_cli_make_demo_and_score_lot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    canonical, lots = tmp_path / "canonical", tmp_path / "lots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "thp",
            "make-demo",
            "--canonical",
            str(canonical),
            "--lots",
            str(lots),
            "--skus",
            "12",
            "--weeks",
            "80",
            "--seed",
            "3",
        ],
    )
    cli.main()
    lot_file = lots / "po_disciplined.json"
    request = json.loads(lot_file.read_text())
    request["options"] = {"n_paths": 200, "seed": 5}
    lot_file.write_text(json.dumps(request))
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        ["thp", "score-lot", str(lot_file), "--tenant", "demo", "--canonical", str(canonical)],
    )
    cli.main()
    scored = json.loads(capsys.readouterr().out)
    # A bare name is looked up in --lots.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "thp",
            "score-lot",
            "po_disciplined",
            "--tenant",
            "demo",
            "--canonical",
            str(canonical),
            "--lots",
            str(lots),
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == scored
    assert scored["name"] == "PO, disciplined"
    assert scored["versions"]["seed"] == 5
    assert set(scored["cash_ratio"]) == {"d30", "d60", "d90"}
    # The n_paths override is part of the configuration the hash describes.
    assert (
        scored["versions"]["config_hash"]
        == load_settings({"simulate": {"n_paths": 200}}).config_hash
    )
