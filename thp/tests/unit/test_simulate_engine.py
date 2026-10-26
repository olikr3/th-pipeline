from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from thp.core.config import MarkdownSettings, SimulateSettings
from thp.schema import QUANTILE_COLS
from thp.simulate.engine import Lot, receipt_week_probs, simulate_lot
from thp.simulate.inputs import MissingForecastError, forecast_knots, series_key

HORIZON_START = date(2026, 1, 5)  # a Monday: horizon week 1 is Jan 5..11
AS_OF = HORIZON_START - timedelta(days=1)  # the Sunday that completes the last week
WEEKS = 13
# Markdowns off unless a test turns them on, so mechanics can be checked exactly.
NO_MARKDOWN = MarkdownSettings(cover_weeks=1e9, age_weeks=1000)
SETTINGS = SimulateSettings(n_paths=200, markdown=NO_MARKDOWN)


def _knots(per_week: float, n: int = 1) -> np.ndarray:
    """Forecasts with every quantile equal: demand is exactly this, whatever the draw."""
    return np.full((n, WEEKS, 7), per_week, dtype=np.float32)


def _lot(**overrides: Any) -> Lot:
    n = len(overrides.get("series", ("A|L1",)))
    args: dict[str, Any] = {
        "as_of": AS_OF,
        "horizon_start": HORIZON_START,
        "disbursement_date": AS_OF,
        "series": ("A|L1",),
        "categories": ("food",),
        "qty": np.full(n, 100, np.float32),
        "unit_cost": np.full(n, 6, np.float32),
        "price": np.full(n, 10, np.float32),
        "stock_ahead": np.zeros(n, np.float32),
        "return_rate": np.zeros(n, np.float32),
        "knots": _knots(10.0, n),
        "tenors_days": (28, 56, 91),
    }
    return Lot(**{**args, **overrides})


def _ratio(lot: Lot, sim_settings: SimulateSettings = SETTINGS, seed: int = 1) -> np.ndarray:
    return simulate_lot(lot, seed, sim_settings).cash_ratio


# --- mechanics with deterministic demand ---------------------------------------


def test_on_hand_lot_sells_at_forecast_rate() -> None:
    # 10 units a week at price 10 against cost 600: 100/600 of cost per week.
    ratio = _ratio(_lot())
    np.testing.assert_allclose(ratio, [[4 / 6, 8 / 6, 10 / 6]] * SETTINGS.n_paths, rtol=1e-6)


def test_zero_demand_gives_zero_cash_and_everything_unsold() -> None:
    sim = simulate_lot(_lot(knots=_knots(0.0)), 1, SETTINGS)
    assert (sim.cash_ratio == 0).all()
    assert (sim.unsold_units == 100).all()


def test_stock_ahead_sells_first() -> None:
    # 25 units ahead take weeks 0, 1 and half of week 2: the lot sells 5 units in
    # week 2, then 10 a week.
    sim = simulate_lot(
        _lot(stock_ahead=np.array([25], dtype=np.float32), tenors_days=(28, 56)), 1, SETTINGS
    )
    np.testing.assert_allclose(sim.cash_ratio[0], [15 / 60, 55 / 60], rtol=1e-6)
    np.testing.assert_allclose(sim.unsold_units[0], [100 - 55], rtol=1e-6)


def test_returns_reduce_cash() -> None:
    ratio = _ratio(_lot(return_rate=np.array([0.1], dtype=np.float32)))
    np.testing.assert_allclose(ratio[0], np.array([4, 8, 10]) / 6 * 0.9, rtol=1e-6)


def test_po_lot_waits_for_receipt() -> None:
    # Ordered on day 0, expected on day 22 (in week 3): sells from week 4, i.e. day 28.
    lot = _lot(expected_receipt_date=HORIZON_START + timedelta(days=22))
    sim = simulate_lot(lot, 1, SETTINGS.model_copy(update={"receipt_sigma": 0.0}))
    assert (sim.receipt_week == 4).all()
    np.testing.assert_allclose(sim.cash_ratio[0], [0, 4 / 6, 9 / 6], rtol=1e-6)


def test_tenor_counts_from_disbursement() -> None:
    # Disbursed at the end of day 3 of a lot on hand since day 0: the 28 days run
    # over days 4..31, still four weeks of sales. By day 88 the lot has sold out
    # (day 70), and the 4/7 of a week sold before disbursement does not count.
    late = HORIZON_START + timedelta(days=3)
    ratio = _ratio(_lot(as_of=late, disbursement_date=late, tenors_days=(28, 59, 84)))
    np.testing.assert_allclose(ratio[0], [4 / 6, (59 / 7) / 6, (1000 - 400 / 7) / 600], rtol=1e-5)


def test_tenor_ending_mid_week_takes_that_fraction() -> None:
    ratio = _ratio(_lot(tenors_days=(10, 31, 45)))
    np.testing.assert_allclose(ratio[0], np.array([10, 31, 45]) / 7 / 6, rtol=1e-6)


# --- markdowns -------------------------------------------------------------------


def test_slow_stock_is_marked_down_in_steps() -> None:
    # 1,000 units at 10 a week is 100 weeks of cover: cut 20% in week 0, 40% in
    # week 4, 50% (the floor) in week 8. Demand rises by (1 - cut) ** -1.5.
    markdown = MarkdownSettings()
    lot = _lot(qty=np.array([1000], dtype=np.float32), tenors_days=(28, 56, 91))
    ratio = _ratio(lot, SimulateSettings(n_paths=10, markdown=markdown))
    cash_per_week = [10 * (1 - c) ** -1.5 * 10 * (1 - c) for c in [0.2] * 4 + [0.4] * 4 + [0.5] * 5]
    expected = np.cumsum(cash_per_week)[[3, 7, 12]] / 6000
    np.testing.assert_allclose(ratio[0], expected, rtol=1e-5)


def test_old_lot_stock_is_marked_down() -> None:
    markdown = MarkdownSettings(age_weeks=2)  # cover stays low; age triggers in week 3
    lot = _lot(qty=np.array([200], dtype=np.float32), tenors_days=(28,))
    ratio = _ratio(lot, SimulateSettings(n_paths=10, markdown=markdown))
    expected = (3 * 100 + 10 * 0.8**-1.5 * 10 * 0.8) / 1200
    np.testing.assert_allclose(ratio[0], [expected], rtol=1e-5)


# --- random demand: invariants -------------------------------------------------------


def _random_lot(stock_ahead: float = 0.0, **overrides: Any) -> Lot:
    series = ("A|L1", "B|L1", "C|L2")
    base = np.array([1, 2, 4, 6, 8, 11, 13], dtype=np.float32)
    knots = (
        np.broadcast_to(base, (3, WEEKS, 7))
        * np.array([1, 0.5, 2], dtype=np.float32)[:, None, None]
    )
    args: dict[str, Any] = {
        "series": series,
        "categories": ("food", "food", "toys"),
        "qty": np.array([60, 30, 120], dtype=np.float32),
        "unit_cost": np.array([5, 8, 3], dtype=np.float32),
        "price": np.array([9, 15, 6], dtype=np.float32),
        "stock_ahead": np.full(3, stock_ahead, np.float32),
        "return_rate": np.array([0.05, 0, 0.1], dtype=np.float32),
        "knots": knots.copy(),
        "expected_receipt_date": HORIZON_START + timedelta(days=20),
        "tenors_days": (30, 60, 90),
    }
    return _lot(**{**args, **overrides})


def test_invariants_with_random_demand() -> None:
    sim = simulate_lot(_random_lot(), 3, SimulateSettings(n_paths=2000))
    ratio = sim.cash_ratio
    assert ratio.shape == (2000, 3)
    assert (ratio >= 0).all()
    assert (np.diff(ratio, axis=1) >= -1e-6).all()  # cash only accumulates
    assert ((sim.unsold_units >= 0) & (sim.unsold_units <= sim.lot.qty)).all()
    assert len(np.unique(sim.receipt_week)) > 1  # the receipt varies by path
    assert np.std(ratio[:, 2]) > 0


def test_same_seed_same_result() -> None:
    lot = _random_lot()
    a = simulate_lot(lot, 5, SimulateSettings(n_paths=500))
    b = simulate_lot(lot, 5, SimulateSettings(n_paths=500))
    np.testing.assert_array_equal(a.cash_ratio, b.cash_ratio)
    c = simulate_lot(lot, 6, SimulateSettings(n_paths=500))
    assert not np.array_equal(a.cash_ratio, c.cash_ratio)


def test_line_order_does_not_matter() -> None:
    lot = _random_lot()
    order = [2, 0, 1]
    shuffled = _random_lot(
        series=tuple(lot.series[i] for i in order),
        categories=tuple(lot.categories[i] for i in order),
        qty=lot.qty[order],
        unit_cost=lot.unit_cost[order],
        price=lot.price[order],
        return_rate=lot.return_rate[order],
        knots=lot.knots[order],
    )
    s = SimulateSettings(n_paths=500)
    np.testing.assert_allclose(
        simulate_lot(lot, 5, s).cash_ratio, simulate_lot(shuffled, 5, s).cash_ratio, rtol=1e-5
    )


@settings(max_examples=20, deadline=None)
@given(st.floats(0, 200), st.floats(0, 200))
def test_more_stock_ahead_never_raises_cash(a: float, b: float) -> None:
    # Holds for the selling mechanics, so markdowns are off. With them on it can
    # fail slightly: more stock ahead raises cover, which triggers an earlier
    # markdown, and with elasticity below -1 a markdown raises revenue.
    low, high = sorted((a, b))
    s = SimulateSettings(n_paths=200, markdown=NO_MARKDOWN)
    less = simulate_lot(_random_lot(stock_ahead=low), 9, s).cash_ratio
    more = simulate_lot(_random_lot(stock_ahead=high), 9, s).cash_ratio
    assert (more <= less + 1e-5).all()


def test_summary() -> None:
    summary = simulate_lot(_random_lot(), 3, SimulateSettings(n_paths=1000)).summary()
    cash_ratio = summary["cash_ratio"]
    assert isinstance(cash_ratio, dict)
    d90 = cash_ratio["d90"]
    assert list(d90) == ["q01", "q05", "q10", "q25", "q50", "q75", "q90"]
    assert d90["q05"] <= d90["q50"] <= d90["q90"]
    assert summary["lot_cost"] == pytest.approx(60 * 5 + 30 * 8 + 120 * 3)
    assert summary["seed"] == 3


# --- receipt distribution ----------------------------------------------------------------


def test_receipt_probabilities() -> None:
    on_hand = receipt_week_probs(_lot(), WEEKS, 0.3)
    assert on_hand[0] == 1

    po = _lot(expected_receipt_date=HORIZON_START + timedelta(days=44))
    exact = receipt_week_probs(po, WEEKS, 0.0)
    assert exact[7] == 1  # arrives in week 6, sells from week 7
    spread = receipt_week_probs(po, WEEKS, 0.3)
    assert spread.sum() == pytest.approx(1)
    assert spread[0] == 0
    cdf = np.cumsum(spread)
    assert cdf[6] < 0.5 <= cdf[7]  # the median is still week 7
    assert spread[WEEKS] > 0  # some paths see no stock within the horizon

    late = _lot(expected_receipt_date=HORIZON_START + timedelta(days=200))
    assert receipt_week_probs(late, WEEKS, 0.0)[WEEKS] == 1


# --- refusals ------------------------------------------------------------------------------


def test_tenor_beyond_the_forecast_is_refused() -> None:
    late = HORIZON_START + timedelta(days=3)
    lot = _lot(as_of=late, disbursement_date=late, tenors_days=(90,))  # days 4..93: week 14
    with pytest.raises(ValueError, match="needs 14 forecast weeks"):
        simulate_lot(lot, 1, SETTINGS)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"qty": np.array([0], dtype=np.float32)}, "positive qty"),
        ({"price": np.array([-1], dtype=np.float32)}, "non-negative"),
        ({"stock_ahead": np.array([1, 2], dtype=np.float32)}, "shape"),
        ({"return_rate": np.array([1], dtype=np.float32)}, "below 1"),
        ({"knots": _knots(1.0)[:, :, :6]}, "last axis"),
        ({"categories": ("a", "b")}, "categories"),
        ({"disbursement_date": AS_OF - timedelta(days=1)}, "before as_of"),
        ({"as_of": AS_OF - timedelta(days=1)}, "no later than the day after as_of"),
        ({"tenors_days": (0,)}, "positive"),
    ],
)
def test_bad_lots_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _lot(**overrides)


# --- forecast knots ------------------------------------------------------------------------------


def _forecast(series: list[tuple[str, str]], weeks: int) -> pl.DataFrame:
    rows = [
        {
            "model_version": "m",
            "as_of": HORIZON_START,
            "sku_id": sku,
            "location_id": loc,
            "horizon_week": h,
            **{col: float(h + j) for j, col in enumerate(QUANTILE_COLS)},
        }
        for sku, loc in series
        for h in range(1, weeks + 1)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("horizon_week").cast(pl.Int16))


def test_forecast_knots_follow_the_requested_order() -> None:
    forecast = _forecast([("A", "L1"), ("B", "L1")], 13)
    knots = forecast_knots(forecast, [("B", "L1"), ("A", "L1")], 13)
    assert knots.shape == (2, 13, 7)
    np.testing.assert_array_equal(knots[0, 2], np.arange(3, 10))
    assert series_key("A", "L1") == "A|L1"


def test_forecast_with_missing_weeks_is_refused() -> None:
    forecast = pl.concat(
        [_forecast([("A", "L1")], 13), _forecast([("B", "L1")], 8)]  # B stops at week 8
    )
    with pytest.raises(MissingForecastError, match=r"lacks 5 of 26 .*B\|L1 week 9"):
        forecast_knots(forecast, [("A", "L1"), ("B", "L1")], 13)
    with pytest.raises(MissingForecastError, match=r"C\|L1 week 1"):
        forecast_knots(forecast, [("C", "L1")], 1)
    # A shorter tenor that needs only 8 weeks is fine.
    assert forecast_knots(forecast, [("B", "L1")], 8).shape == (1, 8, 7)


def test_forecast_mixing_versions_is_refused() -> None:
    forecast = pl.concat(
        [
            _forecast([("A", "L1")], 2),
            _forecast([("A", "L1")], 2).with_columns(model_version=pl.lit("other")),
        ]
    )
    with pytest.raises(ValueError, match="model_version"):
        forecast_knots(forecast, [("A", "L1")], 2)
