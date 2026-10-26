from datetime import date

import numpy as np
import pytest

from thp.core.config import UnderwriteSettings
from thp.simulate.engine import Lot, LotSimulation
from thp.underwrite.advance import advance

SETTINGS = UnderwriteSettings()


def _simulation(ratios_90: np.ndarray) -> LotSimulation:
    lot = Lot(
        as_of=date(2026, 1, 4),
        horizon_start=date(2026, 1, 5),
        disbursement_date=date(2026, 1, 4),
        series=("A|L1",),
        categories=("food",),
        qty=np.array([100], dtype=np.float32),
        unit_cost=np.array([10], dtype=np.float32),  # lot cost 1,000
        price=np.array([20], dtype=np.float32),
        stock_ahead=np.zeros(1, dtype=np.float32),
        return_rate=np.zeros(1, dtype=np.float32),
        knots=np.ones((1, 13, 7), dtype=np.float32),
        tenors_days=(30, 90),
    )
    n = ratios_90.size
    cash_ratio = np.stack([ratios_90 / 2, ratios_90], axis=1).astype(np.float32)
    return LotSimulation(
        lot=lot,
        seed=1,
        cash_ratio=cash_ratio,
        unsold_units=np.zeros((n, 1), dtype=np.float32),
        receipt_week=np.zeros(n, dtype=np.int64),
    )


def test_advance_is_the_low_quantile_of_cash_net_of_fee() -> None:
    sim = _simulation(np.linspace(0.0, 1.0, 101))  # q05 = 0.05
    offer = advance(sim, 90, 0.03, SETTINGS)
    assert offer.cash_ratio_quantile == pytest.approx(0.05)
    assert offer.amount == pytest.approx(0.05 * 1000 / 1.03)
    assert offer.rate == pytest.approx(0.05 / 1.03)
    assert not offer.capped


def test_advance_is_capped_at_the_max_rate() -> None:
    sim = _simulation(np.full(100, 1.5))
    offer = advance(sim, 90, 0.03, SETTINGS)
    assert offer.uncapped == pytest.approx(1.5 * 1000 / 1.03)
    assert offer.amount == pytest.approx(800)
    assert offer.capped


def test_confidence_override_and_tenor() -> None:
    sim = _simulation(np.linspace(0.0, 1.0, 101))
    assert advance(sim, 90, 0.0, SETTINGS, confidence=0.5).amount == pytest.approx(500)
    assert advance(sim, 30, 0.0, SETTINGS).cash_ratio_quantile == pytest.approx(0.025)
    with pytest.raises(ValueError, match="not simulated"):
        advance(sim, 60, 0.0, SETTINGS)
