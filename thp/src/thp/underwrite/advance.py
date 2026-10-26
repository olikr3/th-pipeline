"""The advance amount: the largest loan the lot's cash covers with the given confidence.

    L* = q_(1 - confidence)(cash_ratio at the tenor) x lot cost / (1 + fee)

capped at ``max_advance_rate`` of lot cost. This is the spec's core formula only:
the liquidation floor and the decision rules (data trust, cover, cold start,
shortfall probability, minimum size) are Milestone 6.
"""

from dataclasses import dataclass

import numpy as np

from thp.core.config import UnderwriteSettings
from thp.simulate.engine import LotSimulation


@dataclass(frozen=True)
class Advance:
    tenor_days: int
    confidence: float
    cash_ratio_quantile: float  # q_(1 - confidence) of the cash ratio at the tenor
    uncapped: float  # L*
    amount: float  # min(L*, cap)
    rate: float  # amount / lot cost
    capped: bool


def advance(
    simulation: LotSimulation,
    tenor_days: int,
    fee_rate: float,
    settings: UnderwriteSettings,
    confidence: float | None = None,
) -> Advance:
    tenors = simulation.lot.tenors_days
    if tenor_days not in tenors:
        raise ValueError(f"tenor {tenor_days} days was not simulated (have {tenors})")
    confidence = settings.confidence if confidence is None else confidence
    ratios = simulation.cash_ratio[:, tenors.index(tenor_days)].astype(np.float64)
    quantile = float(np.quantile(ratios, 1 - confidence))
    cost = simulation.lot.cost
    uncapped = max(quantile, 0.0) * cost / (1 + fee_rate)
    cap = settings.max_advance_rate * cost
    amount = min(uncapped, cap)
    return Advance(
        tenor_days=tenor_days,
        confidence=confidence,
        cash_ratio_quantile=quantile,
        uncapped=uncapped,
        amount=amount,
        rate=amount / cost,
        capped=uncapped > cap,
    )
