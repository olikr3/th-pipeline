"""The cash-conversion simulator: a lot's cash ratio at each tenor, over N paths.

Per lot, all paths at once (arrays ``[paths, SKUs, weeks]``):

1. Correlated uniforms from the copula, turned into weekly demand by the inverse
   CDF of each SKU-week's forecast quantiles.
2. A receipt week per path. Goods arriving during a week sell from the next
   full week, so a lot never sells on the day it lands. Lots with goods on hand
   (stock financing) have no receipt delay.
3. Weekly selling: demand is served from the stock ahead of the lot first
   (FIFO), then from lot stock, and unserved demand is lost.
4. Markdowns: a SKU holding received lot stock is cut by ``markdown.step`` of its
   list price when the lot is older than ``markdown.age_weeks`` or its stock
   covers more than ``markdown.cover_weeks`` of median demand, at most once per
   ``markdown.interval_weeks`` and down to ``markdown.max_total``. Demand then
   scales by (price ratio) ** elasticity.
5. Cash from lot units only: units x marked-down price x (1 - return rate).

Tenors count from the disbursement date (theory/tenor-clock.md): a tenor of T
days covers the T days from the start of disbursement day. Cash inside a week is
taken as spread evenly over its days, so a tenor ending mid-week gets that
fraction of the week's cash. B2B receivable delays are not modelled yet.

Time is measured in days from ``horizon_start``, the first day of forecast
horizon week 1 (the day after as_of's last complete week).
"""

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from thp.core.config import SimulateSettings
from thp.core.seeding import make_rng
from thp.simulate.copula import Stream, correlated_uniforms
from thp.simulate.quantiles import FloatArray, check_knots, inverse_cdf

DEFAULT_TENORS = (30, 60, 90)
# Quantiles of the cash ratio a lot score reports.
SCORE_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
_P50 = 3  # index of q50 in the forecast knots

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class Lot:
    """A purchase order or stock position to score; per-line arrays have shape ``[S]``."""

    as_of: date
    horizon_start: date
    disbursement_date: date
    series: tuple[str, ...]
    categories: tuple[str, ...]
    qty: FloatArray  # lot units
    unit_cost: FloatArray
    price: FloatArray  # list price per unit
    stock_ahead: FloatArray  # units of the same SKU x location that sell before the lot's
    return_rate: FloatArray
    knots: FloatArray  # [S, weeks, 7] forecast quantiles for horizon weeks 1, 2, ...
    # None: the goods are on hand (stock financing). Otherwise the order's
    # expected receipt; the lead time is counted from as_of, the order date.
    expected_receipt_date: date | None = None
    tenors_days: tuple[int, ...] = DEFAULT_TENORS

    def __post_init__(self) -> None:
        n = len(self.series)
        if len(self.categories) != n:
            raise ValueError(f"{n} series but {len(self.categories)} categories")
        for name in ("qty", "unit_cost", "price", "stock_ahead", "return_rate"):
            values = getattr(self, name)
            if values.shape != (n,):
                raise ValueError(f"{name} has shape {values.shape}, expected ({n},)")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{name} must be finite and non-negative")
        if not ((self.qty > 0) & (self.unit_cost > 0)).all():
            raise ValueError("every line needs a positive qty and unit_cost")
        if (self.return_rate >= 1).any():
            raise ValueError("return_rate must be below 1")
        check_knots(self.knots)
        if self.knots.ndim != 3 or self.knots.shape[0] != n:
            raise ValueError(f"knots have shape {self.knots.shape}, expected ({n}, weeks, 7)")
        if self.as_of < self.horizon_start:
            raise ValueError("as_of lies before the forecast horizon")
        if self.disbursement_date < self.as_of:
            raise ValueError("disbursement_date lies before as_of")
        if not self.tenors_days or min(self.tenors_days) <= 0:
            raise ValueError("tenors_days must be positive")

    @property
    def cost(self) -> float:
        return float((self.qty * self.unit_cost).sum())

    @property
    def weeks_needed(self) -> int:
        """Forecast weeks from horizon_start to the end of the longest tenor."""
        start = (self.disbursement_date - self.horizon_start).days
        return math.ceil((start + max(self.tenors_days)) / 7)


@dataclass(frozen=True, eq=False)
class LotSimulation:
    lot: Lot
    seed: int
    cash_ratio: FloatArray  # [N, tenors]: lot cash in the tenor / lot cost
    unsold_units: FloatArray  # [N, S]: lot units unsold at the end of the longest tenor
    receipt_week: IntArray = field(repr=False)  # [N]: first week index lot stock can sell

    def summary(self) -> dict[str, object]:
        ratio = self.cash_ratio.astype(np.float64)
        quantiles = np.quantile(ratio, SCORE_QUANTILES, axis=0)
        unsold_value = (self.unsold_units * self.lot.unit_cost).sum(axis=1)
        return {
            "cash_ratio": {
                f"d{tenor}": {
                    f"q{round(q * 100):02d}": float(quantiles[i, j])
                    for i, q in enumerate(SCORE_QUANTILES)
                }
                for j, tenor in enumerate(self.lot.tenors_days)
            },
            "p_cash_ratio_below_1": {
                f"d{tenor}": float((ratio[:, j] < 1).mean())
                for j, tenor in enumerate(self.lot.tenors_days)
            },
            "expected_unsold_value": float(unsold_value.mean()),
            "lot_cost": self.lot.cost,
            "seed": self.seed,
        }


def simulate_lot(lot: Lot, seed: int, settings: SimulateSettings) -> LotSimulation:
    n_weeks = lot.weeks_needed
    if n_weeks > lot.knots.shape[1]:
        raise ValueError(
            f"the {max(lot.tenors_days)}-day tenor needs {n_weeks} forecast weeks, "
            f"the lot has {lot.knots.shape[1]}"
        )
    knots = lot.knots[:, :n_weeks]
    n = settings.n_paths
    md = settings.markdown

    u = correlated_uniforms(seed, lot.series, lot.categories, n_weeks, settings)
    demand = inverse_cdf(knots, u)
    probs = receipt_week_probs(lot, n_weeks, settings.receipt_sigma)
    receipt = make_rng(seed, Stream.RECEIPT).choice(probs.size, size=n, p=probs)

    shape = (n, len(lot.series))
    ahead = np.broadcast_to(lot.stock_ahead, shape).astype(np.float32)
    lot_left = np.broadcast_to(lot.qty, shape).astype(np.float32)
    cut = np.zeros(shape, dtype=np.float32)
    last_cut = np.full(shape, -md.interval_weeks, dtype=np.int64)
    weekly_p50 = knots[:, :, _P50].mean(axis=1)
    net_price = lot.price * (1 - lot.return_rate)

    sold = np.zeros((n, len(lot.series), n_weeks), dtype=np.float32)
    cash = np.zeros((n, n_weeks), dtype=np.float32)
    for t in range(n_weeks):
        received = (receipt <= t)[:, None]
        on_hand = np.where(received, lot_left, 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            cover = np.where(weekly_p50 > 0, (ahead + on_hand) / weekly_p50, np.inf)
        old = (t - receipt)[:, None] > md.age_weeks
        due = (
            (on_hand > 0)
            & (old | (cover > md.cover_weeks))
            & (t - last_cut >= md.interval_weeks)
            & (cut < md.max_total)
        )
        cut = np.where(due, np.minimum(cut + md.step, md.max_total), cut).astype(np.float32)
        last_cut = np.where(due, t, last_cut)

        wanted = demand[:, :, t] * (1 - cut) ** md.elasticity
        from_ahead = np.minimum(wanted, ahead)
        ahead -= from_ahead
        from_lot = np.minimum(wanted - from_ahead, on_hand)
        lot_left -= from_lot
        sold[:, :, t] = from_lot
        cash[:, t] = (from_lot * net_price * (1 - cut)).sum(axis=1)

    start = (lot.disbursement_date - lot.horizon_start).days
    ends = [start + tenor for tenor in lot.tenors_days]
    cash_ratio = np.stack(
        [_through(cash, end) - _through(cash, start) for end in ends], axis=1
    ) / np.float32(lot.cost)
    sold_in_tenor = _through(sold, max(ends)) - _through(sold, start)
    return LotSimulation(
        lot=lot,
        seed=seed,
        cash_ratio=cash_ratio.astype(np.float32),
        unsold_units=np.maximum(lot.qty - sold_in_tenor, 0).astype(np.float32),
        receipt_week=receipt,
    )


def receipt_week_probs(lot: Lot, n_weeks: int, sigma: float) -> npt.NDArray[np.float64]:
    """P(lot stock first sells in week j) for j = 0..n_weeks; j = n_weeks means not in the horizon.

    The lead time from as_of is lognormal with its median at the expected receipt
    and log-spread ``sigma``. Goods arriving in week k sell from week k + 1.
    """
    if lot.expected_receipt_date is None:
        on_hand: npt.NDArray[np.float64] = np.eye(n_weeks + 1)[0]
        return on_hand

    ordered = (lot.as_of - lot.horizon_start).days
    lead = max((lot.expected_receipt_date - lot.as_of).days, 0)
    # Selling from week <= k means arriving before day 7k, for k = 1..n_weeks-1.
    days_after_order = 7.0 * np.arange(1, n_weeks) - ordered
    if lead == 0 or sigma == 0:
        arrived = (days_after_order > lead).astype(np.float64)
    else:
        with np.errstate(divide="ignore"):
            arrived = ndtr(np.log(np.maximum(days_after_order, 0) / lead) / sigma)
    cdf = np.concatenate([[0.0], arrived, [1.0]])
    probs: npt.NDArray[np.float64] = np.diff(cdf, prepend=0.0)
    probs /= probs.sum()
    return probs


def _through(weekly: FloatArray, day: int) -> FloatArray:
    """Cumulative sum of weekly amounts (last axis) up to ``day`` days into the horizon."""
    weeks, rest = divmod(day, 7)
    total = weekly[..., :weeks].sum(axis=-1)
    if rest and weeks < weekly.shape[-1]:
        total = total + weekly[..., weeks] * np.float32(rest / 7)
    return total
