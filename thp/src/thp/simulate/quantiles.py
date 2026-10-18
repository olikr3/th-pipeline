"""Demand samples from forecast quantiles: the inverse CDF the simulator evaluates.

A forecast gives the quantile function Q(u) at the levels in ``QUANTILES`` (the
knots). Between knots Q is linear in u. Beyond the outer knots the tails are
exponential, continuing the slope per halving of tail probability set by the two
outermost knots on each side:

    upper (u > 0.95):  Q(u) = q95 + b_hi * ln(0.05 / (1 - u)),  b_hi = (q95 - q90) / ln 2
    lower (u < 0.05):  Q(u) = q05 - b_lo * ln(0.05 / u),        b_lo = (q10 - q05) / ln 2

and Q is clipped at zero. Repeated knots (e.g. q05 = q25 = 0 for an intermittent
SKU) give a flat segment, which is the point mass at that value. See
theory/cash-conversion-simulator.md.
"""

import numpy as np
import numpy.typing as npt

from thp.schema import QUANTILES

GRID = np.array(QUANTILES, dtype=np.float32)
# Keeps the tail logs finite and 1 - u accurate in float32 (spacing ~6e-8 near 1).
# Q(1 - U_EPS) is about 9 halvings of tail probability past q95.
U_EPS = 1e-4

FloatArray = npt.NDArray[np.float32]


def check_knots(knots: npt.ArrayLike) -> FloatArray:
    """``knots`` as float32, checked: last axis one value per level, sorted, finite, >= 0."""
    out = np.asarray(knots, dtype=np.float32)
    if out.ndim == 0 or out.shape[-1] != GRID.size:
        raise ValueError(f"knots need a last axis of {GRID.size} quantiles, got shape {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError("knots must be finite")
    if (out < 0).any():
        raise ValueError("knots must be non-negative")
    if (np.diff(out, axis=-1) < 0).any():
        raise ValueError("knots must be non-decreasing along the quantile axis")
    return out


def inverse_cdf(knots: FloatArray, u: npt.ArrayLike) -> FloatArray:
    """Q(u) per element of ``u``.

    ``knots`` has shape ``[..., len(QUANTILES)]`` and is assumed checked (see
    ``check_knots``); its leading axes broadcast against ``u``, e.g. knots
    ``[S, W, 7]`` and u ``[N, S, W]`` give demand ``[N, S, W]``.
    """
    u = np.clip(np.asarray(u, dtype=np.float32), U_EPS, 1 - U_EPS)
    shape = np.broadcast_shapes(u.shape, knots.shape[:-1])
    u = np.broadcast_to(u, shape)
    k = np.broadcast_to(knots, (*shape, GRID.size))

    # Segment j holds GRID[j] <= u < GRID[j + 1]; tails reuse the outer segments
    # and are replaced below.
    j = np.clip(np.searchsorted(GRID, u, side="right") - 1, 0, GRID.size - 2)[..., None]
    lo = np.take_along_axis(k, j, axis=-1)[..., 0]
    hi = np.take_along_axis(k, j + 1, axis=-1)[..., 0]
    g_lo, g_hi = GRID[j[..., 0]], GRID[j[..., 0] + 1]
    out = lo + (u - g_lo) / (g_hi - g_lo) * (hi - lo)

    p_hi = 1 - GRID[-1]
    slope_hi = (k[..., -1] - k[..., -2]) / np.log((1 - GRID[-2]) / p_hi)
    upper = k[..., -1] + slope_hi * np.log(p_hi / (1 - u))
    slope_lo = (k[..., 1] - k[..., 0]) / np.log(GRID[1] / GRID[0])
    lower = k[..., 0] - slope_lo * np.log(GRID[0] / u)

    out = np.where(u > GRID[-1], upper, np.where(u < GRID[0], lower, out))
    result: FloatArray = np.maximum(out, 0).astype(np.float32, copy=False)
    return result
