"""Correlated uniforms for the simulator: a Gaussian factor copula.

Per path n, SKU i (in category c) and week t:

    x[n,i,t] = sqrt(rho_m) * zm[n,t] + sqrt(rho_c) * zc[n,c,t]
             + sqrt(rho_s) * zs[n,i,t] + sqrt(1 - rho_m - rho_c - rho_s) * eps[n,i,t]
    u[n,i,t] = Phi(x[n,i,t])

zm (merchant) and zc (category) are shared shocks, AR(1) over weeks with
``shock_persistence``; zs is the SKU's own persistent level error, AR(1) with
``sku_persistence``; eps is independent noise. Every term is standard normal, so
x is too and each u is exactly uniform: feeding u to ``inverse_cdf`` keeps every
SKU-week's forecast distribution while correlating the outcomes. See
theory/copula.md and theory/simulator-draws.md.

Each draw comes from its own substream of the seed, keyed by what it belongs to
(the merchant, a category id, a series id) rather than by position. A SKU's
uniforms therefore do not depend on which other SKUs are simulated with it or in
what order, so a stock position can be simulated in blocks that all share one
merchant and category future.
"""

import hashlib
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from thp.core.config import SimulateSettings
from thp.core.seeding import make_rng

FloatArray = npt.NDArray[np.float32]


class Stream(IntEnum):
    """First substream key per kind of draw. Append new kinds; never renumber."""

    MERCHANT = 0
    CATEGORY = 1
    SKU_LEVEL = 2
    NOISE = 3
    RECEIPT = 4


def stream_key(name: str) -> int:
    """A stable 64-bit substream key for an id (``hash()`` changes per process)."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


def correlated_uniforms(
    seed: int,
    series: Sequence[str],
    categories: Sequence[str],
    n_weeks: int,
    settings: SimulateSettings,
) -> FloatArray:
    """Uniforms shaped ``[n_paths, len(series), n_weeks]``.

    ``series`` identifies each SKU x location (e.g. ``"sku|location"``) and must
    be unique; ``categories`` gives each one's category id.
    """
    if not series:
        raise ValueError("no series to simulate")
    if len(series) != len(categories):
        raise ValueError(f"{len(series)} series but {len(categories)} categories")
    if len(set(series)) != len(series):
        raise ValueError("series must be unique: one row per SKU x location")
    if n_weeks < 1:
        raise ValueError(f"n_weeks must be positive, got {n_weeks}")

    shape = (settings.n_paths, n_weeks)
    phi, phi_sku = settings.shock_persistence, settings.sku_persistence

    def draw(*stream: int) -> FloatArray:
        return make_rng(seed, *stream).standard_normal(shape, dtype=np.float32)

    zm = ar1(draw(Stream.MERCHANT), phi)
    names, cat_index = np.unique(np.asarray(categories, dtype=object), return_inverse=True)
    zc = np.stack([ar1(draw(Stream.CATEGORY, stream_key(c)), phi) for c in names], axis=1)
    keys = [stream_key(s) for s in series]
    zs = np.stack([ar1(draw(Stream.SKU_LEVEL, k), phi_sku) for k in keys], axis=1)
    eps = np.stack([draw(Stream.NOISE, k) for k in keys], axis=1)

    rho_m, rho_c, rho_s = settings.rho_merchant, settings.rho_category, settings.rho_sku
    x = (
        np.float32(np.sqrt(rho_m)) * zm[:, None, :]
        + np.float32(np.sqrt(rho_c)) * zc[:, cat_index, :]
        + np.float32(np.sqrt(rho_s)) * zs
        + np.float32(np.sqrt(1 - rho_m - rho_c - rho_s)) * eps
    )
    u: FloatArray = ndtr(x).astype(np.float32, copy=False)
    return u


def ar1(innovations: FloatArray, phi: float) -> FloatArray:
    """Stationary AR(1) along the last axis with unit variance in every step.

    ``z[0] = e[0]`` and ``z[t] = phi * z[t-1] + sqrt(1 - phi^2) * e[t]``.
    """
    out = innovations.copy()
    scale = np.float32(np.sqrt(1 - phi * phi))
    for t in range(1, out.shape[-1]):
        out[..., t] = np.float32(phi) * out[..., t - 1] + scale * innovations[..., t]
    return out
