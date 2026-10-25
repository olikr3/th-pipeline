import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import ndtri

from thp.core.config import SimulateSettings
from thp.simulate.copula import ar1, correlated_uniforms, stream_key

SERIES = ["A|L1", "B|L1", "C|L1", "D|L2"]
CATEGORIES = ["food", "food", "toys", "toys"]
WEEKS = 13
# Enough paths for correlations to be within ~0.02 of their target.
SETTINGS = SimulateSettings(n_paths=20_000)


@pytest.fixture(scope="module")
def u() -> np.ndarray:
    return correlated_uniforms(7, SERIES, CATEGORIES, WEEKS, SETTINGS)


@pytest.fixture(scope="module")
def x(u: np.ndarray) -> np.ndarray:
    """Back on the latent normal scale."""
    return ndtri(u.astype(np.float64))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_shape_and_range(u: np.ndarray) -> None:
    assert u.shape == (SETTINGS.n_paths, len(SERIES), WEEKS)
    assert u.dtype == np.float32
    assert ((u > 0) & (u < 1)).all()


def test_every_sku_week_is_uniform(u: np.ndarray) -> None:
    # The copula must not distort any forecast: each u[:, i, t] is Uniform(0, 1),
    # including late weeks, where a wrongly scaled AR(1) would have drifted.
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    for i in range(len(SERIES)):
        for t in (0, 6, WEEKS - 1):
            np.testing.assert_allclose(np.quantile(u[:, i, t], levels), levels, atol=0.015)


def test_correlation_structure(x: np.ndarray) -> None:
    s = SETTINGS
    shared_cat = s.rho_merchant + s.rho_category
    a, b, c = 0, 1, 2  # A and B share a category; C is in another
    t = 4

    assert _corr(x[:, a, t], x[:, b, t]) == pytest.approx(shared_cat, abs=0.03)
    assert _corr(x[:, a, t], x[:, c, t]) == pytest.approx(s.rho_merchant, abs=0.03)
    # Other SKUs k weeks apart: only the shared shocks connect them, decaying with phi.
    for k in (1, 4):
        lagged = _corr(x[:, a, t], x[:, b, t + k])
        assert lagged == pytest.approx(shared_cat * s.shock_persistence**k, abs=0.03)
    # The same SKU k weeks apart also keeps its own level, which decays slowly.
    for k in (1, 4, 8):
        own = shared_cat * s.shock_persistence**k + s.rho_sku * s.sku_persistence**k
        assert _corr(x[:, a, t], x[:, a, t + k]) == pytest.approx(own, abs=0.03)


def test_zero_rhos_give_independent_draws() -> None:
    settings = SimulateSettings(n_paths=20_000, rho_merchant=0, rho_category=0, rho_sku=0)
    x = ndtri(correlated_uniforms(7, SERIES, CATEGORIES, 3, settings).astype(np.float64))
    assert _corr(x[:, 0, 0], x[:, 1, 0]) == pytest.approx(0, abs=0.03)
    assert _corr(x[:, 0, 0], x[:, 0, 1]) == pytest.approx(0, abs=0.03)


def test_same_seed_same_draws(u: np.ndarray) -> None:
    np.testing.assert_array_equal(u, correlated_uniforms(7, SERIES, CATEGORIES, WEEKS, SETTINGS))
    other = correlated_uniforms(8, SERIES, CATEGORIES, WEEKS, SETTINGS)
    assert not np.array_equal(u, other)


def test_draws_do_not_depend_on_the_other_skus_or_their_order(u: np.ndarray) -> None:
    # A block with only D and A, reversed: each SKU sees the same future as in the
    # full run, including the shared merchant and category shocks.
    block = correlated_uniforms(7, ["D|L2", "A|L1"], ["toys", "food"], WEEKS, SETTINGS)
    np.testing.assert_array_equal(block[:, 0], u[:, 3])
    np.testing.assert_array_equal(block[:, 1], u[:, 0])


def test_ar1_keeps_unit_variance_and_lag_correlation() -> None:
    e = np.random.default_rng(0).standard_normal((200_000, 10), dtype=np.float32)
    z = ar1(e, 0.7)
    np.testing.assert_array_equal(z[:, 0], e[:, 0])
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=0.01)
    assert _corr(z[:, 3], z[:, 5]) == pytest.approx(0.49, abs=0.01)


def test_stream_key_is_stable() -> None:
    # Fixed across processes and versions: changing it would change every result.
    assert stream_key("A|L1") == stream_key("A|L1")
    assert stream_key("A|L1") != stream_key("A|L2")
    assert stream_key("food") == 0xCF4369F2A576D08C


@pytest.mark.parametrize(
    ("series", "categories", "weeks", "message"),
    [
        ([], [], WEEKS, "no series"),
        (["A|L1", "B|L1"], ["food"], WEEKS, "categories"),
        (["A|L1", "A|L1"], ["food", "food"], WEEKS, "unique"),
        (["A|L1"], ["food"], 0, "n_weeks"),
    ],
)
def test_rejects_bad_input(
    series: list[str], categories: list[str], weeks: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        correlated_uniforms(7, series, categories, weeks, SimulateSettings(n_paths=10))


def test_settings_reject_rhos_that_leave_no_noise() -> None:
    with pytest.raises(ValidationError, match="below 1"):
        SimulateSettings(rho_merchant=0.4, rho_category=0.3, rho_sku=0.3)
