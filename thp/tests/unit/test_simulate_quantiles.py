import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from thp.simulate.quantiles import GRID, U_EPS, check_knots, inverse_cdf

# The worked example in theory/cash-conversion-simulator.md.
EXAMPLE = np.array([2, 4, 7, 10, 14, 19, 23], dtype=np.float32)


def _knots(values: list[float]) -> np.ndarray:
    return check_knots(np.cumsum(values))


# Sorted, non-negative knots as increments; zeros give repeated knots.
increments = st.lists(
    st.one_of(st.just(0.0), st.floats(min_value=0.0, max_value=1000.0)),
    min_size=GRID.size,
    max_size=GRID.size,
)


def test_worked_example() -> None:
    q = inverse_cdf(EXAMPLE, [0.6, 0.98, 0.02])
    assert q[0] == pytest.approx(11.6)
    assert q[1] == pytest.approx(23 + 4 / np.log(2) * np.log(2.5), rel=1e-5)
    assert q[2] == 0.0  # the lower tail runs below zero: clipped


@given(increments)
def test_knots_are_reproduced(values: list[float]) -> None:
    knots = _knots(values)
    np.testing.assert_allclose(inverse_cdf(knots, GRID), knots, rtol=1e-5, atol=1e-3)


@given(increments, st.lists(st.floats(0.0, 1.0), min_size=2, max_size=50))
def test_monotone_in_u(values: list[float], us: list[float]) -> None:
    q = inverse_cdf(_knots(values), np.sort(np.array(us, dtype=np.float32)))
    assert (np.diff(q) >= -1e-3 * (1 + np.abs(q[1:]))).all()
    assert (q >= 0).all()
    assert np.isfinite(q).all()


@given(increments)
def test_continuous_at_the_tail_boundaries(values: list[float]) -> None:
    knots = _knots(values)
    q = inverse_cdf(knots, [GRID[0] - 1e-5, GRID[-1] + 1e-5])
    # A 1e-5 step in u moves Q by at most ~(q95 - q90) / 0.05 * 1e-5.
    np.testing.assert_allclose(q, [knots[0], knots[-1]], atol=1e-3 * (1 + knots[-1]))


def test_uniform_draws_reproduce_the_forecast_quantiles() -> None:
    rng = np.random.default_rng(0)
    samples = inverse_cdf(EXAMPLE, rng.random(200_000, dtype=np.float32))
    np.testing.assert_allclose(np.quantile(samples, GRID), EXAMPLE, rtol=0.02)


def test_repeated_knots_give_a_point_mass() -> None:
    intermittent = np.array([0, 0, 0, 1, 3, 6, 8], dtype=np.float32)
    u = np.linspace(0.0, 0.5, 101, dtype=np.float32)
    q = inverse_cdf(intermittent, u)
    assert (q[u <= 0.25] == 0).all()
    assert (q[u > 0.25] > 0).all()


def test_tails_follow_the_outer_slopes() -> None:
    # Each halving of tail probability repeats the outermost step: q90 -> q95 is
    # +4 units above, q05 -> q10 is 2 units below (shifted up so 0 is not hit).
    np.testing.assert_allclose(inverse_cdf(EXAMPLE, [0.975, 0.9875]), [27.0, 31.0], rtol=1e-5)
    np.testing.assert_allclose(inverse_cdf(EXAMPLE + 10, [0.025, 0.0125]), [10.0, 8.0], rtol=1e-5)


def test_extreme_u_stays_finite() -> None:
    q = inverse_cdf(EXAMPLE, [0.0, 1.0])
    assert q[0] == 0.0
    expected_top = 23 + 4 / np.log(2) * np.log(0.05 / U_EPS)
    assert q[1] == pytest.approx(expected_top, rel=1e-4)


def test_broadcasts_paths_over_skus_and_weeks() -> None:
    n, s, w = 5, 3, 4
    knots = check_knots(
        np.broadcast_to(EXAMPLE, (s, w, GRID.size)) * np.arange(1, s + 1)[:, None, None]
    )
    u = np.random.default_rng(1).random((n, s, w), dtype=np.float32)
    q = inverse_cdf(knots, u)
    assert q.shape == (n, s, w)
    assert q.dtype == np.float32
    expected = inverse_cdf(knots[2, 1], u[:, 2, 1])
    np.testing.assert_allclose(q[:, 2, 1], expected)


@pytest.mark.parametrize(
    ("knots", "message"),
    [
        (EXAMPLE[:6], "last axis"),
        (np.float32(1.0), "last axis"),
        (EXAMPLE[::-1], "non-decreasing"),
        (EXAMPLE - 3, "non-negative"),
        (np.where(np.arange(7) == 3, np.nan, EXAMPLE), "finite"),
    ],
)
def test_check_knots_rejects(knots: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_knots(knots)
