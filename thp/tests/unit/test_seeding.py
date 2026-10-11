import numpy as np

from thp.core.seeding import make_rng


def test_same_seed_gives_same_draws() -> None:
    np.testing.assert_array_equal(make_rng(42).normal(size=8), make_rng(42).normal(size=8))


def test_streams_are_reproducible_and_distinct() -> None:
    np.testing.assert_array_equal(make_rng(42, 3).normal(size=8), make_rng(42, 3).normal(size=8))
    assert not np.array_equal(make_rng(42, 3).normal(size=8), make_rng(42, 4).normal(size=8))
    assert not np.array_equal(make_rng(42).normal(size=8), make_rng(43).normal(size=8))
