"""Reproducible random streams."""

import numpy as np


def make_rng(seed: int, *stream: int) -> np.random.Generator:
    """PCG64 generator for ``seed``.

    ``stream`` selects an independent substream of the same seed, e.g. one per
    SKU block, so chunked runs do not depend on chunk execution order.
    """
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=stream)))
