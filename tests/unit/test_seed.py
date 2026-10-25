"""Unit tests for random seeding and determinism."""

import random

import numpy as np

from src.utils.seed import set_seed


def test_seed_determinism() -> None:
    set_seed(12345)
    r1 = random.random()
    n1 = np.random.rand(5)

    # Re-seed with the same seed
    set_seed(12345)
    r2 = random.random()
    n2 = np.random.rand(5)

    assert r1 == r2
    np.testing.assert_array_equal(n1, n2)


def test_seed_status_dict() -> None:
    info = set_seed(42)
    assert info["seed"] == 42
    assert info["python_random"] is True
    assert info["numpy"] is True
    assert "pytorch" in info
