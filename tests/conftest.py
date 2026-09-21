"""Shared fixtures.  Everything here is deterministic -- seeded RNG only."""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import Grid, LatticeModel


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator, so every test is reproducible run to run."""
    return np.random.default_rng(20240921)


@pytest.fixture
def small_grid() -> Grid:
    """A cheap grid for operator-level and Jacobian checks."""
    return Grid(L=50.0, N=256)


@pytest.fixture
def fk_model() -> LatticeModel:
    """The Frenkel-Kontorova defaults of the note: mu = 1, gamma = 0.1, w = 2."""
    return LatticeModel()
