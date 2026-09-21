"""Conformal symplecticity of the damped flow (R1), integrated."""

from __future__ import annotations

import numpy as np

from dlkin import conformal_symplectic_flow_test


def test_conformal_symplectic_flow() -> None:
    """Phi(T)^T J Phi(T) = e^{-gamma T} J for an arbitrary resampled symmetric K(t)."""
    relerr = conformal_symplectic_flow_test()
    assert relerr < 1e-12


def test_conformal_symplectic_flow_is_deterministic() -> None:
    """Fixed seed, local generator: the same call gives bitwise the same answer."""
    first = conformal_symplectic_flow_test(n=4, T=0.05, dt=1e-3, seed=3)
    second = conformal_symplectic_flow_test(n=4, T=0.05, dt=1e-3, seed=3)
    assert first == second
    assert conformal_symplectic_flow_test(n=4, T=0.05, dt=1e-3, seed=4) != first


def test_conformal_factor_tracks_gamma_and_T() -> None:
    """The contraction factor is e^{-gamma T}; check it is not accidentally 1."""
    for gamma, T in ((0.37, 0.9), (0.10, 1.2)):
        relerr = conformal_symplectic_flow_test(n=4, gamma=gamma, T=T, dt=1e-3, seed=2)
        assert relerr < 1e-12
        assert abs(np.exp(-gamma * T) - 1.0) > 0.1
