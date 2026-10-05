"""Structural identities: the pencil duality and conformal symplecticity.

These are the two propositions the whole note rests on, and both are exact operator
statements rather than numerical observations -- so they are testable to machine precision,
and a failure here means the discretization has stopped representing the mathematics.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import (
    Grid,
    LatticeModel,
    TravelingWaveSolver,
    conformal_symplectic_flow_test,
    init_branch,
    lattice_monodromy,
    sample_profile_at_sites,
    symplectic_form,
)


@pytest.fixture(scope="module")
def small_operators():
    """``M0`` and ``M1`` on a coarse grid about an arbitrary profile.

    The duality ``Q(-gamma-nu) = Q(nu)^T`` is an identity of the operators, not of the
    solution, so an arbitrary smooth ``psi`` exercises it exactly as a converged one would --
    and using one makes the test independent of the solver.
    """
    grid = Grid(L=20.0, N=256)
    model = LatticeModel(gamma=0.17, mu=1.3, mu2=-0.4, couplings=((1, 1.0), (2, 0.3)))
    solver = TravelingWaveSolver(grid, model)
    c = 0.71
    solver.build(c)
    psi = 0.4 * np.sin(2.0 * np.pi * grid.xi / (2.0 * grid.L)) + 0.1
    return solver, psi, c, model


def test_pencil_duality_is_exact(small_operators) -> None:
    """``Q(-gamma-nu) = Q(nu)^T`` for every ``nu`` -- Proposition 3.2 (R2)."""
    solver, psi, _, model = small_operators
    M0, M1 = solver.M0(psi), solver.M1
    eye = np.eye(M0.shape[0])
    gamma = model.gamma

    def Q(nu):
        return nu * nu * eye + nu * M1 + M0

    scale = np.max(np.abs(M0))
    for nu in (0.0, 0.31, -0.22, 0.4 + 0.9j, -1.7 - 0.3j):
        residual = np.max(np.abs(Q(-gamma - nu) - Q(nu).T))
        assert residual < 1e-12 * scale, f"duality failed at nu={nu}: {residual:.3e}"


def test_duality_at_zero_is_the_anti_damped_operator(small_operators) -> None:
    """``Q(-gamma) = M0^T``: the left neutral vector solves the ``gamma -> -gamma`` equation."""
    solver, psi, c, model = small_operators
    M0, M1 = solver.M0(psi), solver.M1
    eye = np.eye(M0.shape[0])
    gamma = model.gamma
    Q_minus_gamma = gamma * gamma * eye - gamma * M1 + M0
    assert np.max(np.abs(Q_minus_gamma - M0.T)) < 1e-12 * np.max(np.abs(M0))


def test_infinitesimal_conformal_symplecticity() -> None:
    """``A^T J + J A = -gamma J`` for any symmetric ``K`` -- the content of Proposition 3.1."""
    rng = np.random.default_rng(3)
    n, gamma = 11, 0.29
    J = symplectic_form(n)
    K = rng.standard_normal((n, n))
    K = 0.5 * (K + K.T)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [K, -gamma * np.eye(n)]])
    assert np.max(np.abs(A.T @ J + J @ A + gamma * J)) < 1e-12


def test_conformal_symplectic_flow_and_shift() -> None:
    """The integrated form, with ``K(t)`` resampled every step and the one-site shift."""
    result = conformal_symplectic_flow_test(n=6, gamma=0.37, T=0.5, dt=1e-3, seed=2)
    assert result["relerr"] < 1e-12
    # the shift is a permutation, hence symplectic: it must not change the residual
    assert abs(result["relerr"] - result["relerr_without_shift"]) < 1e-14


def test_sampling_reproduces_the_grid_values() -> None:
    """Trigonometric interpolation at grid points returns the grid values."""
    grid = Grid(L=40.0, N=512)
    model = LatticeModel()
    solver = TravelingWaveSolver(grid, model)
    solver.build(0.8)
    psi = 0.3 * np.cos(4.0 * np.pi * grid.xi / (2.0 * grid.L)) + 0.05
    picks = np.array([64, 200, 333])
    phi, dphi, d2phi = sample_profile_at_sites(solver, psi, grid.xi[picks])
    assert np.max(np.abs(phi - solver.phi(psi)[picks])) < 1e-10
    assert np.max(np.abs(dphi - solver.phi_prime(psi)[picks])) < 1e-10
    assert np.all(np.isfinite(d2phi))


@pytest.mark.slow
def test_lattice_monodromy_carries_the_structure() -> None:
    """The actual lattice monodromy: conformally symplectic, and ``rho = 1`` on ``phi'``.

    This leaves the advance-delay formulation entirely -- it integrates the lattice in
    ``(n, t)``, the method of the paper this work builds on -- so agreement is an independent
    check that the two descriptions are of the same object.
    """
    grid = Grid(L=100.0, N=1024)
    solver, psi, drive = init_branch(grid, LatticeModel())
    mono = lattice_monodromy(solver, psi, drive, 0.88, steps=1500)

    # the sampled profile really is a lattice traveling wave
    assert mono.drift_position < 1e-4, mono.drift_position
    # M^T J M = e^{-gamma T} J, tested on random pairs without forming M
    assert mono.symplectic_residual(pairs=3) < 1e-10
    # rho = 1 with the predicted eigenvector z = (phi', -c phi'')
    trans = mono.translation_mode_residual()
    assert abs(trans["rayleigh_minus_one"]) < 1e-4
    assert trans["residual_rel"] < 1e-3
