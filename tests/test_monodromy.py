"""Conformal symplecticity, by construction and by direct integration.

``handoff/reference/drivers_session.py:conformal()`` is the reference for the random-K
test; ``dlkin.monodromy`` must reproduce it, and must also get the real thing right -- the
monodromy of the linearized Frenkel-Kontorova lattice about a computed kink, where nothing
is arranged in advance.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from dlkin import (
    Grid,
    LatticeModel,
    branch_scalars,
    conformal_symplectic_flow_test,
    conformal_symplectic_residual,
    init_branch,
    interpolate_psi,
    lattice_monodromy,
    natural_continuation,
    shift_matrix,
    spectral_d1,
    symplectic_form,
)

L, N, C0, C = 200.0, 1024, 0.88, 0.89
APPROACH = [round(0.881 + 0.001 * i, 6) for i in range(10)]


@pytest.fixture(scope="module")
def branch():
    grid = Grid(L=L, N=N)
    solver, psi, drive = init_branch(grid, LatticeModel(), c0=C0)
    records = natural_continuation(solver, psi, drive, APPROACH)
    return solver, records[-1].psi, records[-1].drive


def _reference_conformal(n: int, gam: float, T: float, dt: float, seed: int) -> float:
    """``conformal()`` of handoff/reference/drivers_session.py, inlined."""
    rng = np.random.default_rng(seed)
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    Phi = np.eye(2 * n)
    t = 0.0
    while t < T - 1e-12:
        K = rng.standard_normal((n, n))
        K = (K + K.T) / 2
        A = np.block([[np.zeros((n, n)), np.eye(n)], [K, -gam * np.eye(n)]])
        Phi = expm(A * dt) @ Phi
        t += dt
    P = np.zeros((n, n))
    P[np.arange(n), (np.arange(n) + 1) % n] = 1.0
    M = np.block([[P, np.zeros((n, n))], [np.zeros((n, n)), P]]) @ Phi
    return float(np.max(np.abs(M.T @ J @ M - np.exp(-gam * T) * J)) / np.max(np.abs(J)))


def test_random_K_matches_the_reference() -> None:
    """Same seed, same steps, same answer as drivers_session.conformal()."""
    ours = conformal_symplectic_flow_test(n=6, gamma=0.37, T=0.05, dt=1e-3, seed=1)
    theirs = _reference_conformal(6, 0.37, 0.05, 1e-3, 1)
    assert ours["n_steps"] == 50
    assert ours["relerr"] == pytest.approx(theirs, rel=1e-12, abs=1e-18)


def test_the_relation_needs_only_symmetry_of_K() -> None:
    """K is redrawn every step: no lattice, no traveling wave, not even continuity in t."""
    assert conformal_symplectic_flow_test(n=8, gamma=0.37, T=0.1, dt=1e-3)["relerr"] < 1e-13


def test_rk4_imposes_nothing_and_converges_in_dt() -> None:
    """expm satisfies the relation per step; RK4 does not, so its residual must fall."""
    coarse = conformal_symplectic_flow_test(n=4, gamma=0.37, T=0.1, dt=2e-2, integrator="rk4")
    fine = conformal_symplectic_flow_test(n=4, gamma=0.37, T=0.1, dt=5e-3, integrator="rk4")
    assert coarse["relerr"] > fine["relerr"]
    # fourth order: sixteen-fold refinement in dt^4 per halving, twice over
    assert fine["relerr"] < coarse["relerr"] / 100.0


def test_flow_test_rejects_an_unknown_integrator() -> None:
    with pytest.raises(ValueError, match="integrator must be"):
        conformal_symplectic_flow_test(integrator="midpoint")


def test_shift_matrix_is_orthogonal_and_symplectic() -> None:
    """diag(P, P) must leave the conformal relation alone, or the monodromy is not one."""
    n = 7
    P = shift_matrix(n)
    np.testing.assert_allclose(P @ P.T, np.eye(n), atol=1e-14)
    S = np.block([[P, np.zeros((n, n))], [np.zeros((n, n)), P]])
    J = symplectic_form(n)
    np.testing.assert_allclose(S.T @ J @ S, J, atol=1e-14)


def test_conformal_residual_is_zero_for_an_exact_flow() -> None:
    """A single expm step of a constant A is exactly conformally symplectic."""
    n, gamma, T = 5, 0.31, 0.4
    rng = np.random.default_rng(3)
    K = rng.standard_normal((n, n))
    K = 0.5 * (K + K.T)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [K, -gamma * np.eye(n)]])
    assert conformal_symplectic_residual(expm(A * T), gamma, T) < 1e-13


def test_interpolate_psi_reproduces_the_grid_and_its_derivative(branch) -> None:
    """The trigonometric interpolant is the object the spectral operators act on."""
    solver, psi, _ = branch
    grid = solver.grid
    np.testing.assert_allclose(interpolate_psi(psi, grid, grid.xi), psi, atol=1e-12)
    np.testing.assert_allclose(
        interpolate_psi(psi, grid, grid.xi, 1), spectral_d1(psi, grid), atol=1e-12
    )


#: How close the structural multipliers come at THIS fixture's resolution.  The fixture is
#: N = 1024 (h ~ 0.39) on a ring of 120 sites, chosen so the suite stays fast, and there the
#: interpolated profile solves (TW) at the integer sites only to ~2e-2 -- the kink is 0.46
#: wide and h is 0.39.  Production (N = 2048, 200 sites) reaches 8e-10; see
#: data/07_conformal.json.  The conformal residual is unaffected, because it is a property
#: of the integrator and of the symmetry of K, not of how good the profile is.
RHO_TOL = 1e-4


@pytest.mark.slow
def test_fk_monodromy_is_conformally_symplectic(branch) -> None:
    """The real thing, with RK4, which imposes no structure -- so this is a measurement.

    Also the two multipliers that must be there: ``rho = 1`` for the translation mode, and
    its conformal partner at ``e^{-gamma/c}``.
    """
    solver, psi, drive = branch
    lm = lattice_monodromy(solver, psi, drive, C, n_sites=120, n_steps=300)
    assert lm.cs_relerr < 1e-10
    rho = lm.multipliers
    assert float(np.min(np.abs(rho - 1.0))) < RHO_TOL
    assert float(np.min(np.abs(rho - np.exp(-solver.model.gamma / C)))) < RHO_TOL
    # c = 0.89 is below the first extremum: nothing outside the unit circle
    assert float(np.max(np.abs(rho))) < 1.0 + RHO_TOL


@pytest.mark.slow
def test_fk_monodromy_residual_falls_like_the_time_step(branch) -> None:
    """RK4 is fourth order, so halving dt should cut the residual by about sixteen."""
    solver, psi, drive = branch
    coarse = lattice_monodromy(solver, psi, drive, C, n_sites=60, n_steps=100)
    fine = lattice_monodromy(solver, psi, drive, C, n_sites=60, n_steps=200)
    assert fine.cs_relerr < coarse.cs_relerr / 8.0


@pytest.mark.slow
def test_monodromy_multipliers_match_the_pencil(branch) -> None:
    """exp(nu/c) from the pencil against the directly integrated monodromy.

    The pencil presupposes Floquet solutions of the form ``e^{nu t} p(n - c t)``; the
    monodromy presupposes nothing.  They agree on the localized modes, which is the point.
    """
    solver, psi, drive = branch
    scalars = branch_scalars(solver, psi, drive, C)
    lm = lattice_monodromy(solver, psi, drive, C, n_sites=120, n_steps=300)
    for nu in (0.0, -solver.model.gamma):
        predicted = float(np.exp(nu / C))
        assert float(np.min(np.abs(lm.multipliers - predicted))) < RHO_TOL
    assert scalars.res_phat_rel < 1e-3
