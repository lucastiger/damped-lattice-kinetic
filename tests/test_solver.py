"""The traveling-wave Newton solver."""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import (
    ConvergenceError,
    Grid,
    LatticeModel,
    TravelingWaveSolver,
    init_branch,
)


def _newton_residual(solver: TravelingWaveSolver, psi, drive, c) -> float:
    R = solver.residual(psi, drive, c)
    return max(float(np.max(np.abs(R))), abs(float(psi[solver.grid.j0])))


def test_newton_converges_on_the_branch_at_c088() -> None:
    """L = 200, N = 1024, c = 0.88: the two-stage initialization lands converged."""
    grid = Grid(L=200.0, N=1024)
    solver, psi, drive = init_branch(grid, LatticeModel(), c0=0.88)
    residual = _newton_residual(solver, psi, drive, 0.88)
    assert residual < 1e-11
    # the pinning condition phi(0) = pi, and the far-field boundary conditions
    sigma = solver.sigma_of(drive)
    phi = solver.phi(psi)
    assert phi[grid.j0] == pytest.approx(np.pi, abs=1e-11)
    assert phi[0] == pytest.approx(np.arcsin(sigma) + 2.0 * np.pi, abs=1e-7)
    assert phi[-1] == pytest.approx(np.arcsin(sigma), abs=1e-7)


def test_jacobian_matches_finite_differences(small_grid: Grid, fk_model, rng) -> None:
    """M0 = Lin + diag(V''(A + psi)) is the exact psi-Jacobian of the residual."""
    solver = TravelingWaveSolver(small_grid, fk_model)
    c, drive = 0.88, 0.6
    solver.build(c)
    psi = 0.05 * rng.standard_normal(small_grid.N)
    M0 = solver.M0(psi)

    eps = 1e-6
    columns = rng.choice(small_grid.N, size=8, replace=False)
    for j in columns:
        bump = np.zeros(small_grid.N)
        bump[j] = eps
        fd = (
            solver.residual(psi + bump, drive, c) - solver.residual(psi - bump, drive, c)
        ) / (2.0 * eps)
        assert np.max(np.abs(M0[:, j] - fd)) < 1e-6, j


def test_drive_column_of_the_jacobian_is_minus_one(small_grid: Grid, fk_model, rng) -> None:
    """dR/d(drive) = -1 exactly: the drive enters additively and absorbs mu."""
    solver = TravelingWaveSolver(small_grid, fk_model)
    c = 0.88
    solver.build(c)
    psi = 0.05 * rng.standard_normal(small_grid.N)
    eps = 1e-6
    fd = (solver.residual(psi, 0.6 + eps, c) - solver.residual(psi, 0.6 - eps, c)) / (2.0 * eps)
    assert np.max(np.abs(fd + 1.0)) < 1e-9


def test_sigma_is_drive_over_mu(small_grid: Grid) -> None:
    """The FK and generalized cases share one path; sigma is the convenience view."""
    model = LatticeModel(mu=0.6)
    solver = TravelingWaveSolver(small_grid, model)
    solver.build(0.5)
    result = solver.newton(0.5, np.zeros(small_grid.N), 0.6 * model.mu, maxit=40)
    assert result.sigma == pytest.approx(result.drive / model.mu, rel=0, abs=0)
    assert solver.sigma_of(result.drive) == result.sigma


def test_convergence_error_on_absurd_initial_guess(small_grid: Grid, fk_model) -> None:
    """A hopeless start must raise, not return a silently unconverged iterate."""
    solver = TravelingWaveSolver(small_grid, fk_model)
    absurd = 1.0e4 * np.ones(small_grid.N)
    with pytest.raises(ConvergenceError) as excinfo:
        solver.newton(0.88, absurd, 0.6)
    assert excinfo.value.residual is not None
    assert excinfo.value.residual > 1e-9
    assert "tol_fail" in str(excinfo.value)


def test_init_branch_needs_both_stages() -> None:
    """The fixed-width template alone cannot be started from psi = 0.

    Stage 1 exists because a flat start only converges with the *physical*-width
    template w0 = sqrt((1 - c0^2)/mu); the w = 2 template diverges from psi = 0.  Stage 2
    then re-converges on the fixed-width template, which lands on a slightly different
    discrete solution (the A/psi split changes what the periodic remainder has to carry),
    and it is that branch the reported numbers live on.
    """
    grid = Grid(L=100.0, N=512)
    model = LatticeModel()
    c0 = 0.88

    one_stage = TravelingWaveSolver(grid, model)
    with pytest.raises(ConvergenceError):
        one_stage.newton(c0, np.zeros(grid.N), 0.6 * model.mu)

    solver, psi, drive = init_branch(grid, model, c0=c0)
    assert solver.model.template_width == model.template_width
    assert _newton_residual(solver, psi, drive, c0) < 1e-11

    # stage 2 genuinely moves the solution: it is not a no-op re-solve
    width0 = float(np.sqrt((1.0 - c0**2) / model.mu))
    stage1 = TravelingWaveSolver(grid, LatticeModel(template_width=width0))
    first = stage1.newton(c0, np.zeros(grid.N), 0.6 * model.mu)
    assert abs(drive - first.drive) > 1e-6


def test_init_branch_drive_guess_scales_with_mu(small_grid: Grid) -> None:
    """The default guess is sigma = 0.6, i.e. drive = 0.6 * mu."""
    model = LatticeModel(mu=1.0)
    solver, psi, drive = init_branch(small_grid, model, c0=0.6)
    explicit, psi_e, drive_e = init_branch(small_grid, model, c0=0.6, drive_guess=0.6)
    assert drive == pytest.approx(drive_e, abs=0.0)
    assert np.max(np.abs(psi - psi_e)) == 0.0
