"""THE key test of the numerical core: dlkin must equal handoff/reference/fk.py.

``handoff/reference/fk.py`` is the code that produced every number quoted in the note.
``dlkin`` restructures it -- dataclasses, type hints, error handling, a single ``drive``
variable shared by the Frenkel-Kontorova and generalized cases -- but must not change the
discretization, the template, the initialization, the line search or any tolerance.  This
test imports the reference module directly and compares the two end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dlkin import Grid, LatticeModel, TravelingWaveSolver, init_branch

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "handoff" / "reference"

L, N, C0 = 200.0, 1024, 0.88


@pytest.fixture(scope="module")
def fk():
    """Import handoff/reference/fk.py as a module."""
    if not REFERENCE_DIR.is_dir():
        pytest.fail(f"reference implementation not found at {REFERENCE_DIR}")
    sys.path.insert(0, str(REFERENCE_DIR))
    try:
        import fk as reference  # noqa: PLC0415

        return reference
    finally:
        sys.path.remove(str(REFERENCE_DIR))


def test_init_branch_matches_reference(fk) -> None:
    """sigma to 1e-10 and the profile in max-norm to 1e-10, at L=200, N=1024, c=0.88."""
    ref_solver, ref_psi, ref_sigma = fk.init_branch(L, N, c0=C0)
    ref_phi = ref_solver.A + ref_psi

    grid = Grid(L=L, N=N)
    model = LatticeModel()
    solver, psi, drive = init_branch(grid, model, c0=C0)
    sigma = solver.sigma_of(drive)
    phi = solver.phi(psi)

    assert abs(sigma - ref_sigma) < 1e-10
    assert np.max(np.abs(phi - ref_phi)) < 1e-10


def test_single_stage_solve_matches_reference(fk) -> None:
    """The first stage alone (physical-width template, flat start) also agrees."""
    width = float(np.sqrt(1.0 - C0**2))
    ref_solver = fk.FKSolver(L=L, N=N, tw=width)
    ref_psi, ref_sigma, ref_res, ref_iters = ref_solver.solve(C0, np.zeros(N), 0.6)

    grid = Grid(L=L, N=N)
    solver = TravelingWaveSolver(grid, LatticeModel(template_width=width))
    result = solver.newton(C0, np.zeros(N), 0.6)

    assert result.iters == ref_iters
    assert abs(result.residual - ref_res) < 1e-16
    assert abs(result.sigma - ref_sigma) < 1e-10
    assert np.max(np.abs(result.psi - ref_psi)) < 1e-10


def test_operators_and_template_match_reference(fk) -> None:
    """Lin, M1 and the template arrays are assembled identically."""
    grid = Grid(L=L, N=N)
    model = LatticeModel()
    solver = TravelingWaveSolver(grid, model)
    solver.build(C0)

    ref_solver = fk.FKSolver(L=L, N=N)
    ref_solver.build(C0)

    np.testing.assert_allclose(solver.Lin, ref_solver.Lin, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(solver.M1, ref_solver.M1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(solver.A, ref_solver.A, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(solver.A_prime, ref_solver.Ap, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        solver.template.A_second, ref_solver.App, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        solver.template.coupling_delta, ref_solver.D1A, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(grid.xi, ref_solver.xi, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grid.k, ref_solver.k, rtol=0.0, atol=1e-12)
    assert grid.h == ref_solver.h
    assert grid.j0 == ref_solver.j0


def test_generalized_path_matches_reference_gen(fk) -> None:
    """The same code path, with mu2 and next-nearest coupling, matches gen.py.

    ``gen.py`` is the reference for the scope test; its unknown is the additive drive
    ``f``, which is exactly dlkin's ``drive``.
    """
    sys.path.insert(0, str(REFERENCE_DIR))
    try:
        import gen  # noqa: PLC0415
    finally:
        sys.path.remove(str(REFERENCE_DIR))

    n = 512
    kwargs = dict(mu=1.0, mu2=-0.5, C1=1.0, C2=0.25)
    ref = gen.Gen(L=L, N=n, gamma=0.1, **kwargs)
    ref_psi, ref_f = np.zeros(n), 0.5
    for c in (0.55, 0.65, 0.75):
        ref_psi, ref_f, ref_res = ref.solve(c, ref_psi, ref_f)

    grid = Grid(L=L, N=n)
    model = LatticeModel(gamma=0.1, mu=1.0, mu2=-0.5, couplings=((1, 1.0), (2, 0.25)))
    solver = TravelingWaveSolver(grid, model)
    psi, drive = np.zeros(n), 0.5
    for c in (0.55, 0.65, 0.75):
        # gen.py's Newton uses tol=1e-11; match it so the two stop at the same iterate
        result = solver.newton(c, psi, drive, tol=1e-11)
        psi, drive = result.psi, result.drive

    assert abs(drive - ref_f) < 1e-10
    assert np.max(np.abs(psi - ref_psi)) < 1e-10
