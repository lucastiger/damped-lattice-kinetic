"""Regression against the ground truth recomputed from scratch after the original investigation.

``handoff/reference/verify_core.json`` and ``verify_extra.json`` hold values produced by the
reference implementation in a clean environment, and they agree with the manuscript to every
quoted digit.  These tests pin the *package* to them at production resolution, so a refactor
that changes the physics fails here rather than silently moving a number in the paper.  They
are slow (dense N = 4096 linear algebra) and run under ``pytest -m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlkin import Grid, LatticeModel, TravelingWaveSolver, init_branch, natural_continuation
from dlkin.spectral import branch_scalars, pencil_eigs, real_nontrivial

REFERENCE = Path(__file__).resolve().parents[1] / "handoff" / "reference"


@pytest.fixture(scope="module")
def core():
    return json.loads((REFERENCE / "verify_core.json").read_text())


@pytest.fixture(scope="module")
def branch_4096():
    grid = Grid(L=200.0, N=4096)
    solver, psi, drive = init_branch(grid, LatticeModel())
    path = [round(0.881 + 0.001 * i, 6) for i in range(10)]
    last = natural_continuation(solver, psi, drive, path)[-1]
    return solver, last.psi, last.drive


@pytest.mark.slow
def test_c089_scalars_match_ground_truth(core, branch_4096) -> None:
    solver, psi, drive = branch_4096
    sc = branch_scalars(solver, psi, drive, 0.89, normalization="minus2pi")
    ref = core["b"]
    assert sc.sigma == pytest.approx(ref["sigma"], rel=1e-12)
    assert sc.sigma_prime == pytest.approx(ref["sigp"], rel=1e-10)
    assert sc.kappa == pytest.approx(ref["kappa"], rel=1e-10)
    assert sc.m == pytest.approx(ref["m"], rel=1e-8)
    assert sc.nu2_pred == pytest.approx(ref["nu2_pred"], rel=1e-8)
    # the identity itself, and the two free checks
    assert sc.identity_relerr < 1e-12
    assert sc.power_balance_relerr < 1e-12
    assert abs(sc.phat_dot_Vpp) < 1e-11


@pytest.mark.slow
def test_unstable_eigenvalue_past_the_threshold(core, branch_4096) -> None:
    """At c = 0.899, just past c_hat1, the pencil has a real eigenvalue +0.001931."""
    solver, psi, drive = branch_4096
    last = natural_continuation(solver, psi, drive, [0.8988, 0.89892, 0.8990])[-1]
    sc = branch_scalars(solver, last.psi, last.drive, 0.899, normalization="minus2pi")
    # recorded in handoff/reference/verify_core.log, line "(c) 0.899"
    assert sc.sigma_prime == pytest.approx(-0.09005272389171688, rel=1e-8)
    assert sc.kappa == pytest.approx(-0.5658179516125822, rel=1e-8)
    assert sc.m == pytest.approx(290.4159271788894, rel=1e-6)
    assert sc.nu2_pred == pytest.approx(core["d0.8990"]["nu2_pred"], rel=1e-8)
    nus = pencil_eigs(sc.M0, sc.M1, shift=max(sc.nu2_pred, 0.0) + 0.004, k=24)
    real = real_nontrivial(nus, 0.1)
    assert any(abs(x - core["d0.8990"]["nu2"]) < 1e-7 for x in real), real


@pytest.mark.slow
def test_generalized_lattice_matches_ground_truth() -> None:
    """mu2 = -0.5 at N = 2048: the worst row of Table 7, and its collapse under refinement."""
    extra = json.loads((REFERENCE / "verify_extra.json").read_text())
    ref = extra["mu2m05_N2048"]
    model = LatticeModel(mu=1.0, gamma=0.1, mu2=-0.5)
    solver = TravelingWaveSolver(Grid(L=200.0, N=2048), model)
    psi, drive = np.zeros(2048), 0.5
    for c in (0.55, 0.65, 0.75, 0.82):
        result = solver.newton(c, psi, drive, tol=1e-11)
        psi, drive = result.psi, result.drive
    sc = branch_scalars(solver, psi, drive, 0.82, normalization="unit")
    assert sc.drive == pytest.approx(ref["f"], abs=2e-6)
    assert sc.drive_prime == pytest.approx(ref["fp"], abs=3e-6)
    assert sc.kappa == pytest.approx(ref["kappa"], abs=3e-6)
    assert sc.identity_relerr == pytest.approx(ref["relerr"], rel=0.2)
