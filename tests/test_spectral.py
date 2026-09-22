"""The biorthogonal / spectral layer, against handoff/reference/spectral.py.

``handoff/reference/spectral.py`` is the code that produced the reported ``kappa``, ``m``
and eigenvalues.  ``dlkin.spectral`` restructures it but must not change a single
arithmetic operation: the inverse iteration, the two bordered solves and the shift-invert
Arnoldi all have to reproduce the reference bit for bit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dlkin import (
    Grid,
    LatticeModel,
    classify_eigs,
    branch_scalars,
    init_branch,
    kappa_scalars,
    left_null,
    natural_continuation,
    normalize_phat,
    pencil_eigs,
    real_nontrivial,
)

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "handoff" / "reference"

L, N, C0, C = 200.0, 1024, 0.88, 0.89
APPROACH = [round(0.881 + 0.001 * i, 6) for i in range(10)]


@pytest.fixture(scope="module")
def reference():
    """Import handoff/reference/{fk,spectral}.py as modules."""
    if not REFERENCE_DIR.is_dir():
        pytest.fail(f"reference implementation not found at {REFERENCE_DIR}")
    sys.path.insert(0, str(REFERENCE_DIR))
    try:
        import fk  # noqa: PLC0415
        import spectral  # noqa: PLC0415

        return fk, spectral
    finally:
        sys.path.remove(str(REFERENCE_DIR))


@pytest.fixture(scope="module")
def reference_scalars(reference):
    """``spectral.scalars`` at c = 0.89 on the reference's own branch."""
    fk, spectral = reference
    solver, psi, sigma = fk.init_branch(L, N)
    for c in APPROACH:
        psi, sigma, _, _ = solver.solve(c, psi, sigma)
    return solver, psi, sigma, spectral.scalars(solver, psi, sigma, C)


@pytest.fixture(scope="module")
def branch():
    """The same branch point through dlkin."""
    grid = Grid(L=L, N=N)
    solver, psi, drive = init_branch(grid, LatticeModel(), c0=C0)
    records = natural_continuation(solver, psi, drive, APPROACH)
    return solver, records[-1].psi, records[-1].drive


@pytest.fixture(scope="module")
def scalars(branch):
    solver, psi, drive = branch
    return branch_scalars(solver, psi, drive, C)


def test_left_null_matches_reference(reference_scalars, scalars) -> None:
    """Inverse iteration lands on the same vector, to the last bit."""
    _, _, _, ref = reference_scalars
    np.testing.assert_array_equal(scalars.phat, ref["phat"])


def test_branch_scalars_match_reference(reference_scalars, scalars) -> None:
    """Every scalar of the identity and threshold tables agrees exactly."""
    _, _, _, ref = reference_scalars
    assert scalars.sigma == ref["sigma"]
    assert scalars.sigma_prime == ref["sigp"]
    assert scalars.kappa == ref["kappa"]
    assert scalars.phat_one == ref["one"]
    assert scalars.phat_dot_Vpp == ref["orth"]
    assert scalars.m == ref["m"]
    assert scalars.nu2_pred == ref["nu2_pred"]
    assert scalars.identity_relerr == ref["ident_relerr"]
    assert scalars.sigma_power_balance == ref["sigma_pb"]
    assert scalars.res_M0_phi_prime == ref["res_M0p0"]
    assert scalars.dcphi_edge == ref["dcpsi_bdry"]
    np.testing.assert_array_equal(scalars.phi_prime, ref["p0"])


def test_identity_and_power_balance_hold_at_this_resolution(scalars) -> None:
    """h ~ 0.39 is the coarsest grid in the note; even there the identity is good to 3e-4.

    Neither relation is imposed anywhere: ``kappa`` is a pairing, ``sigma'`` a bordered
    solve, and the power balance a quadrature of ``phi'^2``.
    """
    assert scalars.identity_relerr == pytest.approx(3.05e-4, rel=0.05)
    assert scalars.power_balance_relerr < 2e-3
    assert abs(scalars.phat_dot_Vpp) < 2e-3


def test_dcphi_edge_matches_closed_form(scalars) -> None:
    """d_c phi at the far field is sigma'/sqrt(1-sigma^2), the derivative of arcsin(sigma)."""
    closed = scalars.sigma_prime / np.sqrt(1.0 - scalars.sigma**2)
    assert scalars.dcphi_edge_pred == pytest.approx(closed, rel=1e-14)
    assert scalars.dcphi_edge == pytest.approx(closed, rel=5e-3)


def test_normalizations_differ_only_by_a_common_factor(branch) -> None:
    """minus2pi and unit rescale kappa and <phat,1> together, so ratios are invariant."""
    solver, psi, drive = branch
    a = branch_scalars(solver, psi, drive, C, normalization="minus2pi")
    b = branch_scalars(solver, psi, drive, C, normalization="unit")
    assert a.kappa / a.phat_one == pytest.approx(b.kappa / b.phat_one, rel=1e-12)
    assert np.linalg.norm(b.phat) == pytest.approx(1.0, rel=1e-12)
    assert float(np.sum(b.phat * b.phi_prime)) > 0.0
    assert a.phat_one == pytest.approx(-2.0 * np.pi, rel=1e-12)


def test_kappa_scalars_agrees_with_branch_scalars(branch, scalars) -> None:
    """The cheap operator-level path used inside arclength continuation is the same one."""
    solver, psi, _ = branch
    ks = kappa_scalars(scalars.M0, scalars.M1, scalars.phi_prime, solver.grid.h)
    assert ks.kappa == scalars.kappa
    assert ks.phat_one == scalars.phat_one


def test_normalize_phat_rejects_an_unusable_convention(scalars) -> None:
    with pytest.raises(ValueError, match="unknown phat normalization"):
        normalize_phat(scalars.phat, 1.0, "l1")
    with pytest.raises(ValueError, match="needs phi_prime"):
        normalize_phat(scalars.phat, 1.0, "unit")


def test_left_null_is_deterministic(scalars) -> None:
    """Same seed, same vector -- nothing here reads process state."""
    a = left_null(scalars.M0, seed=0)
    b = left_null(scalars.M0, seed=0)
    np.testing.assert_array_equal(a, b)


def test_real_nontrivial_discards_the_two_guaranteed_eigenvalues() -> None:
    """nu = 0 (translation) and nu = -gamma (its conformal partner) are always there."""
    gamma = 0.1
    nus = [0.0, -gamma, 0.0042, 0.0042 + 1e-12j, -0.3, 0.05 + 1e-3j]
    assert real_nontrivial(nus, gamma) == [0.0042, -0.3]


@pytest.mark.slow
def test_pencil_has_the_two_structural_eigenvalues(scalars) -> None:
    """Q(nu) must have nu = 0 (ker M0 = span phi') and, by R2, nu = -gamma.

    One shift each: a single shift near -gamma/2 does not reach either, because between
    it and them lie the O(N) eigenvalues of the discretized essential spectrum (R3), and
    ARPACK returns only the ``k`` nearest.

    The tolerance is 1e-3, not the 1e-6 of the threshold tables, because this fixture is
    at ``h = 2L/N ~ 0.39`` -- the coarsest grid in the note, chosen here so the suite
    stays fast.  ``nu = 0`` is an eigenvalue only to the accuracy with which the discrete
    ``M0`` is singular, and at this ``h`` that is ~1e-1 in ``|M0 phi'|/|phi'|``; the
    eigenvalue comes out at 1.8e-4.  At ``N = 4096`` it is below 1e-6.
    """
    zero = float(np.min(np.abs(pencil_eigs(scalars.M0, scalars.M1, shift=0.004, k=20))))
    partner = float(
        np.min(np.abs(pencil_eigs(scalars.M0, scalars.M1, shift=-0.096, k=20) + 0.1))
    )
    assert zero < 1e-3
    assert partner < 1e-3
    # R2 is exact, not approximate: the pencil's spectrum is invariant under
    # nu -> -gamma - nu, so the two are resolved to the very same accuracy.
    assert zero == pytest.approx(partner, rel=1e-9)


@pytest.mark.slow
def test_pencil_eigs_matches_reference(reference, scalars) -> None:
    _, spectral = reference
    ours = pencil_eigs(scalars.M0, scalars.M1, shift=-0.045, k=20)
    theirs = spectral.pencil_eigs(scalars.M0, scalars.M1, shift=-0.045, k=20)
    np.testing.assert_array_equal(np.sort_complex(ours), np.sort_complex(theirs))


def test_classify_eigs_separates_the_structural_eigenvalues() -> None:
    """nu = 0 and nu = -gamma are found; pairing measures closure under nu -> -gamma - nu."""
    gamma = 0.1
    nus = np.array([1e-9, -0.1 + 1e-10j, 0.0042, -0.1042, -0.05 + 0.3j, -0.05 - 0.3j])
    info = classify_eigs(nus, gamma)
    assert info.nu_zero_abs == pytest.approx(1e-9)
    assert info.nu_minus_gamma.real == pytest.approx(-0.1)
    assert info.real_nontrivial == [0.0042, -0.1042]
    assert info.n_real_nontrivial == 2
    # this set is closed under the involution to ~1e-9
    assert info.pairing_max_err < 1e-8


def test_classify_eigs_reports_an_unpaired_window() -> None:
    """An eigenvalue whose partner fell outside the computed window shows up as the error."""
    gamma = 0.1
    info = classify_eigs(np.array([0.0, -0.1, 0.02]), gamma)
    # the partner of 0.02 is -0.12, which is not in the set
    assert info.pairing_max_err == pytest.approx(0.02)


def test_classify_eigs_multipliers() -> None:
    """rho = exp(nu/c); nu = 0 is the neutral multiplier and nu = -gamma its partner."""
    info = classify_eigs(np.array([0.0, -0.1]), 0.1)
    rho = info.rho(0.89)
    assert rho[0] == pytest.approx(1.0)
    assert rho[0] * rho[1] == pytest.approx(np.exp(-0.1 / 0.89))


def test_classify_eigs_needs_something_to_classify() -> None:
    with pytest.raises(ValueError, match="at least one eigenvalue"):
        classify_eigs([], 0.1)
