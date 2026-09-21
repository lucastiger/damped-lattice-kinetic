"""Natural and pseudo-arclength continuation in the velocity."""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import (
    ContinuationError,
    Grid,
    LatticeModel,
    PseudoArclength,
    circulant_from_symbol,
    init_branch,
    natural_continuation,
    sym_d1,
    sym_d2,
)


@pytest.fixture(scope="module")
def branch():
    """A converged point on the branch at c = 0.88, L = 200, N = 1024."""
    grid = Grid(L=200.0, N=1024)
    model = LatticeModel()
    solver, psi, drive = init_branch(grid, model, c0=0.88)
    return solver, psi, drive


@pytest.fixture(scope="module")
def fine_branch():
    """A converged point at h = 2L/N ~ 0.156, fine enough for the dR/dc check below."""
    grid = Grid(L=200.0, N=2560)
    solver, psi, drive = init_branch(grid, LatticeModel(), c0=0.88)
    return solver, psi, drive


def test_dR_dc_matches_centred_finite_difference(fine_branch) -> None:
    """The dR/dc column of F_and_J is -M1 phi', sign included.

    R contains c^2 phi'' - gamma c phi', so dR/dc = 2 c phi'' - gamma phi' = -M1 phi'
    with M1 = -2 c d_xi + gamma.  A centred difference in c pins down the sign.

    R is exactly quadratic in c, so the centred difference carries no truncation error at
    all; what is left is pure cancellation round-off, ~eps * |Lin psi| / dc, which is
    around 4e-7 in max-norm for step 1e-6 at this resolution.  The grid is chosen fine
    enough that the Nyquist caveat below is negligible, yet coarse enough that the
    round-off floor stays under the 1e-6 tolerance.
    """
    solver, psi, drive = fine_branch
    N = solver.grid.N
    c = 0.88
    arc = PseudoArclength(solver)
    analytic = arc.F_and_J(arc.pack(psi, drive, c)).J[:N, N + 1]

    dc = 1e-6
    fd = (
        solver.residual(psi, drive, c + dc) - solver.residual(psi, drive, c - dc)
    ) / (2.0 * dc)
    assert np.max(np.abs(analytic - fd)) < 1e-6

    # and it really is -M1 phi', not +M1 phi'
    solver.build(c)
    assert np.max(np.abs(analytic + solver.M1 @ solver.phi_prime(psi))) < 1e-12
    assert np.max(np.abs(-analytic - fd)) > 1e-3


def test_dR_dc_deviation_is_exactly_the_nyquist_mode(branch) -> None:
    """Where -M1 phi' and the true discrete dR/dc part company, and why.

    Exactly: dR/dc = 2c (A'' + D2 psi) - gamma (A' + D1 psi), whereas
    -M1 phi' = 2c (D1 A' + D1 D1 psi) - gamma (A' + D1 psi).  The two differ only through
    D1 D1 vs D2 -- identical symbols except at the Nyquist wavenumber, which sym_d1 zeroes
    -- and through D1 A' vs A'', which agree to round-off because A' decays to nothing at
    the domain edges.  So the deviation is a pure Nyquist mode whose amplitude is set by
    the Nyquist content of psi, and it vanishes as the grid is refined.
    """
    solver, psi, drive = branch
    g = solver.grid
    N, c = g.N, 0.88
    D1 = circulant_from_symbol(sym_d1(g.k, N), N)
    D2 = circulant_from_symbol(sym_d2(g.k), N)
    solver.build(c)

    analytic = -(solver.M1 @ solver.phi_prime(psi))
    exact = 2.0 * c * (solver.template.A_second + D2 @ psi) - solver.model.gamma * (
        solver.A_prime + D1 @ psi
    )
    deviation = analytic - exact

    # it is (up to round-off) a pure Nyquist mode
    spectrum = np.fft.fft(deviation)
    nyquist_share = abs(spectrum[N // 2]) / np.linalg.norm(spectrum)
    assert nyquist_share > 0.999
    # of the amplitude predicted by the Nyquist content of psi
    predicted = 2.0 * c * g.k[N // 2] ** 2 * abs(np.fft.fft(psi)[N // 2]) / N
    assert np.max(np.abs(deviation)) == pytest.approx(predicted, rel=1e-6)
    # and A'' is reproduced by differentiating A' spectrally, to round-off
    assert np.max(np.abs(D1 @ solver.A_prime - solver.template.A_second)) < 1e-9


def test_pseudo_arclength_stays_on_the_branch(branch) -> None:
    """Five arclength steps stay converged and agree with natural continuation."""
    solver, psi, drive = branch
    N = solver.grid.N
    arc = PseudoArclength(solver)

    X = arc.pack(psi, drive, 0.88)
    tau = arc.initial_tangent(X, direction=+1.0)
    assert tau[N + 1] > 0.0
    assert np.linalg.norm(tau) == pytest.approx(1.0)

    for _ in range(5):
        X, J, M0, M1, phi_prime, residual = arc.step(X, tau, ds=0.5)
        assert residual < 1e-9
        assert M0.shape == (N, N)
        assert M1.shape == (N, N)
        assert phi_prime.shape == (N,)
        tau = arc.tangent(J, tau)

    c_end = float(X[N + 1])
    assert 0.88 < c_end < 0.9  # still on the rising part of the first branch

    c_values = np.arange(0.881, c_end, 0.001).tolist() + [c_end]
    records = natural_continuation(solver, psi, drive, c_values)
    assert records[-1].c == pytest.approx(c_end, abs=0.0)
    assert abs(records[-1].drive - X[N]) < 1e-8
    assert np.max(np.abs(records[-1].psi - X[:N])) < 1e-8


def test_natural_continuation_records_and_callback(branch) -> None:
    solver, psi, drive = branch
    seen: list[float] = []
    c_values = [0.881, 0.882, 0.883]
    records = natural_continuation(
        solver, psi, drive, c_values, on_step=lambda rec: seen.append(rec.c)
    )
    assert seen == c_values
    assert [r.c for r in records] == c_values
    assert all(r.residual < 1e-11 for r in records)
    # sigma = drive / mu, and sigma rises along the first branch
    assert all(r.sigma == r.drive / solver.model.mu for r in records)
    assert records[0].sigma < records[-1].sigma


def test_natural_continuation_reports_failure_clearly(branch) -> None:
    """A velocity the branch cannot reach aborts with a message naming it."""
    solver, psi, drive = branch
    # c = 1.05 is supersonic: there is no kink of this family to continue to
    with pytest.raises(ContinuationError, match=r"failed at c=1\.05") as excinfo:
        natural_continuation(solver, psi, drive, [0.881, 1.05])
    assert "after 1 accepted step" in str(excinfo.value)
    assert "last converged c=0.881" in str(excinfo.value)


def test_arclength_is_deterministic(branch) -> None:
    """No RNG anywhere: identical inputs give bitwise identical steps."""
    solver, psi, drive = branch
    arc = PseudoArclength(solver)
    X = arc.pack(psi, drive, 0.88)
    tau = arc.initial_tangent(X)
    first = arc.step(X, tau, ds=0.3)[0]
    second = arc.step(X, tau, ds=0.3)[0]
    assert np.array_equal(first, second)


def test_pack_unpack_roundtrip(branch) -> None:
    solver, psi, drive = branch
    arc = PseudoArclength(solver)
    X = arc.pack(psi, drive, 0.88)
    assert X.shape == (solver.grid.N + 2,)
    psi_back, drive_back, c_back = arc.unpack(X)
    assert np.array_equal(psi_back, psi)
    assert drive_back == drive
    assert c_back == 0.88
