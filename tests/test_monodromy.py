"""The lattice monodromy cross-check (independent of the manuscript's own method)."""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import Grid, LatticeModel, init_branch, lattice_monodromy, natural_continuation
from dlkin.monodromy import _charge_shift, _interpolate_periodic
from dlkin.spectral import biorthogonal_scalars, pencil_eigs

MARCH = tuple(np.round(np.arange(0.881, 0.8901, 0.001), 6))


def test_interpolation_is_exact_at_collocation_points() -> None:
    grid = Grid(L=20.0, N=256)
    values = np.sin(3.0 * np.pi * grid.xi / grid.L) + 0.3 * np.cos(
        np.pi * grid.xi / grid.L
    )
    interpolated = _interpolate_periodic(values, grid, grid.xi, derivative=False)
    assert np.max(np.abs(interpolated - values)) < 1e-12

    from dlkin import spectral_d1

    derivative = _interpolate_periodic(values, grid, grid.xi, derivative=True)
    assert np.max(np.abs(derivative - spectral_d1(values, grid))) < 1e-12


def test_charge_shift_wraps_with_the_topological_jump() -> None:
    u = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    jump = 2.0
    plus = _charge_shift(u, 1, jump)
    assert plus[:-1] == pytest.approx(u[1:])
    assert plus[-1] == pytest.approx(u[0] - jump)

    minus = _charge_shift(u, -1, jump)
    assert minus[1:] == pytest.approx(u[:-1])
    assert minus[0] == pytest.approx(u[-1] + jump)

    # the two shifts undo each other
    assert _charge_shift(_charge_shift(u, 2, jump), -2, jump) == pytest.approx(u)


@pytest.mark.slow
def test_lattice_monodromy_reproduces_the_isolated_multipliers() -> None:
    """The ring's Floquet multipliers agree with the pencil where theory pins them.

    rho = 1 (translation) and rho = e^{-gamma/c} (its conformal partner) are exact; the
    essential spectrum is only a circle, and the ring and the advance-delay grid sample it
    differently, so individual essential multipliers are not expected to line up.  See
    lattice_monodromy's docstring.
    """
    solver, psi, drive = init_branch(Grid(L=200.0, N=2048), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    psi, drive = records[-1].psi, records[-1].drive
    c = 0.89

    scalars = biorthogonal_scalars(solver, psi, drive, c, "minus2pi")
    nus = pencil_eigs(scalars["M0"], scalars["M1"], shift=-0.045, k=24)
    result = lattice_monodromy(solver, psi, drive, c, n_sites=200, pencil_nus=nus)

    assert result["conformal_relerr"] < 1e-9
    assert result["rho_translation_err"] < 1e-5
    assert result["rho_conformal_err"] < 1e-5
    assert result["circle_radius_median_relerr"] < 1e-10
    assert result["circle_radius_expected"] == pytest.approx(np.exp(-0.05 / c))
    assert result["drift"] < 1e-3


@pytest.mark.slow
def test_lattice_monodromy_drift_falls_with_the_ring_size() -> None:
    """The truncation error is real and is reported, not tuned away."""
    solver, psi, drive = init_branch(Grid(L=200.0, N=2048), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    psi, drive = records[-1].psi, records[-1].drive
    scalars = biorthogonal_scalars(solver, psi, drive, 0.89, "minus2pi")
    nus = pencil_eigs(scalars["M0"], scalars["M1"], shift=-0.045, k=12)

    drifts = [
        lattice_monodromy(solver, psi, drive, 0.89, n_sites=n, pencil_nus=nus)["drift"]
        for n in (40, 100, 200)
    ]
    assert drifts[0] > drifts[1] > drifts[2]
    assert drifts[-1] < 1e-4


def test_lattice_monodromy_rejects_a_ring_wider_than_the_window() -> None:
    solver, psi, drive = init_branch(Grid(L=50.0, N=512), LatticeModel(), c0=0.6)
    with pytest.raises(ValueError, match="leaves the computational window"):
        lattice_monodromy(solver, psi, drive, 0.6, n_sites=200)
