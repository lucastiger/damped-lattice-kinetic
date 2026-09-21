"""Grid, symbols and circulant operators."""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import (
    Grid,
    LatticeModel,
    circulant_from_symbol,
    spectral_d1,
    sym_d1,
    sym_d2,
    sym_shift_laplacian,
)


def test_grid_invariants(small_grid: Grid) -> None:
    g = small_grid
    assert g.N % 2 == 0
    assert g.h == pytest.approx(2.0 * g.L / g.N)
    assert abs(g.xi[g.j0]) < 1e-12
    assert g.j0 == g.N // 2
    np.testing.assert_allclose(g.xi, -g.L + g.h * np.arange(g.N))
    np.testing.assert_allclose(g.k, 2.0 * np.pi * np.fft.fftfreq(g.N, d=g.h))


def test_grid_rejects_odd_N() -> None:
    with pytest.raises(ValueError, match="N even"):
        Grid(L=10.0, N=255)


def _symbols(grid: Grid) -> dict[str, np.ndarray]:
    k, N = grid.k, grid.N
    model = LatticeModel(couplings=((1, 1.0), (2, 0.25)))
    return {
        "d1": sym_d1(k, N),
        "d2": sym_d2(k),
        "Delta_1": sym_shift_laplacian(k, 1),
        "Delta_2": sym_shift_laplacian(k, 2),
        "Lin": model.lin_symbol(k, N, 0.88),
        "M1": model.M1_symbol(k, N, 0.88),
    }


def test_circulant_matches_fft_application(small_grid: Grid, rng) -> None:
    """The assembled matrix must agree with the FFT application it was built from."""
    g = small_grid
    vectors = rng.standard_normal((5, g.N))
    for name, s in _symbols(g).items():
        matrix = circulant_from_symbol(s, g.N)
        for v in vectors:
            by_matrix = matrix @ v
            by_fft = np.real(np.fft.ifft(s * np.fft.fft(v)))
            assert np.max(np.abs(by_matrix - by_fft)) < 1e-12, name


def test_circulant_symmetries(small_grid: Grid) -> None:
    """d/dxi is antisymmetric; d^2/dxi^2 and every Delta_j are symmetric."""
    g = small_grid
    D1 = circulant_from_symbol(sym_d1(g.k, g.N), g.N)
    assert np.max(np.abs(D1 + D1.T)) < 1e-12

    D2 = circulant_from_symbol(sym_d2(g.k), g.N)
    assert np.max(np.abs(D2 - D2.T)) < 1e-12

    for j in (1, 2, 3):
        Dj = circulant_from_symbol(sym_shift_laplacian(g.k, j), g.N)
        assert np.max(np.abs(Dj - Dj.T)) < 1e-12, j


def test_circulant_rejects_non_real_symbol(small_grid: Grid) -> None:
    """Without the Nyquist zeroing the derivative symbol has no real circulant."""
    g = small_grid
    bad = 1j * np.asarray(g.k)  # sym_d1 without the Nyquist entry removed
    assert bad[g.N // 2] != 0.0
    with pytest.raises(ValueError, match="conjugate-symmetric"):
        circulant_from_symbol(bad, g.N)


def test_spectral_d1_is_exact_on_grid_modes(small_grid: Grid) -> None:
    """d/dxi of sin(2 pi n xi / (2 L)) is exact -- the mode lives in the DFT basis."""
    g = small_grid
    for n in (1, 3, 17):
        omega = 2.0 * np.pi * n / (2.0 * g.L)
        f = np.sin(omega * g.xi)
        expected = omega * np.cos(omega * g.xi)
        assert np.max(np.abs(spectral_d1(f, g) - expected)) < 1e-10, n


def test_spectral_d1_kills_the_nyquist_mode(small_grid: Grid) -> None:
    """The Nyquist grid function (-1)^j has no representable derivative; it maps to 0."""
    g = small_grid
    nyquist = np.cos(np.pi * np.arange(g.N))
    assert np.max(np.abs(spectral_d1(nyquist, g))) < 1e-12


def test_lin_symbol_assembles_from_its_parts(small_grid: Grid) -> None:
    """Lin(c) = c^2 d^2 - gamma c d - sum_j C_j Delta_j, built two ways."""
    g = small_grid
    c = 0.88
    model = LatticeModel(gamma=0.1, couplings=((1, 1.0), (2, 0.25)))
    direct = circulant_from_symbol(model.lin_symbol(g.k, g.N, c), g.N)
    pieces = (
        c**2 * circulant_from_symbol(sym_d2(g.k), g.N)
        - model.gamma * c * circulant_from_symbol(sym_d1(g.k, g.N), g.N)
        - 1.00 * circulant_from_symbol(sym_shift_laplacian(g.k, 1), g.N)
        - 0.25 * circulant_from_symbol(sym_shift_laplacian(g.k, 2), g.N)
    )
    assert np.max(np.abs(direct - pieces)) < 1e-12
