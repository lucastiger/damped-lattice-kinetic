"""Fourier-spectral collocation grid and the real circulant operators built from symbols.

The advance-delay equation is collocated on ``xi in [-L, L)`` with ``N`` equispaced
points, ``h = 2L/N``, ``xi_j = -L + j h``.  Every linear operator that appears in the
traveling-wave problem is translation invariant, hence diagonal in the discrete Fourier
basis, hence an ``N x N`` *circulant* matrix.  We build each one from its Fourier symbol
via ``real(ifft(symbol))`` and :func:`scipy.linalg.circulant`; the matrix form is what the
dense Newton and bordered solves downstream need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import circulant

__all__ = [
    "Grid",
    "circulant_from_symbol",
    "sym_d1",
    "sym_d2",
    "sym_shift_laplacian",
    "spectral_d1",
]

#: Relative bound on the imaginary part discarded by :func:`circulant_from_symbol`.
IMAG_TOL = 1e-10


@dataclass(frozen=True)
class Grid:
    """Periodic collocation grid on ``[-L, L)``.

    Parameters
    ----------
    L:
        Half-length of the periodic window.
    N:
        Number of collocation points.  Must be even so that ``xi = 0`` -- where the kink
        is pinned by ``phi(0) = pi`` -- is a grid point.

    Attributes
    ----------
    h:
        Grid spacing ``2L/N``.
    xi:
        Collocation points ``-L + h * arange(N)`` (read-only).
    k:
        Wavenumbers ``2 pi fftfreq(N, d=h)`` in FFT order (read-only).
    j0:
        Index of ``xi = 0``, i.e. ``N // 2``.
    """

    L: float
    N: int
    h: float = field(init=False, compare=False)
    xi: np.ndarray = field(init=False, repr=False, compare=False)
    k: np.ndarray = field(init=False, repr=False, compare=False)
    j0: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        L = float(self.L)
        N = int(self.N)
        if N <= 0:
            raise ValueError(f"Grid requires N > 0, got N={N}.")
        if N % 2 != 0:
            raise ValueError(
                f"Grid requires N even so that xi = 0 is a collocation point, got N={N}."
            )
        if not L > 0.0:
            raise ValueError(f"Grid requires L > 0, got L={L}.")

        h = 2.0 * L / N
        xi = -L + h * np.arange(N)
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=h)
        j0 = N // 2
        if not abs(xi[j0]) < 1e-12:
            raise AssertionError(
                f"xi[j0] should be the origin, got xi[{j0}]={xi[j0]!r} for L={L}, N={N}."
            )
        xi.setflags(write=False)
        k.setflags(write=False)

        object.__setattr__(self, "L", L)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "j0", j0)

    def pair(self, f: np.ndarray, g: np.ndarray) -> float:
        """Discrete bilinear pairing ``<f, g>_h = h * sum_j f_j g_j`` (no conjugation)."""
        return float(np.sum(f * g) * self.h)


def circulant_from_symbol(symbol: np.ndarray, N: int) -> np.ndarray:
    """Build the real ``(N, N)`` circulant matrix whose Fourier symbol is ``symbol``.

    The first column of the circulant is ``ifft(symbol)``.  It is real exactly when the
    symbol is conjugate-symmetric, ``s(-k) = conj(s(k))``; every operator used here is
    arranged to satisfy that (see :func:`sym_d1` for the one place it needs care).  The
    imaginary part that survives is pure round-off and is discarded, after checking it
    against ``IMAG_TOL * max(1, max|real part|)``.  The ``max(1, .)`` guard is the
    reference implementation's: it keeps the test meaningful for symbols whose inverse
    transform is itself tiny, where a pure relative test would trip on round-off.
    """
    symbol = np.asarray(symbol)
    if symbol.shape != (N,):
        raise ValueError(f"symbol must have shape ({N},), got {symbol.shape}.")
    col = np.fft.ifft(symbol)
    max_real = float(np.max(np.abs(col.real)))
    max_imag = float(np.max(np.abs(col.imag)))
    limit = IMAG_TOL * max(1.0, max_real)
    if not max_imag < limit:
        raise ValueError(
            "symbol is not conjugate-symmetric: ifft(symbol) has imaginary part "
            f"{max_imag:.3e}, which is not below {limit:.3e} "
            f"(= {IMAG_TOL:g} * max(1, {max_real:.3e})). "
            "A real circulant requires s(-k) = conj(s(k)); check in particular that the "
            "Nyquist entry of any odd symbol has been zeroed (see sym_d1)."
        )
    return circulant(col.real)


def sym_d1(k: np.ndarray, N: int) -> np.ndarray:
    """Symbol of ``d/dxi``: ``1j * k`` with the Nyquist entry ``k[N//2]`` zeroed.

    The zeroing is REQUIRED, not cosmetic.  A circulant matrix is real iff its symbol
    obeys ``s(-k) = conj(s(k))``.  On an even grid the Nyquist wavenumber
    ``k_{N/2} = -pi/h`` is its own alias (``-k_{N/2}`` and ``k_{N/2}`` are the same
    mode), so conjugate symmetry forces ``s(k_{N/2})`` to be *real*.  The derivative
    symbol ``i k`` is purely imaginary there, so the only admissible value is ``0``.
    Equivalently: the Nyquist grid function ``(-1)^j`` is even about every grid point and
    its derivative is not representable on the grid, so the spectral derivative assigns
    it zero.  Leaving the entry in place makes ``ifft(symbol)`` complex and
    :func:`circulant_from_symbol` rejects it.
    """
    s = 1j * np.asarray(k, dtype=float).copy()
    s[N // 2] = 0.0
    return s


def sym_d2(k: np.ndarray) -> np.ndarray:
    """Symbol of ``d^2/dxi^2``: ``-k**2``.

    No Nyquist surgery is needed -- ``-k**2`` is real, hence already conjugate-symmetric.
    """
    return -np.asarray(k, dtype=float) ** 2


def sym_shift_laplacian(k: np.ndarray, j: int) -> np.ndarray:
    """Symbol of the shift Laplacian ``Delta_j p = p(.+j) + p(.-j) - 2 p``.

    Shifting by ``+/- j`` multiplies the mode ``e^{i k xi}`` by ``e^{+/- i j k}``, so the
    symbol is ``2 cos(j k) - 2``.  Real, hence a real symmetric circulant.
    """
    return 2.0 * np.cos(j * np.asarray(k, dtype=float)) - 2.0


def spectral_d1(f: np.ndarray, grid: Grid) -> np.ndarray:
    """Spectral first derivative of the periodic grid function ``f``."""
    return np.real(np.fft.ifft(sym_d1(grid.k, grid.N) * np.fft.fft(f)))
