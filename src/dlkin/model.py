"""The damped, dc-driven lattice and the analytic kink template.

Model (Vainchtein, Cuevas-Maraver, Kevrekidis & Xu, CNSNS **85** (2020) 105236, eq. (1)),
in the generalized form that the Frenkel-Kontorova case is a special member of::

    u_n'' + gamma u_n' = sum_j C_j (u_{n+j} - 2 u_n + u_{n-j}) + f - V'(u_n)
    V'(u) = mu sin(u) + mu2 sin(2 u)

With ``couplings = ((1, 1.0),)``, ``mu2 = 0`` and ``f = mu sigma`` this is exactly the
Frenkel-Kontorova lattice of the note.

For the traveling wave ``u_n(t) = phi(xi)``, ``xi = n - c t``::

    c^2 phi'' - gamma c phi' = sum_j C_j Delta_j phi + f - V'(phi)          (TW)

which we write as ``Lin psi + ... = 0`` with

    Lin(c) = c^2 d_xi^2 - gamma c d_xi - sum_j C_j Delta_j
    M1(c)  = -2 c d_xi + gamma                    (= d/d nu of the co-traveling pencil)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from .grid import sym_d1, sym_d2, sym_shift_laplacian

__all__ = ["LatticeModel", "Template", "template_arrays"]

#: Argument clip for ``exp``/``cosh`` in the template, so that the far field of a long
#: domain evaluates to a clean 0 or 2*pi instead of overflowing.
_CLIP = 600.0


@dataclass(frozen=True)
class Template:
    """The analytic kink template and the closed-form combinations the residual needs.

    Attributes
    ----------
    A:
        ``A(xi) = 2 pi - 4 arctan(exp(xi / w))``; runs from ``2 pi`` at ``xi -> -inf`` to
        ``0`` at ``xi -> +inf`` and carries the whole topological jump.
    A_prime, A_second:
        ``A'`` and ``A''``, in closed form.
    coupling_delta:
        ``sum_j C_j Delta_j A`` with ``Delta_j A(xi) = A(xi+j) + A(xi-j) - 2 A(xi)``,
        evaluated by calling ``A`` at the shifted points.
    """

    A: np.ndarray = field(repr=False)
    A_prime: np.ndarray = field(repr=False)
    A_second: np.ndarray = field(repr=False)
    coupling_delta: np.ndarray = field(repr=False)


def _template_A(x: np.ndarray, width: float) -> np.ndarray:
    """``A(x) = 2 pi - 4 arctan(exp(x / w))``, evaluated at arbitrary (shifted) points."""
    return 2.0 * np.pi - 4.0 * np.arctan(np.exp(np.clip(x / width, -_CLIP, _CLIP)))


def template_arrays(
    xi: np.ndarray, width: float, couplings: Sequence[Tuple[int, float]]
) -> Template:
    """Evaluate the template and its derivatives / shift Laplacians on ``xi``.

    ``A`` is *not* periodic on ``[-L, L)`` -- it jumps by ``2 pi`` across the window --
    so ``A'``, ``A''`` and ``Delta_j A`` must NEVER be computed spectrally: an FFT would
    see the wrap-around jump and contaminate the whole domain with Gibbs oscillations.
    ``A'`` and ``A''`` are therefore taken in closed form, and ``Delta_j A`` by evaluating
    the analytic ``A`` at ``xi +/- j`` directly.  Only the periodic remainder ``psi`` is
    ever differentiated spectrally.
    """
    w = float(width)
    if not w > 0.0:
        raise ValueError(f"template width must be positive, got {w!r}.")
    xi = np.asarray(xi, dtype=float)

    A = _template_A(xi, w)
    scaled = np.clip(xi / w, -_CLIP, _CLIP)
    sech = 1.0 / np.cosh(scaled)
    A_prime = -(2.0 / w) * sech
    A_second = (2.0 / w**2) * sech * np.tanh(xi / w)

    coupling_delta = np.zeros_like(xi)
    for j, coeff in couplings:
        shift = float(j)
        coupling_delta = coupling_delta + coeff * (
            _template_A(xi + shift, w) + _template_A(xi - shift, w) - 2.0 * A
        )
    return Template(A, A_prime, A_second, coupling_delta)


@dataclass(frozen=True)
class LatticeModel:
    """Parameters of the damped, driven lattice.

    Parameters
    ----------
    gamma:
        Uniform scalar damping.
    mu:
        Amplitude of the first harmonic of ``V'``.
    mu2:
        Amplitude of the second harmonic of ``V'`` (0 for Frenkel-Kontorova).
    couplings:
        ``((j, C_j), ...)`` -- symmetric finite-range coupling, range ``j`` with
        coefficient ``C_j``.  The default ``((1, 1.0),)`` is nearest-neighbour.
    template_width:
        Width ``w`` of the fixed analytic template ``A``.  This is a *numerical* choice,
        not a physical one: the converged ``phi = A + psi`` does not depend on it (only
        the split between ``A`` and ``psi`` does).
    """

    gamma: float = 0.1
    mu: float = 1.0
    mu2: float = 0.0
    couplings: Tuple[Tuple[int, float], ...] = ((1, 1.0),)
    template_width: float = 2.0

    def __post_init__(self) -> None:
        couplings = tuple((int(j), float(coeff)) for j, coeff in self.couplings)
        if not couplings:
            raise ValueError("couplings must contain at least one (range, coefficient).")
        for j, _ in couplings:
            if j < 1:
                raise ValueError(f"coupling ranges must be >= 1, got j={j}.")
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "mu", float(self.mu))
        object.__setattr__(self, "mu2", float(self.mu2))
        object.__setattr__(self, "template_width", float(self.template_width))

    # -- substrate ---------------------------------------------------------------
    def Vp(self, u: np.ndarray) -> np.ndarray:
        """``V'(u) = mu sin(u) + mu2 sin(2 u)``."""
        return self.mu * np.sin(u) + self.mu2 * np.sin(2.0 * u)

    def Vpp(self, u: np.ndarray) -> np.ndarray:
        """``V''(u) = mu cos(u) + 2 mu2 cos(2 u)``."""
        return self.mu * np.cos(u) + 2.0 * self.mu2 * np.cos(2.0 * u)

    # -- symbols -----------------------------------------------------------------
    def lin_symbol(self, k: np.ndarray, N: int, c: float) -> np.ndarray:
        """Symbol of ``Lin(c) = c^2 d_xi^2 - gamma c d_xi - sum_j C_j Delta_j``."""
        s = (c**2) * sym_d2(k) - self.gamma * c * sym_d1(k, N)
        for j, coeff in self.couplings:
            s = s - coeff * sym_shift_laplacian(k, j)
        return s

    def M1_symbol(self, k: np.ndarray, N: int, c: float) -> np.ndarray:
        """Symbol of ``M1(c) = -2 c d_xi + gamma``."""
        return -2.0 * c * sym_d1(k, N) + self.gamma

    # -- template ----------------------------------------------------------------
    def template(self, xi: np.ndarray, width: float | None = None) -> Template:
        """Template arrays on ``xi``; ``width`` defaults to :attr:`template_width`."""
        return template_arrays(
            xi, self.template_width if width is None else width, self.couplings
        )
