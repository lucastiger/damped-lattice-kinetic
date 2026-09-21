"""Traveling-wave solver: damped Newton on the collocated advance-delay equation.

Unknowns are the periodic remainder ``psi = phi - A`` on the grid plus the scalar
``drive``, which is ``mu * sigma`` for the Frenkel-Kontorova model and the additive force
``f`` for the generalized model.  Using ``drive`` throughout -- rather than ``sigma`` --
is what lets the two models share a single code path: the drive enters (TW) additively
and undifferentiated in both cases, so its Jacobian column is ``-1`` regardless of ``mu``.
:attr:`NewtonResult.sigma` converts back for the Frenkel-Kontorova reading.

The pinning row ``psi(0) = 0`` (equivalently ``phi(0) = pi``) removes the translation
freedom and closes the ``(N + 1) x (N + 1)`` system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Tuple

import numpy as np

from .grid import Grid, circulant_from_symbol, spectral_d1
from .model import LatticeModel, Template

__all__ = [
    "ConvergenceError",
    "NewtonResult",
    "TravelingWaveSolver",
    "init_branch",
]

logger = logging.getLogger(__name__)


class ConvergenceError(RuntimeError):
    """Raised when Newton fails to reach an acceptable residual.

    Carries the last iterate so that a caller can inspect or restart from it.
    """

    def __init__(
        self,
        message: str,
        psi: np.ndarray | None = None,
        drive: float | None = None,
        residual: float | None = None,
        iters: int | None = None,
    ) -> None:
        super().__init__(message)
        self.psi = psi
        self.drive = drive
        self.residual = residual
        self.iters = iters


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of one damped-Newton solve at fixed ``c``."""

    psi: np.ndarray = field(repr=False)
    drive: float
    residual: float
    iters: int
    converged: bool
    mu: float = 1.0

    @property
    def sigma(self) -> float:
        """The Frenkel-Kontorova driving force, ``sigma = drive / mu``."""
        return self.drive / self.mu


class TravelingWaveSolver:
    """Fourier-spectral collocation of (TW), with a damped Newton solve at fixed ``c``.

    The operators ``Lin(c)`` and ``M1(c)`` are ``c``-dependent and are cached by
    :meth:`build`; the template arrays are ``c``-independent and are built once.
    """

    def __init__(self, grid: Grid, model: LatticeModel) -> None:
        self.grid = grid
        self.model = model
        self._template: Template = model.template(grid.xi)
        self._c: float | None = None
        self._Lin: np.ndarray | None = None
        self._M1: np.ndarray | None = None

    # -- construction ------------------------------------------------------------
    def build(self, c: float) -> None:
        """Assemble ``Lin(c)`` and ``M1(c)``; a no-op if already built at this ``c``."""
        c = float(c)
        if self._c is not None and c == self._c:
            return
        k, N = self.grid.k, self.grid.N
        self._Lin = circulant_from_symbol(self.model.lin_symbol(k, N, c), N)
        self._M1 = circulant_from_symbol(self.model.M1_symbol(k, N, c), N)
        self._c = c
        logger.debug("built operators at c=%r (N=%d)", c, N)

    def _require_built(self) -> None:
        if self._c is None:
            raise RuntimeError("call build(c) before using the cached operators.")

    @property
    def c(self) -> float:
        """The velocity the cached operators were built at."""
        self._require_built()
        return self._c  # type: ignore[return-value]

    @property
    def Lin(self) -> np.ndarray:
        """``c^2 d_xi^2 - gamma c d_xi - sum_j C_j Delta_j`` as a real circulant."""
        self._require_built()
        return self._Lin  # type: ignore[return-value]

    @property
    def M1(self) -> np.ndarray:
        """``-2 c d_xi + gamma`` as a real circulant."""
        self._require_built()
        return self._M1  # type: ignore[return-value]

    @property
    def template(self) -> Template:
        """Template arrays ``A``, ``A'``, ``A''``, ``sum_j C_j Delta_j A``."""
        return self._template

    @property
    def A(self) -> np.ndarray:
        return self._template.A

    @property
    def A_prime(self) -> np.ndarray:
        return self._template.A_prime

    def sigma_of(self, drive: float) -> float:
        """Convert an internal ``drive`` back to ``sigma = drive / mu``."""
        return drive / self.model.mu

    # -- the equations -----------------------------------------------------------
    def residual(self, psi: np.ndarray, drive: float, c: float) -> np.ndarray:
        """The ``N`` collocated residuals of (TW) at ``phi = A + psi``.

        ``R = [c^2 A'' - gamma c A' - sum_j C_j Delta_j A] + Lin psi - drive + V'(A + psi)``
        """
        self.build(c)
        t = self._template
        lin_A = c**2 * t.A_second - self.model.gamma * c * t.A_prime - t.coupling_delta
        return lin_A + self._Lin @ psi - drive + self.model.Vp(t.A + psi)

    def M0(self, psi: np.ndarray) -> np.ndarray:
        """Linearization ``M0 = Lin + diag(V''(A + psi))`` -- the ``psi`` Jacobian block."""
        self._require_built()
        return self._Lin + np.diag(self.model.Vpp(self._template.A + psi))

    def phi(self, psi: np.ndarray) -> np.ndarray:
        """The profile ``phi = A + psi``."""
        return self._template.A + psi

    def phi_prime(self, psi: np.ndarray) -> np.ndarray:
        """``phi' = A' + (spectral derivative of psi)``.

        ``A'`` is closed-form because ``A`` is not periodic; ``psi`` is, so it is
        differentiated spectrally.
        """
        return self._template.A_prime + spectral_d1(psi, self.grid)

    # -- Newton ------------------------------------------------------------------
    def newton(
        self,
        c: float,
        psi0: np.ndarray,
        drive0: float,
        tol: float = 1e-12,
        maxit: int = 40,
        tol_fail: float = 1e-9,
    ) -> NewtonResult:
        """Damped Newton on ``(psi, drive)`` at fixed ``c``.

        The Jacobian is the bordered matrix::

            [[ M0, -1 ],
             [ e_j0^T, 0 ]]

        The drive column is ``-1`` (not ``-mu``) because ``drive`` already absorbs ``mu``.

        Step acceptance is a backtracking line search: halve ``lam`` until
        ``max(|R|_inf, |psi[j0]|)`` decreases, stopping the halving once ``lam`` drops
        below ``1e-3`` (so the smallest factor actually taken is ``2^-10``).

        Raises
        ------
        ConvergenceError
            If the iteration exhausts ``maxit`` and the final residual still exceeds
            ``tol_fail``.  The solver never returns a silently unconverged iterate.
        """
        if maxit < 1:
            raise ValueError(f"maxit must be >= 1, got {maxit}.")
        self.build(c)
        N, j0 = self.grid.N, self.grid.j0
        psi = np.array(psi0, dtype=float, copy=True)
        if psi.shape != (N,):
            raise ValueError(f"psi0 must have shape ({N},), got {psi.shape}.")
        drive = float(drive0)

        for it in range(maxit):
            R = self.residual(psi, drive, c)
            g = psi[j0]
            res = max(float(np.max(np.abs(R))), abs(float(g)))
            if res < tol:
                logger.debug("newton converged at c=%r in %d iters, res=%.3e", c, it, res)
                return NewtonResult(psi, drive, res, it, True, self.model.mu)

            J = np.zeros((N + 1, N + 1))
            J[:N, :N] = self.M0(psi)
            J[:N, N] = -1.0
            J[N, j0] = 1.0
            dx = np.linalg.solve(J, np.concatenate([-R, [-g]]))

            lam = 1.0
            for _ in range(20):
                psi_new, drive_new = psi + lam * dx[:N], drive + lam * dx[N]
                R_new = self.residual(psi_new, drive_new, c)
                trial = max(float(np.max(np.abs(R_new))), abs(float(psi_new[j0])))
                if trial < res or lam < 1e-3:
                    break
                lam *= 0.5
            psi, drive = psi_new, drive_new

        R = self.residual(psi, drive, c)
        res = max(float(np.max(np.abs(R))), abs(float(psi[j0])))
        if res > tol_fail:
            raise ConvergenceError(
                f"Newton failed to converge at c={c!r}: after {maxit} iterations the "
                f"residual max(|R|_inf, |psi(0)|) = {res:.3e} still exceeds "
                f"tol_fail={tol_fail:.3e}.",
                psi=psi,
                drive=drive,
                residual=res,
                iters=maxit,
            )
        logger.debug(
            "newton exhausted maxit=%d at c=%r with res=%.3e (<= tol_fail)", maxit, c, res
        )
        return NewtonResult(psi, drive, res, maxit, res < tol, self.model.mu)


def init_branch(
    grid: Grid,
    model: LatticeModel | None = None,
    c0: float = 0.88,
    drive_guess: float | None = None,
    tol: float = 1e-12,
    maxit: int = 40,
) -> Tuple[TravelingWaveSolver, np.ndarray, float]:
    """Land on the traveling-kink branch at ``c0`` by the reference two-stage procedure.

    Stage 1 solves from ``psi = 0`` with the *physical-width* template
    ``w0 = sqrt((1 - c0^2) / mu)`` -- the sine-Gordon kink width at speed ``c0``, which is
    close enough to the true profile for Newton to find the branch from a flat start.
    Stage 2 rebuilds the solver with the model's fixed ``template_width`` (default 2.0),
    transfers the converged profile as ``psi = phi_converged - A_new``, and re-converges.

    Do not "simplify" this to a single stage.  The reported numbers depend on the branch
    this procedure lands on, and the fixed-width template is what every downstream
    quantity (``phi'``, ``kappa``, the continuation records) is defined against.

    Returns
    -------
    (solver, psi, drive)
        ``solver`` is built at ``c0`` and carries the fixed-width template.
        ``drive_guess`` defaults to ``0.6 * mu`` (i.e. ``sigma = 0.6``).
    """
    if model is None:
        model = LatticeModel()
    if drive_guess is None:
        drive_guess = 0.6 * model.mu
    c0 = float(c0)

    w0 = np.sqrt((1.0 - c0**2) / model.mu)
    stage1 = TravelingWaveSolver(grid, replace(model, template_width=float(w0)))
    first = stage1.newton(c0, np.zeros(grid.N), drive_guess, tol=tol, maxit=maxit)
    logger.debug(
        "init_branch stage 1 (w0=%.6f): drive=%.12f res=%.3e", w0, first.drive, first.residual
    )

    profile = stage1.phi(first.psi)
    solver = TravelingWaveSolver(grid, model)
    solver.build(c0)
    second = solver.newton(c0, profile - solver.A, first.drive, tol=tol, maxit=maxit)
    logger.debug(
        "init_branch stage 2 (w=%.6f): drive=%.12f res=%.3e",
        model.template_width,
        second.drive,
        second.residual,
    )
    return solver, second.psi, second.drive
