"""Continuation of the traveling-wave branch: natural in ``c``, and pseudo-arclength.

Natural continuation walks a prescribed list of velocities, reusing the previous solution
as the initial guess.  It is the method of choice on the rising part of the branch and
fails, by construction, at a turning point in ``c``.

Pseudo-arclength continuation treats ``c`` as an unknown, ``X = (psi, drive, c)`` in
``R^{N+2}``, and closes the system with the arclength condition ``tau . (X - X_pred) = 0``.
That is what carries the branch through ``c_max``.

Everything in this module is deterministic: no random state, no hidden defaults that
depend on process state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np

from .solver import ConvergenceError, TravelingWaveSolver

__all__ = [
    "ContinuationError",
    "ContinuationRecord",
    "natural_continuation",
    "ArclengthSystem",
    "PseudoArclength",
]

logger = logging.getLogger(__name__)


class ContinuationError(RuntimeError):
    """Raised when a continuation run cannot be carried further."""


@dataclass(frozen=True)
class ContinuationRecord:
    """One accepted continuation point."""

    c: float
    psi: np.ndarray = field(repr=False)
    drive: float
    residual: float
    iters: int
    mu: float = 1.0

    @property
    def sigma(self) -> float:
        """The Frenkel-Kontorova driving force, ``sigma = drive / mu``."""
        return self.drive / self.mu


def natural_continuation(
    solver: TravelingWaveSolver,
    psi: np.ndarray,
    drive: float,
    c_values: Iterable[float],
    on_step: Optional[Callable[[ContinuationRecord], None]] = None,
    tol: float = 1e-12,
    maxit: int = 40,
) -> List[ContinuationRecord]:
    """Continue the branch through ``c_values``, previous solution as initial guess.

    Parameters
    ----------
    solver, psi, drive:
        A converged starting point, e.g. from :func:`dlkin.solver.init_branch`.
    c_values:
        The velocities to visit, in order.
    on_step:
        Optional callback invoked with each accepted :class:`ContinuationRecord`.

    Raises
    ------
    ContinuationError
        As soon as one velocity fails to converge, naming the velocity and how far the
        run got.  The run is never continued from an unconverged iterate.
    """
    records: List[ContinuationRecord] = []
    psi = np.asarray(psi, dtype=float)
    drive = float(drive)

    for c in c_values:
        c = float(c)
        try:
            result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
        except ConvergenceError as exc:
            last = records[-1].c if records else None
            raise ContinuationError(
                f"natural continuation failed at c={c!r} after {len(records)} accepted "
                f"step(s) (last converged c={last!r}). Underlying solver error: {exc}. "
                "Near a turning point in c, use PseudoArclength instead."
            ) from exc
        psi, drive = result.psi, result.drive
        record = ContinuationRecord(
            c=c,
            psi=psi,
            drive=drive,
            residual=result.residual,
            iters=result.iters,
            mu=solver.model.mu,
        )
        records.append(record)
        logger.debug(
            "natural continuation: c=%r drive=%.12f res=%.3e", c, drive, result.residual
        )
        if on_step is not None:
            on_step(record)
    return records


class ArclengthSystem(NamedTuple):
    """Return value of :meth:`PseudoArclength.F_and_J`.

    The first two entries are the mathematical content, ``F`` in ``R^{N+1}`` and ``J`` in
    ``R^{(N+1) x (N+2)}``; the remaining three are the intermediate blocks that the
    corrector and the downstream spectral quantities would otherwise have to rebuild.
    """

    F: np.ndarray
    J: np.ndarray
    M0: np.ndarray
    M1: np.ndarray
    phi_prime: np.ndarray


class PseudoArclength:
    """Pseudo-arclength continuation in ``X = (psi, drive, c)`` in ``R^{N+2}``."""

    def __init__(self, solver: TravelingWaveSolver) -> None:
        self.solver = solver
        self.grid = solver.grid
        self.model = solver.model

    # -- packing -----------------------------------------------------------------
    def pack(self, psi: np.ndarray, drive: float, c: float) -> np.ndarray:
        """Assemble the state vector ``X = (psi, drive, c)``."""
        return np.concatenate([np.asarray(psi, dtype=float), [float(drive)], [float(c)]])

    def unpack(self, X: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Split ``X`` into ``(psi, drive, c)``."""
        N = self.grid.N
        return X[:N], float(X[N]), float(X[N + 1])

    # -- the extended system -----------------------------------------------------
    def F_and_J(self, X: np.ndarray) -> ArclengthSystem:
        """Residual and rectangular Jacobian of the branch equations at ``X``.

        ``F = (R(psi, drive, c), psi(0))`` in ``R^{N+1}``; the Jacobian is::

            J = [[ M0,     -1,  dR/dc ],
                 [ e_j0^T,  0,  0     ]]

        with ``dR/dc = -M1 phi'``.  Sign check: the ``c``-dependent part of ``R`` is
        ``c^2 phi'' - gamma c phi'`` (the drive and the coupling terms carry no ``c``), so

            dR/dc = 2 c phi'' - gamma phi' = -(-2 c d_xi + gamma) phi' = -M1 phi'.

        Note ``phi`` itself is held fixed here -- this is the partial derivative that the
        extended Jacobian needs, not the derivative along the branch.  It is checked
        against a centred finite difference in ``tests/test_continuation.py``.
        """
        N, j0 = self.grid.N, self.grid.j0
        psi, drive, c = self.unpack(X)
        solver = self.solver
        solver.build(c)

        R = solver.residual(psi, drive, c)
        g = psi[j0]
        M0 = solver.M0(psi)
        M1 = solver.M1
        p0 = solver.phi_prime(psi)

        J = np.zeros((N + 1, N + 2))
        J[:N, :N] = M0
        J[:N, N] = -1.0
        J[:N, N + 1] = -(M1 @ p0)
        J[N, j0] = 1.0
        return ArclengthSystem(np.concatenate([R, [g]]), J, M0, M1, p0)

    # -- tangent -----------------------------------------------------------------
    def tangent(self, J: np.ndarray, tau_prev: np.ndarray) -> np.ndarray:
        """Unit null vector of ``J``, oriented by continuity with ``tau_prev``.

        Solves the bordered square system ``[J; tau_prev^T] t = e_{N+2}``.  The last row
        forces ``tau_prev . t = 1 > 0``, which both selects the orientation that continues
        the previous direction and normalises away the null vector's free scaling.
        """
        N = self.grid.N
        tau_prev = np.asarray(tau_prev, dtype=float).reshape(1, -1)
        if tau_prev.shape != (1, N + 2):
            raise ValueError(f"tau_prev must have shape ({N + 2},), got {tau_prev.shape[1]}.")
        B = np.vstack([J, tau_prev])
        t = np.linalg.solve(B, np.concatenate([np.zeros(N + 1), [1.0]]))
        return t / np.linalg.norm(t)

    def initial_tangent(self, X: np.ndarray, direction: float = 1.0) -> np.ndarray:
        """Tangent at ``X`` with no previous direction, oriented by ``sign(direction)``.

        Seeds the bordered solve with the pure-``c`` row ``e_{N+2}``, then flips the sign
        so that ``tau[N+1]`` has the requested sign (i.e. ``c`` increases for
        ``direction > 0``).
        """
        N = self.grid.N
        seed = np.zeros(N + 2)
        seed[N + 1] = 1.0
        tau = self.tangent(self.F_and_J(X).J, seed)
        if tau[N + 1] * direction < 0:
            tau = -tau
        return tau

    # -- corrector ---------------------------------------------------------------
    def step(
        self,
        X: np.ndarray,
        tau: np.ndarray,
        ds: float,
        tol: float = 1e-10,
        maxit: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """One predictor-corrector step of length ``ds`` along ``tau``.

        Predictor ``X_pred = X + ds tau``; the corrector is Newton on the square system
        ``[J; tau^T]`` with the arclength row ``tau . (X - X_pred) = 0``.

        Returns
        -------
        (X_new, J, M0, M1, phi_prime, residual)
            All blocks are evaluated at the accepted ``X_new``; ``residual`` is
            ``|F|_inf`` there.
        """
        X_pred = X + ds * tau
        X_new = X_pred.copy()
        for _ in range(maxit):
            F, J, M0, M1, p0 = self.F_and_J(X_new)
            arc = tau @ (X_new - X_pred)
            if max(float(np.max(np.abs(F))), abs(float(arc))) < tol:
                break
            B = np.vstack([J, tau.reshape(1, -1)])
            X_new = X_new + np.linalg.solve(B, -np.concatenate([F, [arc]]))
        F, J, M0, M1, p0 = self.F_and_J(X_new)
        residual = float(np.max(np.abs(F)))
        logger.debug(
            "arclength step: ds=%r c=%.8f drive=%.8f res=%.3e",
            ds,
            X_new[self.grid.N + 1],
            X_new[self.grid.N],
            residual,
        )
        return X_new, J, M0, M1, p0, residual
