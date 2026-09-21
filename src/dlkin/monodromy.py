"""Conformal symplecticity (R1) and an independent lattice monodromy cross-check.

Two unrelated things live here, both about the *time-dependent* picture that the
co-traveling pencil of :mod:`dlkin.spectral` replaces:

:func:`conformal_symplectic_flow_test`
    The identity ``A^T J + J A = -gamma J`` for ``A = [[0, I], [K(t), -gamma I]]`` with any
    symmetric ``K``, integrated: ``Phi(T)^T J Phi(T) = e^{-gamma T} J``.  This is R1 and it
    underlies the manuscript's multiplier pairing ``rho rho_hat = e^{-gamma/c}``.

:func:`lattice_monodromy`
    An INDEPENDENT CROSS-CHECK that is **not used by the manuscript**.  It samples the
    computed ``xi``-profile at integer lattice sites, runs the genuine nonlinear lattice
    forward one period together with its fundamental matrix, applies the one-site shift,
    and compares the resulting Floquet multipliers with ``exp(nu/c)`` from the pencil.  See
    that function's docstring for its accuracy limits -- they are real and are reported,
    not tuned away.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .model import template_arrays
from .solver import TravelingWaveSolver
from .spectral import pencil_eigs

__all__ = [
    "symplectic_form",
    "conformal_symplectic_flow_test",
    "lattice_monodromy",
]

logger = logging.getLogger(__name__)


def symplectic_form(n: int) -> np.ndarray:
    """``J = [[0, I], [-I, 0]]`` on ``R^{2n}``."""
    zero, eye = np.zeros((n, n)), np.eye(n)
    return np.block([[zero, eye], [-eye, zero]])


def conformal_symplectic_flow_test(
    n: int = 8, gamma: float = 0.37, T: float = 0.9, dt: float = 1e-4, seed: int = 1
) -> float:
    """``max|M^T J M - e^{-gamma T} J| / max|J|`` for an arbitrary symmetric coupling.

    ``K(t)`` is **resampled at every step** from a fixed-seed generator.  That is
    deliberate, not sloppiness: it demonstrates that conformal symplecticity is a property
    of the *structure* ``A = [[0, I], [K, -gamma I]]`` with ``K = K^T`` and uniform scalar
    damping, not of the particular Frenkel-Kontorova coupling.  The flow is composed as
    ``expm(A dt)`` per step, and the one-site cyclic shift is applied at the end, so the
    quantity tested is the full monodromy of the note, not just ``Phi(T)``.

    Deterministic: the generator is created locally from ``seed``.
    """
    rng = np.random.default_rng(seed)
    J = symplectic_form(n)
    Phi = np.eye(2 * n)
    zero, eye = np.zeros((n, n)), np.eye(n)
    t = 0.0
    while t < T - 1e-12:
        K = rng.standard_normal((n, n))
        K = (K + K.T) / 2.0
        A = np.block([[zero, eye], [K, -gamma * eye]])
        Phi = expm(A * dt) @ Phi
        t += dt

    P = np.zeros((n, n))
    P[np.arange(n), (np.arange(n) + 1) % n] = 1.0
    M = np.block([[P, zero], [zero, P]]) @ Phi
    relerr = float(
        np.max(np.abs(M.T @ J @ M - np.exp(-gamma * T) * J)) / np.max(np.abs(J))
    )
    logger.debug("conformal symplectic relerr (n=%d, T=%g, dt=%g): %.3e", n, T, dt, relerr)
    return relerr


def _interpolate_periodic(values: np.ndarray, grid, x: np.ndarray, derivative: bool):
    """Trigonometric (FFT) interpolation of a periodic grid function at arbitrary ``x``.

    Exact at the collocation points.  With ``derivative=True`` the band-limited derivative
    is returned, using the same Nyquist-zeroed symbol as :func:`dlkin.grid.sym_d1`, so the
    interpolated derivative is consistent with ``spectral_d1`` on the grid.
    """
    N = grid.N
    coefficients = np.fft.fft(values)
    k = np.asarray(grid.k, dtype=float).copy()
    if derivative:
        symbol = 1j * k
        symbol[N // 2] = 0.0
        coefficients = coefficients * symbol
    phase = np.exp(1j * np.outer(np.asarray(x, dtype=float) + grid.L, k))
    return np.real(phase @ coefficients) / N


def _charge_shift(u: np.ndarray, j: int, jump: float) -> np.ndarray:
    """``u_{n+j}`` on a ring of ``len(u)`` sites carrying a topological charge.

    The kink drops by ``jump`` (= ``2 pi``) across the window, so going once around the
    ring subtracts it: ``u_{n + n_sites} = u_n - jump``.  Positive ``j`` wraps at the right
    edge, negative ``j`` at the left.
    """
    rolled = np.roll(u, -j)
    if j > 0:
        rolled[-j:] -= jump
    elif j < 0:
        rolled[: -j] += jump
    return rolled


def lattice_monodromy(
    solver: TravelingWaveSolver,
    psi: np.ndarray,
    drive: float,
    c: float,
    n_sites: int = 200,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    *,
    pencil_nus: Optional[Sequence[complex]] = None,
    pencil_shift: complex = -0.045,
    pencil_k: int = 24,
    n_report: int = 10,
) -> Dict[str, Any]:
    """Independent cross-check: the real lattice's monodromy versus the pencil spectrum.

    **This is not used by the manuscript.**  Everything the note reports comes from the
    advance-delay formulation in :mod:`dlkin.spectral`.  This function exists to confirm,
    by a completely different route, that the pencil eigenvalues really are the Floquet
    exponents of the damped lattice.

    What it does: sample ``phi = A + psi`` and ``phi'`` at integer sites by trigonometric
    interpolation of ``psi`` plus the closed-form template; set ``u_n(0) = phi(n)``,
    ``udot_n(0) = -c phi'(n)`` on a ring of ``n_sites`` sites with the ``2 pi``-charge
    preserving wrap; integrate the nonlinear lattice and its ``2n x 2n`` fundamental matrix
    over one period ``T = 1/c`` with DOP853; apply the one-site shift; and read off the
    multipliers.

    ACCURACY, HONESTLY.  Two approximations are unavoidable here and neither is tuned:

    1.  *Ring truncation.*  The true kink lives on an infinite lattice; this uses
        ``n_sites`` sites with a charge-preserving wrap, which glues the profile's left
        tail to its right tail.  The tails do not match exactly.  This shows up as
        ``drift`` -- ``max_n |u_{n+1}(T) - u_n(0)|``, which is zero for an exact lattice
        traveling wave.  Measured at ``c = 0.89``, ``L = 200``: ``5.3e-2`` at 40 sites,
        ``6.1e-3`` at 100, and ``4.5e-5`` at 200, 300 and 398 -- so by ``n_sites = 200``
        the ring is no longer what limits it.
    2.  *Resolution of the xi-profile.*  What limits it past that point is how well the
        advance-delay solution is resolved.  At ``N = 1024`` (``h ~ 0.39``) the drift
        sticks at ``4.0e-3`` no matter how large the ring; at ``N = 2048`` and ``N = 4096``
        it drops to ``4.5e-5``.  Sampling itself is spectrally accurate and is not the
        limiting error.

    What this buys, measured at ``c = 0.89``, ``L = 200``, ``N = 4096``, 200 sites:

    ==========================================  ===========  ==============
    quantity                                    value        meaning
    ==========================================  ===========  ==============
    ``rho_translation_err``                     ``9.2e-10``  the translation multiplier
                                                             ``rho = 1``
    ``rho_conformal_err``                       ``7.8e-10``  its partner
                                                             ``rho = e^{-gamma/c}``
    ``circle_radius_median_relerr``             ``6e-14``    the essential-spectrum circle
                                                             ``|rho| = e^{-gamma/(2c)}``
    ``conformal_relerr``                        ``9.5e-11``  ``M^T J M = e^{-gamma/c} J``
    ``drift``                                   ``4.5e-5``   ring truncation
    ==========================================  ===========  ==============

    So the two *isolated* multipliers, which theory pins exactly, come back to ``1e-9``,
    and the essential-spectrum radius to ``1e-13``.  ``multiplier_match_err``, by contrast,
    is NOT small (median ``4.5e-3``) and should not be read as a failure: it nearest-
    neighbour-matches ``exp(nu/c)`` against the ring's multipliers, and away from the two
    isolated modes both sets are merely different discretizations of the same continuous
    essential spectrum -- 200 ring sites versus N advance-delay collocation points -- so
    individual points have no reason to coincide.  The circle they lie on does, and that is
    what ``circle_radius_median_relerr`` measures.

    COST.  The ODE carries ``2 n_sites + (2 n_sites)^2`` states; at ``n_sites = 200`` that
    is 160400 and the integration takes on the order of a minute.  Tests that call this are
    marked ``slow``.

    Returns a dict with ``multipliers``, ``multipliers_predicted`` (``exp(nu/c)`` from the
    pencil), ``multiplier_match_err``, ``rho_translation_err``, ``rho_conformal_err``,
    ``conformal_relerr``, ``drift``, and the run parameters.
    """
    started = time.time()
    if n_sites < 8 or n_sites % 2 != 0:
        raise ValueError(f"n_sites must be even and >= 8, got {n_sites}.")

    solver.build(c)
    grid, model = solver.grid, solver.model
    gamma = model.gamma
    jump = 2.0 * np.pi
    n = int(n_sites)

    # --- sample the profile at integer sites -----------------------------------------
    sites = np.arange(n, dtype=float) - n // 2
    if sites[0] < grid.xi[0] or sites[-1] > grid.xi[-1]:
        raise ValueError(
            f"the ring of {n} sites spans [{sites[0]}, {sites[-1]}], which leaves the "
            f"computational window [{grid.xi[0]}, {grid.xi[-1]}]; enlarge L or shrink n_sites."
        )
    template = template_arrays(sites, model.template_width, model.couplings)
    u0 = template.A + _interpolate_periodic(psi, grid, sites, derivative=False)
    phi_prime_sites = template.A_prime + _interpolate_periodic(
        psi, grid, sites, derivative=True
    )
    v0 = -c * phi_prime_sites

    # --- the lattice and its variational equation ------------------------------------
    couplings = tuple(model.couplings)
    n_state = 2 * n

    def coupling_sum(u: np.ndarray) -> np.ndarray:
        total = np.zeros_like(u)
        for j, coefficient in couplings:
            total = total + coefficient * (
                _charge_shift(u, j, jump) + _charge_shift(u, -j, jump) - 2.0 * u
            )
        return total

    def stiffness(u: np.ndarray) -> np.ndarray:
        """``K = sum_j C_j (S^j + S^-j - 2 I) - diag(V''(u))``; symmetric, as R1 needs."""
        K = -np.diag(model.Vpp(u))
        index = np.arange(n)
        for j, coefficient in couplings:
            for shift in (j, -j):
                K[index, (index + shift) % n] += coefficient
            K[index, index] -= 2.0 * coefficient
        return K

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u, v = y[:n], y[n:n_state]
        Phi = y[n_state:].reshape(n_state, n_state)
        du = v
        dv = coupling_sum(u) + drive - model.Vp(u) - gamma * v
        P1, P2 = Phi[:n], Phi[n:]
        dPhi = np.empty_like(Phi)
        dPhi[:n] = P2
        dPhi[n:] = stiffness(u) @ P1 - gamma * P2
        return np.concatenate([du, dv, dPhi.ravel()])

    period = 1.0 / c
    y0 = np.concatenate([u0, v0, np.eye(n_state).ravel()])
    solution = solve_ivp(
        rhs, (0.0, period), y0, method="DOP853", rtol=rtol, atol=atol, dense_output=False
    )
    if not solution.success:
        raise RuntimeError(f"lattice integration failed: {solution.message}")
    yT = solution.y[:, -1]
    uT, Phi_T = yT[:n], yT[n_state:].reshape(n_state, n_state)

    # --- one-site shift, multipliers, conformal residual -----------------------------
    P = np.zeros((n, n))
    P[np.arange(n), (np.arange(n) + 1) % n] = 1.0
    zero = np.zeros((n, n))
    M = np.block([[P, zero], [zero, P]]) @ Phi_T

    J = symplectic_form(n)
    conformal_relerr = float(
        np.max(np.abs(M.T @ J @ M - np.exp(-gamma / c) * J)) / np.max(np.abs(J))
    )
    multipliers = np.linalg.eigvals(M)
    order = np.argsort(-np.abs(multipliers))
    multipliers = multipliers[order]

    drift = float(np.max(np.abs(_charge_shift(uT, 1, jump) - u0)))

    # --- the two isolated multipliers the theory pins exactly -------------------------
    conformal_rho = float(np.exp(-gamma / c))
    index_translation = int(np.argmin(np.abs(multipliers - 1.0)))
    index_conformal = int(np.argmin(np.abs(multipliers - conformal_rho)))

    # --- the essential-spectrum circle |rho| = exp(-gamma/(2c))  (R3) -----------------
    essential = np.delete(multipliers, sorted({index_translation, index_conformal}))
    circle_expected = float(np.exp(-gamma / (2.0 * c)))
    circle_median = float(np.median(np.abs(essential))) if essential.size else float("nan")

    # --- comparison with exp(nu/c) ---------------------------------------------------
    if pencil_nus is None:
        pencil_nus = pencil_eigs(
            solver.M0(psi), solver.M1, shift=pencil_shift, k=pencil_k
        )
    predicted = np.exp(np.asarray(pencil_nus, dtype=complex) / c)
    match_err = [
        float(np.min(np.abs(multipliers - rho))) for rho in predicted
    ]

    result: Dict[str, Any] = {
        "n_sites": n,
        "c": float(c),
        "period": period,
        "rtol": rtol,
        "atol": atol,
        "n_rhs_evaluations": int(solution.nfev),
        "wall_time_seconds": time.time() - started,
        "multipliers": [[float(z.real), float(z.imag)] for z in multipliers[:n_report]],
        "multipliers_predicted": [
            [float(z.real), float(z.imag)] for z in predicted[:n_report]
        ],
        "multiplier_match_err": match_err,
        "multiplier_match_err_max": float(max(match_err)) if match_err else float("nan"),
        "multiplier_match_err_median": float(np.median(match_err))
        if match_err
        else float("nan"),
        "rho_translation_err": float(abs(multipliers[index_translation] - 1.0)),
        "rho_conformal_err": float(abs(multipliers[index_conformal] - conformal_rho)),
        "rho_conformal_expected": conformal_rho,
        "circle_radius_median": circle_median,
        "circle_radius_expected": circle_expected,
        "circle_radius_median_relerr": abs(circle_median - circle_expected)
        / circle_expected,
        "conformal_relerr": conformal_relerr,
        "drift": drift,
    }
    logger.debug(
        "lattice monodromy (n_sites=%d): drift=%.3e conformal=%.3e rho(1)=%.3e in %.1fs",
        n,
        drift,
        conformal_relerr,
        result["rho_translation_err"],
        result["wall_time_seconds"],
    )
    return result
