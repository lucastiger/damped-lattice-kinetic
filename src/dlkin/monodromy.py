"""Direct time integration of the linearized lattice, and the structure of its monodromy.

Everything in :mod:`dlkin.spectral` works with the co-traveling pencil
``Q(nu) = nu^2 + nu M1 + M0``, which is an *ansatz*: it presupposes that the Floquet
solutions of the linearized lattice have the form ``e^{nu t} p(n - c t)``.  This module
does the other thing -- it integrates the linearized lattice in time and forms the
monodromy matrix directly -- so that the two can be compared without either assuming the
other.

Both computations here rest on R1 of ``handoff/SCIENCE_BRIEF.md``.  Writing the linearized
lattice as a first-order system in ``(x, x')``,

    d/dt [x; x'] = A(t) [x; x'],      A(t) = [[0, I], [K(t), -gamma I]]

with ``K(t) = sum_j C_j Delta_j - diag(V''(phi(xi - c t)))`` symmetric, one has
``A^T J + J A = -gamma J`` for *any* symmetric ``K``, hence

    Phi(T)^T J Phi(T) = e^{-gamma T} J

for the fundamental solution, and the same for ``M = S Phi(T)`` with ``S`` the one-site
shift, which is orthogonal and symplectic.  The consequence is the multiplier pairing
``rho rho_hat = e^{-gamma T}``: dissipation does not destroy the pairing, it rescales it.

:func:`conformal_symplectic_flow_test` checks the algebra on a random symmetric ``K(t)``
with no lattice in sight; :func:`lattice_monodromy` checks it on the actual
Frenkel-Kontorova monodromy about a computed traveling kink, and compares the multipliers
with ``exp(nu/c)`` from the pencil.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
from scipy.linalg import expm

from .grid import Grid
from .solver import TravelingWaveSolver

__all__ = [
    "symplectic_form",
    "shift_matrix",
    "conformal_symplectic_residual",
    "conformal_symplectic_flow_test",
    "interpolate_psi",
    "LatticeMonodromy",
    "lattice_monodromy",
]


# ------------------------------------------------------------- structure ----
def symplectic_form(n: int) -> np.ndarray:
    """``J = [[0, I], [-I, 0]]`` on ``R^{2n}``."""
    I, Z = np.eye(n), np.zeros((n, n))
    return np.block([[Z, I], [-I, Z]])


def shift_matrix(n: int, step: int = 1) -> np.ndarray:
    """Permutation ``(P x)_i = x_{i + step}`` on a ring of ``n`` sites.

    ``step = 1`` is the shift that turns ``x_i(T) = f(i - 1)`` back into ``f(i)`` after one
    period ``T = 1/c``, so it is the one that makes the translation mode an eigenvector of
    the monodromy at multiplier ``+1``.  ``P`` is orthogonal and commutes with the
    ``(x, x')`` block structure, so ``diag(P, P)`` is symplectic and leaves the conformal
    relation untouched.
    """
    P = np.zeros((n, n))
    P[np.arange(n), (np.arange(n) + step) % n] = 1.0
    return P


def conformal_symplectic_residual(M: np.ndarray, gamma: float, T: float) -> float:
    """``|M^T J M - e^{-gamma T} J|_max / |J|_max``."""
    n = M.shape[0] // 2
    J = symplectic_form(n)
    return float(np.max(np.abs(M.T @ J @ M - np.exp(-gamma * T) * J)) / np.max(np.abs(J)))


# ------------------------------------------------------- integrating A(t) ---
def _rk4(deriv: Callable[[float, np.ndarray], np.ndarray], Y: np.ndarray, t0: float, dt: float, n_steps: int) -> np.ndarray:
    """Classical RK4 on ``Y' = deriv(t, Y)``; fixed step, hence deterministic."""
    t = t0
    for _ in range(n_steps):
        k1 = deriv(t, Y)
        k2 = deriv(t + 0.5 * dt, Y + (0.5 * dt) * k1)
        k3 = deriv(t + 0.5 * dt, Y + (0.5 * dt) * k2)
        k4 = deriv(t + dt, Y + dt * k3)
        Y = Y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
    return Y


def conformal_symplectic_flow_test(
    n: int = 8,
    gamma: float = 0.37,
    T: float = 0.9,
    dt: float = 1e-4,
    seed: int = 1,
    integrator: str = "expm",
) -> Dict[str, object]:
    """``M^T J M = e^{-gamma T} J`` for a freshly resampled symmetric ``K(t)`` each step.

    The point of the test is that R1 needs *nothing* of ``K`` beyond symmetry: no lattice,
    no traveling wave, no smoothness in ``t``.  ``K`` is redrawn from a seeded Gaussian at
    every step, so the coefficient matrix is as far from the physical one as it can be, and
    the relation still holds to round-off.

    ``integrator``:

    ``"expm"``
        the reference method: ``Phi <- expm(A dt) Phi``.  Each factor satisfies the
        relation exactly in exact arithmetic, so what the residual measures is the
        round-off accumulated over the steps.  This is what ``claims.yaml`` records.
    ``"rk4"``
        classical RK4, which imposes nothing.  The residual is then the integration error,
        and its decrease with ``dt`` is evidence that the relation is a property of the
        flow rather than of the integrator.

    The number of steps is ``round(T/dt)`` and the step actually taken is ``T/n_steps``, so
    the integrated time is exactly ``T``.  That matters: integrating to ``T - dt`` instead
    and comparing against ``e^{-gamma T}`` would show a spurious error of ``gamma*dt``,
    which at ``dt = 1e-4`` is 3.7e-5 -- seven orders above the tolerance being tested.
    """
    if integrator not in ("expm", "rk4"):
        raise ValueError(f"integrator must be 'expm' or 'rk4', got {integrator!r}.")
    n_steps = int(round(T / dt))
    if n_steps < 1:
        raise ValueError(f"T/dt = {T / dt} rounds to {n_steps} steps; need at least one.")
    step = T / n_steps

    rng = np.random.default_rng(seed)
    Z = np.zeros((n, n))
    I = np.eye(n)
    Phi = np.eye(2 * n)

    def draw_K() -> np.ndarray:
        K = rng.standard_normal((n, n))
        return 0.5 * (K + K.T)

    for _ in range(n_steps):
        K = draw_K()
        A = np.block([[Z, I], [K, -gamma * I]])
        if integrator == "expm":
            Phi = expm(A * step) @ Phi
        else:
            Phi = _rk4(lambda _t, Y, A=A: A @ Y, Phi, 0.0, step, 1)

    P = shift_matrix(n)
    M = np.block([[P, Z], [Z, P]]) @ Phi
    return {
        "n": n,
        "gamma": gamma,
        "T": T,
        "dt": step,
        "n_steps": n_steps,
        "seed": seed,
        "integrator": integrator,
        "relerr": conformal_symplectic_residual(M, gamma, T),
    }


# -------------------------------------------------------- the FK monodromy --
def interpolate_psi(
    psi: np.ndarray, grid: Grid, x: np.ndarray, derivative: int = 0
) -> np.ndarray:
    """Evaluate ``psi`` (or a derivative of it) at arbitrary ``x`` by its Fourier series.

    ``psi`` is a band-limited periodic function sampled at ``xi_j = -L + j h``, so its
    trigonometric interpolant is the exact object the spectral operators act on; there is
    no interpolation error to speak of, only the Nyquist mode's usual ambiguity, whose
    coefficient here is at round-off.  This is what lets the profile be sampled at integer
    lattice sites, which are not grid points.

    ``derivative`` multiplies the coefficients by ``(ik)^derivative`` before summing, so the
    derivatives are spectrally exact too -- a finite difference of the interpolant would
    cap ``phi''`` at about ``1e-8`` and hide the quantity being measured.  As in
    :func:`dlkin.grid.sym_d1`, the Nyquist mode is zeroed for odd orders.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    coeff = np.fft.fft(psi)
    if derivative:
        factor = (1j * grid.k) ** derivative
        if derivative % 2:
            factor[grid.N // 2] = 0.0
        coeff = coeff * factor
    phase = np.exp(1j * np.outer(x + grid.L, grid.k))
    return np.real(phase @ coeff) / grid.N


@dataclass(frozen=True)
class LatticeMonodromy:
    """The monodromy of the linearized lattice about a traveling kink, and its diagnostics."""

    c: float
    n_sites: int
    T: float
    dt: float
    n_steps: int
    site_xi: np.ndarray = field(repr=False)
    M: np.ndarray = field(repr=False)
    multipliers: np.ndarray = field(repr=False)
    cs_relerr: float
    tw_residual_max: float
    potential_wrap_mismatch: float
    elapsed_sec: float


def lattice_monodromy(
    solver: TravelingWaveSolver,
    psi: np.ndarray,
    drive: float,
    c: float,
    n_sites: int = 200,
    n_steps: int = 1124,
) -> LatticeMonodromy:
    """Integrate the linearized lattice over one period and compose with the one-site shift.

    The lattice is a ring of ``n_sites`` sites carrying integer positions
    ``xi = -n_sites//2 ... n_sites//2 - 1``, with the kink centred at ``xi = 0``.  The
    linearization about ``u_n(t) = phi(n - c t)`` is

        x'' + gamma x' = sum_j C_j Delta_j x - V''(phi(xi - c t)) x,

    whose coefficient is ``2 pi``-periodic in ``phi`` and therefore continuous around the
    ring even though ``phi`` itself jumps by ``2 pi`` across it.  ``potential_wrap_mismatch``
    reports how continuous: it is the kink's own exponential tail at the ring's seam, and
    it is the one place where the ring is not the infinite lattice.

    Integration is by **RK4**, which imposes no structure at all.  ``cs_relerr`` is then a
    genuine measurement: nothing in the computation was arranged to make
    ``M^T J M = e^{-gamma T} J`` hold.  (Contrast
    :func:`conformal_symplectic_flow_test`'s ``"expm"`` mode, where each step satisfies it
    by construction.)

    ``T = 1/c`` is one period exactly, and ``dt = T/n_steps``.
    """
    grid, model = solver.grid, solver.model
    c = float(c)
    if not c > 0.0:
        raise ValueError(f"lattice_monodromy needs c > 0, got {c}.")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")

    import time  # noqa: PLC0415 -- only to time the integration, nothing branches on it

    started = time.perf_counter()
    n = int(n_sites)
    T = 1.0 / c
    dt = T / n_steps
    site_xi = np.arange(n) - n // 2

    w = model.template_width

    def profile(x: np.ndarray, derivative: int = 0) -> np.ndarray:
        """``phi = A + psi`` at arbitrary points: ``A`` in closed form, ``psi`` spectrally."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        scaled = np.clip(x / w, -600.0, 600.0)
        sech = 1.0 / np.cosh(scaled)
        if derivative == 0:
            A = 2.0 * np.pi - 4.0 * np.arctan(np.exp(scaled))
        elif derivative == 1:
            A = -(2.0 / w) * sech
        elif derivative == 2:
            A = (2.0 / w**2) * sech * np.tanh(scaled)
        else:
            raise ValueError(f"profile supports derivative 0, 1 or 2, got {derivative}.")
        return A + interpolate_psi(psi, grid, x, derivative)

    # -- how well the interpolated profile still solves (TW) at the site positions -----
    #    Collocation forces the residual to vanish at the N grid points; the sites are at
    #    integer xi, which are not grid points, so this is an independent check.
    xs = site_xi.astype(float)
    phi_x = profile(xs)
    coupling = np.zeros_like(xs)
    for j, coeff in model.couplings:
        coupling += coeff * (profile(xs + j) + profile(xs - j) - 2.0 * phi_x)
    tw_residual = (
        c**2 * profile(xs, 2)
        - model.gamma * c * profile(xs, 1)
        - coupling
        - float(drive)
        + model.Vp(phi_x)
    )
    tw_residual_max = float(np.max(np.abs(tw_residual)))

    # -- the ring's seam: V'' must be continuous around it ----------------------------
    seam = []
    for t in np.linspace(0.0, T, 8, endpoint=False):
        left = model.Vpp(profile(np.array([site_xi[0] - c * t], dtype=float)))[0]
        right = model.Vpp(profile(np.array([site_xi[-1] + 1.0 - c * t], dtype=float)))[0]
        seam.append(abs(left - right))
    potential_wrap_mismatch = float(max(seam))

    # -- the coupling operator, applied by rolls rather than as a dense matrix ---------
    couplings = tuple(model.couplings)

    def apply_coupling(X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X)
        for j, coeff in couplings:
            out += coeff * (np.roll(X, -j, axis=0) + np.roll(X, j, axis=0) - 2.0 * X)
        return out

    # -- V''(phi(xi - c t)) at the sites, the only time-dependent part of A(t) ---------
    #    The sites translate rigidly, so the Fourier evaluation of psi at time t differs
    #    from the one at t = 0 by the scalar phase exp(-i k c t) on each mode.  Building
    #    the (n_sites x N) phase matrix once and reusing it turns what would be ~2e9
    #    complex exponentials over the run into one matrix-vector product per evaluation.
    coeff0 = np.fft.fft(psi)
    base_phase = np.exp(1j * np.outer(site_xi + grid.L, grid.k))
    kc = grid.k * c

    def potential_at(t: float) -> np.ndarray:
        scaled = np.clip((site_xi - c * t) / w, -600.0, 600.0)
        A = 2.0 * np.pi - 4.0 * np.arctan(np.exp(scaled))
        psi_t = np.real(base_phase @ (coeff0 * np.exp(-1j * kc * t))) / grid.N
        return model.Vpp(A + psi_t)

    def rate(v: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [Y[n:], apply_coupling(Y[:n]) - v[:, None] * Y[:n] - model.gamma * Y[n:]]
        )

    # Classical RK4, which imposes no structure -- see the docstring.  The stage times are
    # t, t + dt/2 (twice) and t + dt, so each step needs two new potential evaluations and
    # reuses the previous step's endpoint.
    Phi = np.eye(2 * n)
    v_start = potential_at(0.0)
    for i in range(n_steps):
        t = i * dt
        v_mid = potential_at(t + 0.5 * dt)
        v_end = potential_at(t + dt)
        k1 = rate(v_start, Phi)
        k2 = rate(v_mid, Phi + (0.5 * dt) * k1)
        k3 = rate(v_mid, Phi + (0.5 * dt) * k2)
        k4 = rate(v_end, Phi + dt * k3)
        Phi = Phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        v_start = v_end
    P = shift_matrix(n)
    Z = np.zeros((n, n))
    M = np.block([[P, Z], [Z, P]]) @ Phi

    return LatticeMonodromy(
        c=c,
        n_sites=n,
        T=T,
        dt=dt,
        n_steps=n_steps,
        site_xi=site_xi,
        M=M,
        multipliers=np.linalg.eigvals(M),
        cs_relerr=conformal_symplectic_residual(M, model.gamma, T),
        tw_residual_max=tw_residual_max,
        potential_wrap_mismatch=potential_wrap_mismatch,
        elapsed_sec=time.perf_counter() - started,
    )
