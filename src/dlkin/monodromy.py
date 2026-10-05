"""Conformal symplecticity, and the lattice monodromy as an independent cross-check.

Two unrelated computations live here, both about Proposition 3.1 of the note,
``M^T J M = e^{-gamma/c} J``:

:func:`conformal_symplectic_flow_test`
    The *structural* statement.  ``A^T J + J A = -gamma J`` uses nothing about the
    Frenkel-Kontorova lattice beyond three facts -- Newtonian second-order form, a
    symmetric force Jacobian, uniform scalar damping -- so the test resamples a random
    symmetric ``K(t)`` at every step.  If the identity survives an arbitrary symmetric
    coupling that changes at every instant, it is not an artefact of the particular wave.

:func:`lattice_monodromy`
    The *concrete* statement, on the actual lattice.  Everything else in this repository
    works with the advance-delay equation in ``xi``; this function goes back to the lattice
    in ``n`` and ``t``, samples the computed profile at the integer sites, integrates the
    nonlinear lattice and its linearization over one period ``T = 1/c``, applies the
    one-site shift, and compares the resulting Floquet multipliers with ``exp(nu/c)`` from
    the co-traveling pencil.  It is the method of [VCKX20] rather than ours, so agreement
    is a genuinely independent check of the whole chain.

    It is NOT used by any claim in the manuscript.  Its purpose is to fail loudly if the
    xi-formulation has drifted away from the lattice it is supposed to describe.

The monodromy is never formed as a matrix.  One matrix-vector product is one integration
of the ``2 S``-dimensional linearized lattice, so the multipliers come from Arnoldi on a
LinearOperator and the symplectic form is tested on random pairs through

    (M x)^T J (M y)  =  e^{-gamma T} x^T J y ,

which holds for every ``x, y`` exactly when ``M^T J M = e^{-gamma T} J``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigs

from .model import LatticeModel
from .solver import TravelingWaveSolver

__all__ = [
    "symplectic_form",
    "conformal_symplectic_flow_test",
    "sample_profile_at_sites",
    "LatticeMonodromy",
    "lattice_monodromy",
]


def symplectic_form(n: int) -> np.ndarray:
    """``J = [[0, I], [-I, 0]]`` of size ``2n``."""
    eye = np.eye(n)
    zero = np.zeros((n, n))
    return np.block([[zero, eye], [-eye, zero]])


# ------------------------------------------------------- structural test ----
def conformal_symplectic_flow_test(
    n: int = 8,
    gamma: float = 0.37,
    T: float = 0.9,
    dt: float = 1e-4,
    seed: int = 1,
    resample: bool = True,
) -> Dict[str, Any]:
    """Integrate ``dz/dt = A(t) z`` with random symmetric ``K(t)`` and test (R1).

    ``A = [[0, I], [K(t), -gamma I]]``.  The flow is advanced with a matrix exponential per
    step, which is exact for the frozen ``A`` of that step, so the only error in the test is
    the time-splitting -- and the identity it tests is preserved by each exact step, so the
    residual measures arithmetic, not the splitting.

    With ``resample=True`` a fresh symmetric ``K`` is drawn at every step, which is the point
    of the test: no property of the lattice, the potential or the wave is used.  The one-site
    cyclic shift is applied at the end because the monodromy of a traveling wave carries it,
    and it must not disturb the identity (a permutation is symplectic).

    Returns the relative residual ``max|M^T J M - e^{-gamma T} J| / max|J|`` and the
    parameters, so that a sweep can be recorded verbatim.
    """
    if n < 1 or dt <= 0.0 or T <= 0.0:
        raise ValueError("conformal_symplectic_flow_test needs n >= 1 and dt, T > 0.")
    rng = np.random.default_rng(seed)
    J = symplectic_form(n)
    Phi = np.eye(2 * n)
    zero = np.zeros((n, n))
    eye = np.eye(n)

    K = rng.standard_normal((n, n))
    K = 0.5 * (K + K.T)
    steps = int(np.ceil(T / dt))
    for step in range(steps):
        step_dt = min(dt, T - step * dt)
        if resample and step > 0:
            K = rng.standard_normal((n, n))
            K = 0.5 * (K + K.T)
        A = np.block([[zero, eye], [K, -gamma * eye]])
        Phi = expm(A * step_dt) @ Phi

    perm = np.zeros((n, n))
    perm[np.arange(n), (np.arange(n) + 1) % n] = 1.0
    shift = np.block([[perm, zero], [zero, perm]])
    M = shift @ Phi

    scale = np.max(np.abs(J))
    flow_err = float(np.max(np.abs(Phi.T @ J @ Phi - np.exp(-gamma * T) * J)) / scale)
    full_err = float(np.max(np.abs(M.T @ J @ M - np.exp(-gamma * T) * J)) / scale)
    return {
        "relerr": full_err,
        "relerr_without_shift": flow_err,
        "n": n,
        "gamma": gamma,
        "T": T,
        "dt": dt,
        "steps": steps,
        "seed": seed,
        "resample": resample,
    }


# ----------------------------------------------------- lattice monodromy ----
def sample_profile_at_sites(
    solver: TravelingWaveSolver, psi: np.ndarray, sites: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``phi``, ``phi'`` and ``phi''`` at arbitrary ``xi`` (here: the integer sites).

    The template part is closed-form; the periodic remainder ``psi`` is evaluated by
    trigonometric interpolation of its grid samples, which is spectrally accurate for the
    smooth, resolved ``psi`` the solver produces.  The Nyquist coefficient is dropped: it
    cannot be assigned a phase from real samples, and for a resolved profile it sits at the
    level of the truncation error anyway.

    ``phi''`` is needed because the translation mode of the lattice monodromy is
    ``z = (phi'(n), -c phi''(n))`` (Appendix A of the note).
    """
    grid = solver.grid
    xi = np.asarray(sites, dtype=float)
    coeff = np.fft.fft(psi) / grid.N
    coeff[grid.N // 2] = 0.0
    k = grid.k
    phase = np.exp(1j * np.outer(xi + grid.L, k))
    psi_at = np.real(phase @ coeff)
    dpsi_at = np.real(phase @ (1j * k * coeff))
    d2psi_at = np.real(phase @ (-(k ** 2) * coeff))

    width = solver.model.template_width
    arg = np.clip(xi / width, -600.0, 600.0)
    sech = 1.0 / np.cosh(arg)
    A = 2.0 * np.pi - 4.0 * np.arctan(np.exp(arg))
    dA = -(2.0 / width) * sech
    d2A = (2.0 / width ** 2) * sech * np.tanh(arg)
    return A + psi_at, dA + dpsi_at, d2A + d2psi_at


@dataclass
class LatticeMonodromy:
    """The one-period-plus-one-site map of the linearized lattice, as an operator."""

    sites: np.ndarray
    c: float
    T: float
    steps: int
    model: LatticeModel
    cos_table: np.ndarray  # V''(u) at the RK4 stage times, shape (steps, 3, S)
    drift_position: float
    drift_velocity: float
    phi_prime_sites: np.ndarray = field(repr=False, default=None)
    phi_second_sites: np.ndarray = field(repr=False, default=None)

    @property
    def dim(self) -> int:
        return 2 * self.sites.size

    def _coupling(self, x: np.ndarray) -> np.ndarray:
        """``sum_j C_j (x_{n+j} - 2 x_n + x_{n-j})`` with periodic (charge-free) wrap."""
        out = np.zeros_like(x)
        for j, coeff in self.model.couplings:
            out += coeff * (np.roll(x, -j) + np.roll(x, j) - 2.0 * x)
        return out

    def matvec(self, z: np.ndarray) -> np.ndarray:
        """Integrate the linearization over one period, then shift back one site."""
        S = self.sites.size
        z = np.asarray(z, dtype=float).reshape(2, S) if z.ndim == 1 else z
        xi_, eta = z[0].copy(), z[1].copy()
        dt = self.T / self.steps
        gamma = self.model.gamma

        def rhs(x, v, stage_cos):
            return v, self._coupling(x) - stage_cos * x - gamma * v

        for s in range(self.steps):
            c0, ch, c1 = self.cos_table[s]
            k1x, k1v = rhs(xi_, eta, c0)
            k2x, k2v = rhs(xi_ + 0.5 * dt * k1x, eta + 0.5 * dt * k1v, ch)
            k3x, k3v = rhs(xi_ + 0.5 * dt * k2x, eta + 0.5 * dt * k2v, ch)
            k4x, k4v = rhs(xi_ + dt * k3x, eta + dt * k3v, c1)
            xi_ = xi_ + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            eta = eta + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        # (M z)_n = z_{n+1}(T): the perturbation is genuinely periodic, so the shift is a
        # permutation -- unlike the profile itself, which carries the 2 pi charge.
        return np.concatenate([np.roll(xi_, -1), np.roll(eta, -1)])

    def as_operator(self) -> LinearOperator:
        return LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=float)

    def symplectic_residual(self, pairs: int = 8, seed: int = 0) -> float:
        """Test ``(Mx)^T J (My) = e^{-gamma T} x^T J y`` on random pairs.

        Equivalent to ``M^T J M = e^{-gamma T} J`` without forming ``M``.  Reported
        relative to ``|x^T J y|`` so it is a relative residual.
        """
        rng = np.random.default_rng(seed)
        S = self.sites.size
        worst = 0.0
        factor = np.exp(-self.model.gamma * self.T)
        for _ in range(pairs):
            x = rng.standard_normal(2 * S)
            y = rng.standard_normal(2 * S)
            Mx, My = self.matvec(x), self.matvec(y)
            lhs = float(Mx[:S] @ My[S:] - Mx[S:] @ My[:S])
            rhs = float(x[:S] @ y[S:] - x[S:] @ y[:S])
            worst = max(worst, abs(lhs - factor * rhs) / max(abs(rhs), 1e-300))
        return worst

    def translation_mode_residual(self) -> Dict[str, float]:
        """Test ``M z = z`` on the translation mode -- the decisive check that ``rho = 1``.

        Appendix A of the note identifies the eigenvector of the neutral multiplier as
        ``xi_n(t) = phi'(n - c t)``, i.e. ``z = (phi'(n), -c phi''(n))`` at ``t = 0``.  One
        matrix-vector product settles whether the lattice monodromy really carries it, which
        is more informative than many iterations of a Krylov method against a spectrum that
        clusters on ``|rho| = e^{-gamma/(2c)}``.
        """
        z = np.concatenate([self.phi_prime_sites, -self.c * self.phi_second_sites])
        Mz = self.matvec(z)
        rayleigh = float(z @ Mz / (z @ z))
        return {
            "residual_rel": float(np.linalg.norm(Mz - z) / np.linalg.norm(z)),
            "rayleigh": rayleigh,
            "rayleigh_minus_one": rayleigh - 1.0,
        }

    def dominant_multiplier(
        self, iters: int = 400, tol: float = 1e-9, seed: int = 0
    ) -> Dict[str, Any]:
        """Power iteration for the largest-modulus multiplier.

        All but a handful of the ``2 S`` multipliers sit on the circle
        ``|rho| = e^{-gamma/(2c)} = 0.945``, so the neutral multiplier (or, past the
        threshold, the unstable real one) is separated from the rest by a factor of about
        ``0.945``: slow for a restarted Krylov method, but serviceable for power iteration,
        which needs no restart strategy and whose convergence is transparent.
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(self.dim)
        z /= np.linalg.norm(z)
        lam = float("nan")
        prev = float("nan")
        converged = False
        for it in range(iters):
            Mz = self.matvec(z)
            prev, lam = lam, float(z @ Mz)
            nrm = float(np.linalg.norm(Mz))
            if nrm == 0.0:
                break
            z = Mz / nrm
            if it > 5 and abs(lam - prev) < tol * max(abs(lam), 1e-300):
                converged = True
                break
        Mz = self.matvec(z)
        return {
            "rho": lam,
            "residual_rel": float(np.linalg.norm(Mz - lam * z) / max(abs(lam), 1e-300)),
            "iterations": it + 1,
            "converged": converged,
        }

    def multipliers(
        self, k: int = 6, tol: float = 1e-8, maxiter: int = 3000, ncv: int | None = None
    ) -> np.ndarray:
        """The dominant Floquet multipliers, by Arnoldi on the matrix-free operator.

        ``which="LM"`` is the right request: stability is decided by the largest modulus.
        Convergence is slow and not guaranteed -- the circle of ``2 S`` multipliers sits only
        a factor ``0.945`` below the neutral one -- so callers must be prepared for
        ``ArpackNoConvergence`` and should prefer :meth:`translation_mode_residual` and
        :meth:`dominant_multiplier`, which answer the two questions that matter without a
        restart strategy.  The start vector is fixed, so the result is deterministic.
        """
        v0 = np.ones(self.dim) / np.sqrt(self.dim)
        vals, _ = eigs(
            self.as_operator(), k=k, which="LM", tol=tol, maxiter=maxiter, v0=v0, ncv=ncv
        )
        return vals[np.argsort(-np.abs(vals))]


def lattice_monodromy(
    solver: TravelingWaveSolver,
    psi: np.ndarray,
    drive: float,
    c: float,
    n_sites: int | None = None,
    steps: int = 4000,
) -> LatticeMonodromy:
    """Build the lattice monodromy of the traveling wave sampled at the integer sites.

    ``n_sites`` defaults to ``2L`` -- every integer site of the computational domain -- so
    that the periodic lattice carries exactly the charge ``2 pi`` of one kink.  The nonlinear
    trajectory is integrated once with RK4 and its ``V''(u)`` cached at the stage times, so
    every subsequent matrix-vector product is exactly linear and reuses the same reference
    orbit; recomputing the orbit per product would make the operator only approximately
    linear and quietly poison the Arnoldi run.

    ``drift_position`` and ``drift_velocity`` measure how well the sampled profile really is
    a lattice traveling wave: after one period the state should equal the initial one shifted
    by one site.  They are the numbers to read before trusting anything else here.
    """
    grid = solver.grid
    if n_sites is None:
        n_sites = int(round(2 * grid.L))
    sites = np.arange(n_sites) - n_sites // 2
    phi, dphi, d2phi = sample_profile_at_sites(solver, psi, sites.astype(float))
    T = 1.0 / c
    dt = T / steps
    model = solver.model
    charge = -2.0 * np.pi  # phi decreases by 2 pi from left to right

    def coupling(x: np.ndarray) -> np.ndarray:
        """``sum_j C_j (x_{n+j} - 2 x_n + x_{n-j})`` with the 2 pi charge across the seam."""
        out = np.zeros_like(x)
        for j, coeff in model.couplings:
            plus = np.roll(x, -j)
            plus[-j:] += charge
            minus = np.roll(x, j)
            minus[:j] -= charge
            out += coeff * (plus + minus - 2.0 * x)
        return out

    def rhs(u, v):
        return v, coupling(u) + drive - model.Vp(u) - model.gamma * v

    cos_table = np.empty((steps, 3, n_sites))
    u, v = phi.copy(), -c * dphi
    u0, v0 = u.copy(), v.copy()
    for s in range(steps):
        cos_table[s, 0] = model.Vpp(u)
        k1u, k1v = rhs(u, v)
        um = u + 0.5 * dt * k1u
        cos_table[s, 1] = model.Vpp(um)
        k2u, k2v = rhs(um, v + 0.5 * dt * k1v)
        k3u, k3v = rhs(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v)
        ue = u + dt * k3u
        cos_table[s, 2] = model.Vpp(ue)
        k4u, k4v = rhs(ue, v + dt * k3v)
        u = u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    # the interior drift: the seam sites see the template's truncation error, not the dynamics
    interior = slice(2, -2)
    drift_u = float(np.max(np.abs(np.roll(u, -1) - u0)[interior]))
    drift_v = float(np.max(np.abs(np.roll(v, -1) - v0)[interior]))
    return LatticeMonodromy(
        sites=sites,
        c=c,
        T=T,
        steps=steps,
        model=model,
        cos_table=cos_table,
        drift_position=drift_u,
        drift_velocity=drift_v,
        phi_prime_sites=dphi,
        phi_second_sites=d2phi,
    )
