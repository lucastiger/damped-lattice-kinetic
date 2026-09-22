"""Biorthogonal and spectral layer: ``phat``, the exact ``sigma'(c)``, ``kappa``, ``m``
and the eigenvalues of the co-traveling quadratic pencil.

This is the numerical content of Sections 4-7 of ``handoff/SCIENCE_BRIEF.md``, and a
restructured -- but numerically identical -- form of ``handoff/reference/spectral.py``
and the ``scalars`` method of ``handoff/reference/arc.py``.

Definitions, all with the **bilinear** discrete pairing ``<f,g>_h = h * sum_j f_j g_j``
(no conjugation)::

    M0 = c^2 d_xi^2 - gamma c d_xi - sum_j C_j Delta_j + V''(phi)   ker M0 = span{phi'}
    M1 = -2 c d_xi + gamma                                          ker M0^T = span{phat}
    Q(nu) = nu^2 + nu M1 + M0                                       (the pencil)

    kappa(c) = <phat, M1 phi'>_h
    m(c)     = <phat, phi'>_h - <phat, M1 q1>_h,   M0 q1 = (I - Pi) M1 phi', <phi',q1> = 0
    nu2_pred = -kappa / m

The Jordan-chain identity (R4) that the whole note turns on is ``kappa = -f'(c) <phat,1>``
with ``f = mu sigma`` the additive drive; with the ``<phat,1>_h = -2 pi`` normalization
that reads ``kappa = 2 pi mu sigma'(c)``.  :attr:`BranchScalars.identity_relerr` is the
relative error of that identity and is *never* imposed anywhere in the computation.

Normalization of ``phat``
-------------------------
``phat`` spans a one-dimensional space, so a convention is needed.  Two are in use and
**both appear in the note**:

``"minus2pi"``
    ``<phat,1>_h = -2 pi``, matching ``<phi',1>_h = -2 pi``.  Used for the identity and
    threshold tables, where ``kappa`` is quoted against ``2 pi mu sigma'``.
``"unit"``
    ``||phat||_2 = 1`` (Euclidean norm of the grid vector), oriented by
    ``<phat, phi'>_h > 0``.  Used for the fold, where ``<phat,1>_h`` passes through zero
    and so cannot be normalized to ``-2 pi``.

Which one is in force changes ``kappa``, ``m`` and ``<phat,1>`` individually but not
their relation or any ratio.  Every record must say which was used -- hence
:attr:`BranchScalars.normalization`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigs

from .solver import TravelingWaveSolver

__all__ = [
    "PHAT_NORMALIZATIONS",
    "left_null",
    "normalize_phat",
    "KappaScalars",
    "kappa_scalars",
    "BranchScalars",
    "branch_scalars",
    "pencil_eigs",
    "real_nontrivial",
    "positive_real_union",
    "EigClassification",
    "classify_eigs",
]

#: The two admissible normalizations of ``phat`` (see the module docstring).
PHAT_NORMALIZATIONS = ("minus2pi", "unit")


# --------------------------------------------------------------------- phat ---
def left_null(
    M0: np.ndarray, seed: int = 0, iters: int = 80, shift: float = 1e-14
) -> np.ndarray:
    """Left null vector of ``M0`` (i.e. the null vector of ``M0^T``), ``||.||_2 = 1``.

    Inverse iteration on ``M0^T + shift * I``.  ``M0`` is singular to working precision
    (its smallest singular value is ~3e-9 at ``N = 2048``), so the tiny shift is what
    makes the LU factorization usable rather than a coin toss on the pivot; the iteration
    itself converges in a handful of steps and the remaining ``iters`` are free insurance.

    The start vector is drawn from a **seeded** generator, so the result is bit-for-bit
    reproducible: nothing downstream depends on process state.  The sign of the returned
    vector is whatever inverse iteration lands on -- fix it with :func:`normalize_phat`.
    """
    N = M0.shape[0]
    if M0.shape != (N, N):
        raise ValueError(f"M0 must be square, got shape {M0.shape}.")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}.")
    lu = lu_factor(M0.T + shift * np.eye(N))
    v = np.random.default_rng(seed).standard_normal(N)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = lu_solve(lu, v)
        v /= np.linalg.norm(v)
    return v


def normalize_phat(
    v: np.ndarray,
    h: float,
    normalization: str = "minus2pi",
    phi_prime: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one of :data:`PHAT_NORMALIZATIONS` to a left null vector.

    ``"minus2pi"`` rescales so that ``<phat,1>_h = -2 pi``; it is undefined where
    ``<phat,1>_h`` vanishes, which is exactly the fold -- use ``"unit"`` there.
    ``"unit"`` normalizes in the Euclidean norm and orients by ``<phat, phi'>_h > 0``,
    which requires ``phi_prime``.
    """
    v = np.asarray(v, dtype=float)
    if normalization == "minus2pi":
        one = float(np.sum(v) * h)
        if one == 0.0:
            raise ValueError(
                "cannot impose <phat,1>_h = -2 pi: the left null vector has <phat,1>_h = 0 "
                "(this is the fold condition). Use normalization='unit' there."
            )
        return v * (-2.0 * np.pi) / one
    if normalization == "unit":
        if phi_prime is None:
            raise ValueError(
                "normalization='unit' orients phat by <phat, phi'>_h > 0 and therefore "
                "needs phi_prime."
            )
        v = v / np.linalg.norm(v)
        return -v if float(np.sum(v * phi_prime)) < 0.0 else v
    raise ValueError(
        f"unknown phat normalization {normalization!r}; expected one of {PHAT_NORMALIZATIONS}."
    )


@dataclass(frozen=True)
class KappaScalars:
    """``phat`` and the two pairings that the fold dichotomy (R5) is stated in."""

    phat: np.ndarray = field(repr=False)
    kappa: float
    phat_one: float
    res_phat_rel: float
    normalization: str


def kappa_scalars(
    M0: np.ndarray,
    M1: np.ndarray,
    phi_prime: np.ndarray,
    h: float,
    normalization: str = "minus2pi",
    seed: int = 0,
    iters: int = 80,
    shift: float = 1e-14,
) -> KappaScalars:
    """``kappa = <phat, M1 phi'>_h`` and ``<phat,1>_h`` from the assembled operators.

    This is the cheap path used inside arclength continuation, where ``M0``, ``M1`` and
    ``phi'`` are already in hand from the extended Jacobian and only the pairings are
    wanted.  :func:`branch_scalars` is the full version.
    """
    v = left_null(M0, seed=seed, iters=iters, shift=shift)
    phat = normalize_phat(v, h, normalization, phi_prime)
    return KappaScalars(
        phat=phat,
        kappa=float(np.sum(phat * (M1 @ phi_prime)) * h),
        phat_one=float(np.sum(phat) * h),
        res_phat_rel=float(np.linalg.norm(M0.T @ phat) / np.linalg.norm(phat)),
        normalization=normalization,
    )


# ------------------------------------------------------------ branch scalars ---
@dataclass(frozen=True)
class BranchScalars:
    """Everything the identity, threshold and fold tables are built from, at one point.

    ``drive = mu * sigma`` is the additive drive; ``drive_prime = mu * sigma'(c)``.  The
    ``sigma`` readings are the Frenkel-Kontorova ones and divide by ``mu``.
    """

    c: float
    drive: float
    drive_prime: float
    mu: float
    normalization: str
    # null vectors and pairings
    phat: np.ndarray = field(repr=False)
    phi_prime: np.ndarray = field(repr=False)
    kappa: float
    phat_one: float
    phat_dot_Vpp: float
    m: float
    # branch derivative
    dc_psi: np.ndarray = field(repr=False)
    dcphi_edge: float
    dcphi_edge_pred: float
    # free checks
    drive_power_balance: float
    res_M0_phi_prime: float
    res_phat_rel: float
    # operators, kept so that callers need not reassemble them for the pencil
    M0: np.ndarray = field(repr=False)
    M1: np.ndarray = field(repr=False)

    @property
    def sigma(self) -> float:
        """``sigma = drive / mu``."""
        return self.drive / self.mu

    @property
    def sigma_prime(self) -> float:
        """The exact ``sigma'(c)`` from the bordered solve -- never a finite difference."""
        return self.drive_prime / self.mu

    @property
    def sigma_power_balance(self) -> float:
        """``sigma`` as read off the power balance (R6), ``mu sigma = gamma c <phi',phi'>/2 pi``."""
        return self.drive_power_balance / self.mu

    @property
    def nu2_pred(self) -> float:
        """Leading-order reduction (R7) of the non-trivial eigenvalue, ``-kappa/m``."""
        return -self.kappa / self.m

    @property
    def identity_relerr(self) -> float:
        """Relative error of the Jordan-chain identity ``kappa = -f'(c) <phat,1>`` (R4)."""
        predicted = -self.drive_prime * self.phat_one
        return float(abs(self.kappa - predicted) / abs(predicted))

    @property
    def power_balance_relerr(self) -> float:
        """Relative error of the power balance (R6); nowhere imposed, hence a free check."""
        return float(abs(self.drive - self.drive_power_balance) / abs(self.drive))


def branch_scalars(
    solver: TravelingWaveSolver,
    psi: np.ndarray,
    drive: float,
    c: float,
    normalization: str = "minus2pi",
    phat_seed: int = 0,
    phat_iters: int = 80,
    phat_shift: float = 1e-14,
) -> BranchScalars:
    """All biorthogonal scalars at one converged point ``(psi, drive)`` of the branch.

    Three dense solves, in this order:

    1. the **branch derivative**, from ``d/dc`` of the collocated (TW) residual::

           [[ M0, -1 ], [ e_j0^T, 0 ]] [d_c psi ; f'(c)] = [M1 phi' ; 0]

       (``dR/dc = -M1 phi'`` at fixed ``phi``, and the pinning row is ``c``-independent).
       This is the exact derivative; a centred difference with step ``1e-3`` is wrong by
       a grid-independent ``1.6e-4`` near the curvature maximum -- see
       ``scripts/02_identity_convergence.py``.
    2. ``phat`` by inverse iteration (:func:`left_null`), then the pairings.
    3. the **reduced coefficient** ``m`` of R7, from the bordered system
       ``[[M0, phat], [phi'^T, 0]] [q1 ; beta] = [M1 phi' ; 0]``, whose last row imposes
       ``<phi', q1> = 0`` and whose bordering column projects the right-hand side off
       ``phat``, i.e. solves ``M0 q1 = (I - Pi) M1 phi'``.

    ``psi`` and ``drive`` must be a converged solve at ``c``: nothing here re-solves.
    """
    c = float(c)
    drive = float(drive)
    solver.build(c)
    grid, model = solver.grid, solver.model
    N, j0, h = grid.N, grid.j0, grid.h

    psi = np.asarray(psi, dtype=float)
    if psi.shape != (N,):
        raise ValueError(f"psi must have shape ({N},), got {psi.shape}.")

    phi = solver.phi(psi)
    M0 = solver.M0(psi)
    M1 = solver.M1
    p0 = solver.phi_prime(psi)
    rhs = M1 @ p0

    # 1. exact branch derivative (d_c psi, f'(c))
    B = np.zeros((N + 1, N + 1))
    B[:N, :N] = M0
    B[:N, N] = -1.0
    B[N, j0] = 1.0
    sol = np.linalg.solve(B, np.concatenate([rhs, [0.0]]))
    dc_psi, drive_prime = sol[:N], float(sol[N])

    # 2. phat and the pairings
    ks = kappa_scalars(
        M0,
        M1,
        p0,
        h,
        normalization=normalization,
        seed=phat_seed,
        iters=phat_iters,
        shift=phat_shift,
    )
    phat = ks.phat

    # 3. the reduced coefficient m
    Bb = np.zeros((N + 1, N + 1))
    Bb[:N, :N] = M0
    Bb[:N, N] = phat
    Bb[N, :N] = p0
    q1 = np.linalg.solve(Bb, np.concatenate([rhs, [0.0]]))[:N]
    m = float(np.sum(phat * p0) * h - np.sum(phat * (M1 @ q1)) * h)

    # the far-field value of phi, where d_c phi is read off: V'(phi_inf) = f, so
    # d_c phi_inf = f'(c) / V''(phi_inf).  For the FK substrate phi_inf = arcsin(sigma)
    # exactly and this is the closed form sigma' / sqrt(1 - sigma^2).
    sigma = drive / model.mu
    phi_far = float(np.arcsin(sigma)) if model.mu2 == 0.0 else float(phi[-1])

    return BranchScalars(
        c=c,
        drive=drive,
        drive_prime=drive_prime,
        mu=model.mu,
        normalization=normalization,
        phat=phat,
        phi_prime=p0,
        kappa=ks.kappa,
        phat_one=ks.phat_one,
        phat_dot_Vpp=float(np.sum(phat * model.Vpp(phi)) * h),
        m=m,
        dc_psi=dc_psi,
        dcphi_edge=float(0.5 * (dc_psi[0] + dc_psi[-1])),
        dcphi_edge_pred=float(drive_prime / model.Vpp(phi_far)),
        drive_power_balance=float(model.gamma * c * np.sum(p0**2) * h / (2.0 * np.pi)),
        res_M0_phi_prime=float(np.max(np.abs(M0 @ p0)) / np.max(np.abs(p0))),
        res_phat_rel=ks.res_phat_rel,
        M0=M0,
        M1=M1,
    )


# ------------------------------------------------------------------- pencil ---
def pencil_eigs(
    M0: np.ndarray,
    M1: np.ndarray,
    shift: complex,
    k: int = 16,
    tol: float = 1e-10,
    maxiter: int = 8000,
) -> np.ndarray:
    """Eigenvalues ``nu`` of ``Q(nu) = nu^2 I + nu M1 + M0`` nearest ``shift``.

    Shift-invert Arnoldi (ARPACK) on the companion linearization
    ``[[0, I], [-M0, -M1]]``.  The shift-invert solve is done without ever forming the
    ``2N x 2N`` matrix: one ``N x N`` LU of ``Q(s) = M0 + s M1 + s^2 I`` gives

        x1 = -Q(s)^{-1} (b2 + (M1 + s I) b1),      x2 = b1 + s x1

    and ``nu = s + 1 / val`` recovers the eigenvalue from the shift-inverted one.  The
    start vector ``v0`` is fixed (all ones, normalized), so the run is deterministic:
    ARPACK's default would seed from the global RNG.
    """
    N = M0.shape[0]
    s = complex(shift)
    lu = lu_factor(M0 + s * M1 + s * s * np.eye(N))
    Ms = M1 + s * np.eye(N)

    def op(b: np.ndarray) -> np.ndarray:
        b1, b2 = b[:N], b[N:]
        x1 = -lu_solve(lu, b2 + Ms @ b1)
        return np.concatenate([x1, b1 + s * x1])

    v0 = np.ones(2 * N, dtype=complex) / np.sqrt(2 * N)
    vals, _ = eigs(
        LinearOperator((2 * N, 2 * N), matvec=op, dtype=complex),
        k=k,
        which="LM",
        tol=tol,
        maxiter=maxiter,
        v0=v0,
    )
    return s + 1.0 / vals


def real_nontrivial(
    nus: Iterable[complex],
    gamma: float,
    imtol: float = 1e-7,
    zero_tol: float = 1e-6,
    decimals: int = 9,
) -> List[float]:
    """The real eigenvalues of the pencil, minus the two that are always there.

    Discarded: ``|Im nu| >= imtol`` (not real), ``|nu| < zero_tol`` (the translation mode
    ``phi'``, which is in ``ker M0`` by construction) and ``|nu + gamma| < zero_tol`` (its
    conformal partner, forced by the duality ``Q(-gamma-nu) = Q(nu)^T`` of R2).  Values
    are rounded to ``decimals`` before de-duplication, since ARPACK returns each
    eigenvalue once per converged Ritz pair.  Returned in decreasing order.
    """
    return sorted(
        {
            round(float(np.real(x)), decimals)
            for x in nus
            if abs(np.imag(x)) < imtol
            and abs(x) > zero_tol
            and abs(x + gamma) > zero_tol
        },
        reverse=True,
    )


def positive_real_union(
    M0: np.ndarray,
    M1: np.ndarray,
    shifts: Sequence[float],
    gamma: float,
    k: int = 16,
    tol: float = 1e-10,
    imtol: float = 1e-7,
    decimals: int = 8,
) -> List[float]:
    """Positive real non-trivial eigenvalues found from several shifts, deduplicated.

    Each shift resolves the eigenvalues in its own neighbourhood, so the unstable ones are
    collected by taking the union over a few shifts and rounding to ``decimals`` to merge
    the duplicates that two shifts both find.  Returned in decreasing order, so entry 0 is
    the leading unstable eigenvalue and the length is the number of unstable real modes.
    """
    found: List[float] = []
    for s in shifts:
        nus = pencil_eigs(M0, M1, s, k=k, tol=tol)
        found += [x for x in real_nontrivial(nus, gamma, imtol=imtol) if x > 0]
    return sorted({round(float(x), decimals) for x in found}, reverse=True)


@dataclass(frozen=True)
class EigClassification:
    """What a computed piece of pencil spectrum contains, sorted into its structural parts.

    ``nu = 0`` and ``nu = -gamma`` are there by construction -- the first because
    ``phi' in ker M0``, the second because R2 forces the spectrum to be invariant under
    ``nu -> -gamma - nu`` -- so finding them, and finding the whole set closed under that
    involution, is a check on the computation rather than a result.  What is a result is
    how many *other* real eigenvalues there are: none means no real multiplier has left the
    unit circle.
    """

    nus: np.ndarray = field(repr=False)
    gamma: float
    nu_zero_abs: float
    nu_minus_gamma: complex
    pairing_max_err: float
    real_nontrivial: List[float]

    @property
    def n_real_nontrivial(self) -> int:
        """How many real eigenvalues survive after the two structural ones are removed."""
        return len(self.real_nontrivial)

    def rho(self, c: float) -> np.ndarray:
        """Floquet multipliers ``rho = exp(nu / c)`` of the monodromy."""
        return np.exp(np.asarray(self.nus) / float(c))


def classify_eigs(
    nus: Iterable[complex],
    gamma: float,
    imtol: float = 1e-7,
    zero_tol: float = 1e-6,
    decimals: int = 9,
) -> EigClassification:
    """Sort a computed piece of pencil spectrum into its structural and non-trivial parts.

    ``pairing_max_err`` is the quantitative form of R2: for every computed ``nu`` it asks
    how close ``-gamma - nu`` comes to some other computed eigenvalue, and reports the
    worst case.  A shift-invert window is not closed under the involution at its edge --
    the partner of an eigenvalue near the rim can lie outside the ``k`` that converged --
    so this is a statement about the window as computed, and it is only small when the
    window happens to be (nearly) self-dual, which the ``-gamma/2``-centred shift makes it.
    """
    nus = np.asarray(list(nus))
    if nus.size == 0:
        raise ValueError("classify_eigs needs at least one eigenvalue.")
    partner_err = np.abs(nus[:, None] + gamma + nus[None, :]).min(axis=1)
    return EigClassification(
        nus=nus,
        gamma=float(gamma),
        nu_zero_abs=float(np.min(np.abs(nus))),
        nu_minus_gamma=complex(nus[int(np.argmin(np.abs(nus + gamma)))]),
        pairing_max_err=float(np.max(partner_err)),
        real_nontrivial=real_nontrivial(
            nus, gamma, imtol=imtol, zero_tol=zero_tol, decimals=decimals
        ),
    )
