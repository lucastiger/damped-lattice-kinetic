"""Biorthogonal / Jordan-chain quantities and the co-traveling quadratic pencil.

Everything here is built on the linearization of (TW) at a converged traveling wave::

    M0 = c^2 d_xi^2 - gamma c d_xi - sum_j C_j Delta_j + V''(phi)     (SCIENCE_BRIEF sec. 2)
    M1 = -2 c d_xi + gamma
    Q(nu) = nu^2 I + nu M1 + M0                       (co-traveling quadratic pencil)

``phi'`` spans ``ker M0``; ``phat`` spans ``ker M0^T``, the kernel of the *anti-damped*
operator (``M0^T`` is ``M0`` with ``gamma -> -gamma``).  The transpose is taken with respect
to the bilinear pairing ``<f, g>_h = h sum_j f_j g_j`` -- no conjugation anywhere.

Two normalizations of ``phat`` are in use in the note and they give **different** values of
``kappa`` and ``<phat, 1>`` individually (though not of any ratio, and not of the identity
they satisfy).  :func:`normalize_phat` therefore takes ``mode`` as a required argument, and
every function that returns ``kappa`` also returns the mode it was computed under.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigs

from .solver import TravelingWaveSolver

__all__ = [
    "PHAT_MODES",
    "left_null",
    "normalize_phat",
    "exact_drive_derivative",
    "biorthogonal_scalars",
    "scalar_keys",
    "drop_arrays",
    "pencil_matrix",
    "pencil_eigs",
    "classify_eigs",
]

logger = logging.getLogger(__name__)

#: The two admissible normalizations of the left null vector.
PHAT_MODES = ("minus2pi", "unit")

#: Keys of :func:`biorthogonal_scalars` that are plain JSON-serializable scalars.
_SCALAR_KEYS = (
    "sigma",
    "sigma_prime",
    "drive",
    "drive_prime",
    "kappa",
    "phat_one",
    "phat_dot_Vpp",
    "m",
    "nu2_pred",
    "identity_relerr",
    "power_balance",
    "power_balance_relerr",
    "res_M0_phiprime",
    "res_phat_rel",
    "dcphi_edge",
    "dcphi_edge_pred",
    "phat_mode",
)


def scalar_keys() -> tuple[str, ...]:
    """The JSON-safe keys of a :func:`biorthogonal_scalars` record."""
    return _SCALAR_KEYS


def drop_arrays(record: Dict[str, Any]) -> Dict[str, Any]:
    """The scalar part of a :func:`biorthogonal_scalars` record, ready for JSON."""
    return {key: record[key] for key in _SCALAR_KEYS if key in record}


def left_null(
    M0: np.ndarray, seed: int = 0, iters: int = 80, shift: float = 1e-14
) -> np.ndarray:
    """Null vector of ``M0^T`` (the *left* null vector of ``M0``) by inverse iteration.

    One LU of ``M0^T + shift I`` is reused for every iteration.  The start is a Gaussian
    draw from ``numpy.random.default_rng(seed)``, so the result is deterministic: no
    process-global RNG is touched and no seed is left to chance.

    The returned vector has Euclidean norm 1 but **no orientation and no scaling
    convention applied** -- pass it through :func:`normalize_phat` before using it.
    """
    M0 = np.asarray(M0, dtype=float)
    n = M0.shape[0]
    if M0.shape != (n, n):
        raise ValueError(f"M0 must be square, got {M0.shape}.")
    lu = lu_factor(M0.T + shift * np.eye(n))
    v = np.random.default_rng(seed).standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = lu_solve(lu, v)
        v /= np.linalg.norm(v)
    return v


def normalize_phat(
    phat: np.ndarray, phi_prime: np.ndarray, grid, mode: str
) -> np.ndarray:
    """Apply one of the two ``phat`` normalizations.  ``mode`` is required, deliberately.

    ``mode="minus2pi"``
        Scale so that ``<phat, 1>_h = -2 pi``, matching ``<phi', 1>_h = -2 pi`` (the kink
        drops by ``2 pi``).  Under this convention the Jordan-chain identity reads
        ``kappa = 2 pi mu sigma'(c)``.  Used for the identity and threshold tables.
    ``mode="unit"``
        Euclidean norm 1 (of the grid vector, not the ``L^2`` function), with the sign
        fixed by ``<phat, phi'>_h > 0``.  Used for the fold and scope tables.

    The two give different ``kappa`` and different ``<phat, 1>``; only their relation and
    ratios such as ``-kappa/m`` are invariant.  Note also that ``"unit"`` depends on ``N``
    (a grid vector's Euclidean norm scales like ``sqrt(N)`` against a fixed function norm),
    which is why the reference ``kappa`` for the same physical state differs by ``sqrt(2)``
    between ``N = 2048`` and ``N = 4096``.
    """
    if mode not in PHAT_MODES:
        raise ValueError(f"phat mode must be one of {PHAT_MODES}, got {mode!r}.")
    v = np.asarray(phat, dtype=float)
    if mode == "minus2pi":
        denominator = float(np.sum(v) * grid.h)
        if denominator == 0.0:
            raise ValueError(
                "cannot normalize phat to <phat,1>_h = -2 pi: <phat,1>_h is exactly zero "
                "(this happens at a turning point in c -- use mode='unit' there)."
            )
        return v * (-2.0 * np.pi) / denominator
    v = v / np.linalg.norm(v)
    if float(np.sum(v * phi_prime)) < 0.0:
        v = -v
    return v


def exact_drive_derivative(
    solver: TravelingWaveSolver, psi: np.ndarray, drive: float, c: float
):
    """``(d_c psi, d(drive)/dc)`` from the bordered linearized branch equation.

    Differentiating (TW) along the branch gives ``M0 [d_c psi] - drive'(c) = M1 phi'``,
    closed by ``d_c psi(0) = 0`` (the pinning row, which holds along the whole branch)::

        [[ M0,     -1 ],  [ d_c psi ]   [ M1 phi' ]
         [ e_j0^T,  0 ]]  [ drive'  ] = [    0    ]

    For the Frenkel-Kontorova model ``sigma'(c) = drive'(c) / mu``.

    NEVER replace this with a finite difference in ``c``: a centred difference with step
    ``1e-3`` carries a grid-independent error of ``1.6e-4`` near the curvature maximum,
    which is four orders of magnitude above what the identity checks need.
    """
    solver.build(c)
    N, j0 = solver.grid.N, solver.grid.j0
    M0 = solver.M0(psi)
    phi_prime = solver.phi_prime(psi)

    B = np.zeros((N + 1, N + 1))
    B[:N, :N] = M0
    B[:N, N] = -1.0
    B[N, j0] = 1.0
    solution = np.linalg.solve(B, np.concatenate([solver.M1 @ phi_prime, [0.0]]))
    return solution[:N], float(solution[N])


def biorthogonal_scalars(
    solver: TravelingWaveSolver,
    psi: np.ndarray,
    drive: float,
    c: float,
    phat_mode: str,
    *,
    seed: int = 0,
    iters: int = 80,
    shift: float = 1e-14,
) -> Dict[str, Any]:
    """All the Jordan-chain scalars of SCIENCE_BRIEF sections 3 and 4 at one branch point.

    ``phat_mode`` is required and is echoed back in the record, because ``kappa`` and
    ``phat_one`` are meaningless without it.

    Returned scalars (see :func:`scalar_keys`)::

        sigma, drive              the driving force, as sigma = drive/mu and as drive
        sigma_prime, drive_prime  exact derivatives along the branch (bordered solve)
        kappa        = <phat, M1 phi'>_h                                          (R4)
        phat_one     = <phat, 1>_h
        phat_dot_Vpp = <phat, V''(phi)>_h        -- zero by Lemma 4.2, a free check
        m            = <phat, phi'>_h - <phat, M1 q1>_h                           (R7)
        nu2_pred     = -kappa / m
        identity_relerr = |kappa + drive' phat_one| / |drive' phat_one|           (R4)
        power_balance   = gamma c <phi', phi'>_h / (2 pi)  -- equals `drive`      (R6)
        power_balance_relerr = |drive - power_balance| / |drive|
        res_M0_phiprime = max|M0 phi'| / max|phi'|     (how well phi' is in ker M0)
        res_phat_rel    = ||M0^T phat||_2 / ||phat||_2
        dcphi_edge      = mean of d_c psi at the two domain edges
        dcphi_edge_pred = sigma' / sqrt(1 - sigma^2)   -- the far-field prediction

    Also returned, as arrays (strip them with :func:`drop_arrays` before writing JSON):
    ``M0``, ``M1``, ``phat``, ``phi_prime``, ``dc_psi``, ``q1``.
    """
    if phat_mode not in PHAT_MODES:
        raise ValueError(f"phat_mode must be one of {PHAT_MODES}, got {phat_mode!r}.")

    solver.build(c)
    grid, model = solver.grid, solver.model
    N, h, j0 = grid.N, grid.h, grid.j0
    mu, gamma = model.mu, model.gamma

    phi = solver.phi(psi)
    phi_prime = solver.phi_prime(psi)
    M0 = solver.M0(psi)
    M1 = solver.M1

    # exact derivative along the branch
    B = np.zeros((N + 1, N + 1))
    B[:N, :N] = M0
    B[:N, N] = -1.0
    B[N, j0] = 1.0
    solution = np.linalg.solve(B, np.concatenate([M1 @ phi_prime, [0.0]]))
    dc_psi, drive_prime = solution[:N], float(solution[N])

    # left null vector, under the requested convention
    phat = normalize_phat(left_null(M0, seed=seed, iters=iters, shift=shift),
                          phi_prime, grid, phat_mode)

    kappa = float(np.sum(phat * (M1 @ phi_prime)) * h)
    phat_one = float(np.sum(phat) * h)
    phat_dot_Vpp = float(np.sum(phat * model.Vpp(phi)) * h)

    # reduced coefficient m: [[M0, phat], [phi'^T, 0]] [q1; beta] = [M1 phi'; 0]
    Bb = np.zeros((N + 1, N + 1))
    Bb[:N, :N] = M0
    Bb[:N, N] = phat
    Bb[N, :N] = phi_prime
    q1 = np.linalg.solve(Bb, np.concatenate([M1 @ phi_prime, [0.0]]))[:N]
    m = float(np.sum(phat * phi_prime) * h - np.sum(phat * (M1 @ q1)) * h)

    sigma = drive / mu
    sigma_prime = drive_prime / mu
    power_balance = float(gamma * c * np.sum(phi_prime**2) * h / (2.0 * np.pi))

    record: Dict[str, Any] = {
        "sigma": sigma,
        "sigma_prime": sigma_prime,
        "drive": float(drive),
        "drive_prime": drive_prime,
        "kappa": kappa,
        "phat_one": phat_one,
        "phat_dot_Vpp": phat_dot_Vpp,
        "m": m,
        "nu2_pred": -kappa / m,
        "identity_relerr": abs(kappa + drive_prime * phat_one)
        / abs(drive_prime * phat_one),
        "power_balance": power_balance,
        "power_balance_relerr": abs(drive - power_balance) / abs(drive),
        "res_M0_phiprime": float(
            np.max(np.abs(M0 @ phi_prime)) / np.max(np.abs(phi_prime))
        ),
        "res_phat_rel": float(np.linalg.norm(M0.T @ phat) / np.linalg.norm(phat)),
        "dcphi_edge": float(0.5 * (dc_psi[0] + dc_psi[-1])),
        "dcphi_edge_pred": float(sigma_prime / np.sqrt(1.0 - sigma**2)),
        "phat_mode": phat_mode,
        # arrays, for downstream use
        "M0": M0,
        "M1": M1,
        "phat": phat,
        "phi_prime": phi_prime,
        "dc_psi": dc_psi,
        "q1": q1,
    }
    logger.debug(
        "scalars at c=%r (%s): sigma'=%.10f kappa=%.10f m=%.4f identity_relerr=%.2e",
        c,
        phat_mode,
        sigma_prime,
        kappa,
        m,
        record["identity_relerr"],
    )
    return record


def pencil_matrix(M0: np.ndarray, M1: np.ndarray, nu: complex) -> np.ndarray:
    """``Q(nu) = nu^2 I + nu M1 + M0``, the co-traveling quadratic pencil."""
    n = M0.shape[0]
    return nu * nu * np.eye(n) + nu * M1 + M0


def pencil_eigs(
    M0: np.ndarray,
    M1: np.ndarray,
    shift: complex,
    k: int = 16,
    tol: float = 1e-10,
    maxiter: int = 8000,
) -> np.ndarray:
    """Eigenvalues ``nu`` of ``Q(nu) p = 0`` nearest ``shift``, by shift-invert Arnoldi.

    The companion linearization of ``Q`` is ``[[0, I], [-M0, -M1]]``.  Shift-inverting it
    needs only ONE ``N x N`` LU, of ``Q(s) = M0 + s M1 + s^2 I``::

        x1 = -Q(s)^{-1} (b2 + (M1 + s I) b1)
        x2 = b1 + s x1

    The Arnoldi start vector is fixed (``ones / sqrt(2N)``), so repeated calls on the same
    matrices return the same eigenvalues in the same order.

    Returns ``nu = shift + 1/val`` for the ARPACK eigenvalues ``val``.
    """
    N = M0.shape[0]
    s = complex(shift)
    lu = lu_factor(M0 + s * M1 + s * s * np.eye(N))
    shifted_M1 = M1 + s * np.eye(N)

    def matvec(b: np.ndarray) -> np.ndarray:
        b1, b2 = b[:N], b[N:]
        x1 = -lu_solve(lu, b2 + shifted_M1 @ b1)
        return np.concatenate([x1, b1 + s * x1])

    v0 = np.ones(2 * N, dtype=complex) / np.sqrt(2 * N)
    vals, _ = eigs(
        LinearOperator((2 * N, 2 * N), matvec=matvec, dtype=complex),
        k=k,
        which="LM",
        tol=tol,
        maxiter=maxiter,
        v0=v0,
    )
    return s + 1.0 / vals


def classify_eigs(
    nus: Iterable[complex], gamma: float, im_tol: float = 1e-7
) -> Dict[str, Any]:
    """Sort a pencil spectrum into the pieces the note talks about.

    ``real_nontrivial``
        Real parts of the eigenvalues with ``|Im nu| < im_tol``, with the translation mode
        ``nu = 0`` and its conformal partner ``nu = -gamma`` removed, deduplicated by
        rounding to 9 decimals and sorted descending.  A positive entry means an unstable
        traveling wave (``|rho| = |e^{nu/c}| > 1``).
    ``has_zero`` / ``has_minus_gamma``
        Whether ``nu = 0`` and ``nu = -gamma`` were found (they must both be in the
        spectrum: the first by translation invariance, the second by the duality R2).
    ``pairing_max_err``
        Numerical test of the duality ``Q(-gamma-nu) = Q(nu)^T``: for each ``nu`` in the
        set, the distance from its partner ``-gamma-nu`` to the nearest member of the set,
        maximised over the set.  As in the reference, the minimisation ranges over the
        whole set including ``nu`` itself, which matters only for a self-paired
        ``nu = -gamma/2``.
    """
    nus = np.asarray(list(nus), dtype=complex)
    real_nontrivial: List[float] = sorted(
        {
            round(float(x.real), 9)
            for x in nus
            if abs(x.imag) < im_tol and abs(x) > 1e-6 and abs(x + gamma) > 1e-6
        },
        reverse=True,
    )
    if nus.size:
        pairing_max_err = float(
            max(float(np.min(np.abs(x + gamma + nus))) for x in nus)
        )
    else:
        pairing_max_err = float("nan")
    return {
        "real_nontrivial": real_nontrivial,
        "has_zero": bool(nus.size and np.min(np.abs(nus)) < 1e-6),
        "has_minus_gamma": bool(nus.size and np.min(np.abs(nus + gamma)) < 1e-6),
        "pairing_max_err": pairing_max_err,
    }
