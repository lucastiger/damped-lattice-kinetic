"""Reference implementation of the biorthogonal / Jordan-chain quantities
(reconstructed from the investigation drivers t6/t9/t11/r4)."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigs
from fk import spec_d1


def phi_prime(S, psi):
    return S.Ap + spec_d1(psi, S.k, S.N)


def left_null(M0, seed=0, iters=80, shift=1e-14):
    """Left null vector of M0 (null vector of M0^T) by inverse iteration, ||.||_2 = 1."""
    N = M0.shape[0]
    lu = lu_factor(M0.T + shift * np.eye(N))
    v = np.random.default_rng(seed).standard_normal(N); v /= np.linalg.norm(v)
    for _ in range(iters):
        v = lu_solve(lu, v); v /= np.linalg.norm(v)
    return v


def scalars(S, psi, sigma, c, normalise="minus2pi"):
    """sigma'(c) exactly (bordered linearised branch equation), phat, kappa, <phat,1>,
    <phat,cos phi>, reduced coefficient m, nu2_pred = -kappa/m, power-balance sigma."""
    S.build(c)
    N, h, j0, mu, gamma = S.N, S.h, S.j0, S.mu, S.gamma
    phi = S.A + psi
    M0 = S.jac_psi(psi); M1 = S.M1
    p0 = phi_prime(S, psi)
    B = np.zeros((N + 1, N + 1)); B[:N, :N] = M0; B[:N, N] = -mu; B[N, j0] = 1.0
    sol = np.linalg.solve(B, np.concatenate([M1 @ p0, [0.0]]))
    dcpsi, sigp = sol[:N], sol[N]
    v = left_null(M0)
    if normalise == "minus2pi":
        v = v * (-2.0 * np.pi) / (np.sum(v) * h)          # <phat,1> = -2 pi
    else:                                                  # ||phat||_2 = 1, <phat,phi'> > 0
        if np.sum(v * p0) < 0: v = -v
    ph = v
    kappa = np.sum(ph * (M1 @ p0)) * h
    one = np.sum(ph) * h
    orth = np.sum(ph * np.cos(phi)) * h
    Bb = np.zeros((N + 1, N + 1)); Bb[:N, :N] = M0; Bb[:N, N] = ph; Bb[N, :N] = p0
    q1 = np.linalg.solve(Bb, np.concatenate([M1 @ p0, [0.0]]))[:N]
    m = np.sum(ph * p0) * h - np.sum(ph * (M1 @ q1)) * h
    return dict(sigma=sigma, sigp=sigp, kappa=kappa, one=one, orth=orth, m=m,
                nu2_pred=-kappa / m,
                ident_relerr=abs(kappa - (-mu * sigp * one)) / abs(mu * sigp * one),
                sigma_pb=gamma * c * np.sum(p0 ** 2) * h / (2 * np.pi * mu),
                res_M0p0=np.max(np.abs(M0 @ p0)) / np.max(np.abs(p0)),
                res_phat=np.linalg.norm(M0.T @ ph) / np.linalg.norm(ph),
                dcpsi_bdry=0.5 * (dcpsi[0] + dcpsi[-1]),
                phat=ph, p0=p0, M0=M0, M1=M1)


def pencil_eigs(M0, M1, shift, k=16, tol=1e-10):
    """Eigenvalues nu of Q(nu) = nu^2 I + nu M1 + M0 nearest 'shift' (shift-invert Arnoldi
    on the companion form; one N x N LU of Q(shift))."""
    N = M0.shape[0]; s = complex(shift)
    lu = lu_factor(M0 + s * M1 + s * s * np.eye(N)); Ms = M1 + s * np.eye(N)
    def op(b):
        b1, b2 = b[:N], b[N:]
        x1 = -lu_solve(lu, b2 + Ms @ b1)
        return np.concatenate([x1, b1 + s * x1])
    v0 = np.ones(2 * N, dtype=complex) / np.sqrt(2 * N)
    vals, _ = eigs(LinearOperator((2 * N, 2 * N), matvec=op, dtype=complex),
                   k=k, which='LM', tol=tol, maxiter=8000, v0=v0)
    return s + 1.0 / vals


def real_nontrivial(nus, gamma, imtol=1e-7):
    return sorted({round(x.real, 9) for x in nus
                   if abs(x.imag) < imtol and abs(x) > 1e-6 and abs(x + gamma) > 1e-6},
                  reverse=True)
