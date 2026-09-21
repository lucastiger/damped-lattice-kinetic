"""Reference implementation (verbatim from the session): pseudo-arclength continuation of
the traveling-wave branch in (psi, sigma, c), with the biorthogonal scalars tracked through
turning points.  Used for the fold results (drivers r1, r4)."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from fk import FKSolver, spec_d1

class Branch:
    def __init__(self, L, N, mu=1.0, gamma=0.1, tw=2.0):
        self.S = FKSolver(L=L, N=N, mu=mu, gamma=gamma, tw=tw); self.mu = mu; self.gamma = gamma
        self.N = N; self.h = self.S.h; self.j0 = self.S.j0

    def F_and_J(self, psi, sigma, c):
        S = self.S; S.build(c); N = self.N
        R = S.residual(psi, sigma, c); g = psi[self.j0]
        M0 = S.jac_psi(psi); M1 = S.M1
        p0 = S.Ap + spec_d1(psi, S.k, N)
        J = np.zeros((N + 1, N + 2))
        J[:N, :N] = M0; J[:N, N] = -self.mu; J[:N, N + 1] = -(M1 @ p0)   # dR/dc = -M1 phi'
        J[N, self.j0] = 1.0
        return np.concatenate([R, [g]]), J, M0, M1, p0

    def tangent(self, J, tprev):
        N = self.N
        B = np.vstack([J, tprev.reshape(1, -1)])
        t = np.linalg.solve(B, np.concatenate([np.zeros(N + 1), [1.0]]))
        return t / np.linalg.norm(t)

    def step(self, X, tau, ds, tol=1e-10, maxit=12):
        Xp = X + ds * tau; Xn = Xp.copy(); N = self.N
        for _ in range(maxit):
            F, J, M0, M1, p0 = self.F_and_J(Xn[:N], Xn[N], Xn[N + 1])
            arc = tau @ (Xn - Xp)
            if max(np.max(np.abs(F)), abs(arc)) < tol: break
            B = np.vstack([J, tau.reshape(1, -1)])
            Xn = Xn + np.linalg.solve(B, -np.concatenate([F, [arc]]))
        F, J, M0, M1, p0 = self.F_and_J(Xn[:N], Xn[N], Xn[N + 1])
        return Xn, J, M0, M1, p0, np.max(np.abs(F))

    def scalars(self, M0, M1, p0):
        """kappa, <phat,1>, residual, with ||phat||_2 = 1 and <phat,phi'> > 0."""
        N, h = self.N, self.h
        lu = lu_factor(M0.T + 1e-14 * np.eye(N))
        v = np.random.default_rng(0).standard_normal(N); v /= np.linalg.norm(v)
        for _ in range(70): v = lu_solve(lu, v); v /= np.linalg.norm(v)
        if np.sum(v * p0) < 0: v = -v
        return np.sum(v * (M1 @ p0)) * h, np.sum(v) * h, np.linalg.norm(M0.T @ v)
