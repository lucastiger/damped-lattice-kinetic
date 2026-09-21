"""Reference implementation (verbatim from the session): generalised damped-driven lattice
   u_n'' + gamma u_n' = C1*Delta_1 u + C2*Delta_2 u + f - V'(u_n),  V'(u) = mu sin u + mu2 sin 2u
used for the scope test (driver r3).  Identity tested: kappa = -f'(c) <phat,1>."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from fk import make_grid, circ_from_symbol, sym_d1, template

class Gen:
    def __init__(self, L, N, gamma=0.1, mu=1.0, mu2=0.0, C1=1.0, C2=0.0, tw=2.0):
        self.__dict__.update(L=L, N=N, gamma=gamma, mu=mu, mu2=mu2, C1=C1, C2=C2, tw=tw)
        self.xi, self.k, self.h, self.j0 = make_grid(L, N)
    def build(self, c):
        N, k = self.N, self.k
        s = -(c**2)*k**2 - self.gamma*c*sym_d1(k, N) - self.C1*(2*np.cos(k)-2) - self.C2*(2*np.cos(2*k)-2)
        self.Lin = circ_from_symbol(s, N)
        self.M1 = circ_from_symbol(-2.0*c*sym_d1(k, N) + self.gamma, N)
        A, Ap, App, _ = template(self.xi, self.tw)
        def Aof(x): return 2*np.pi - 4*np.arctan(np.exp(np.clip(x/self.tw, -600, 600)))
        self.A, self.Ap, self.App = A, Ap, App
        self.DA = self.C1*(Aof(self.xi+1)+Aof(self.xi-1)-2*A) + self.C2*(Aof(self.xi+2)+Aof(self.xi-2)-2*A)
        self.c = c
    def Vp(self, p): return self.mu*np.sin(p) + self.mu2*np.sin(2*p)
    def Vpp(self, p): return self.mu*np.cos(p) + 2*self.mu2*np.cos(2*p)
    def res(self, psi, f, c):
        return (c**2*self.App - self.gamma*c*self.Ap - self.DA) + self.Lin@psi - f + self.Vp(self.A+psi)
    def M0(self, psi): return self.Lin + np.diag(self.Vpp(self.A+psi))
    def solve(self, c, psi, f, tol=1e-11):
        N = self.N
        for _ in range(40):
            self.build(c); R = self.res(psi, f, c); g = psi[self.j0]
            if max(np.max(np.abs(R)), abs(g)) < tol: break
            J = np.zeros((N+1, N+1)); J[:N, :N] = self.M0(psi); J[:N, N] = -1.0; J[N, self.j0] = 1.0
            d = np.linalg.solve(J, -np.concatenate([R, [g]]))
            lam = 1.0
            for _ in range(15):
                pn, fn = psi+lam*d[:N], f+lam*d[N]
                if max(np.max(np.abs(self.res(pn, fn, c))), abs(pn[self.j0])) < max(np.max(np.abs(R)), abs(g)) or lam < 1e-3: break
                lam *= 0.5
            psi, f = pn, fn
        self.build(c)
        return psi, f, max(np.max(np.abs(self.res(psi, f, c))), abs(psi[self.j0]))
    def scalars(self, psi, c):
        N, h = self.N, self.h; self.build(c)
        M0 = self.M0(psi); M1 = self.M1
        p0 = self.Ap + np.real(np.fft.ifft(sym_d1(self.k, N)*np.fft.fft(psi)))
        B = np.zeros((N+1, N+1)); B[:N, :N] = M0; B[:N, N] = -1.0; B[N, self.j0] = 1.0
        fp = np.linalg.solve(B, np.concatenate([M1@p0, [0.0]]))[N]
        lu = lu_factor(M0.T + 1e-14*np.eye(N)); v = np.random.default_rng(0).standard_normal(N); v /= np.linalg.norm(v)
        for _ in range(80): v = lu_solve(lu, v); v /= np.linalg.norm(v)
        if np.sum(v*p0) < 0: v = -v
        return fp, np.sum(v*(M1@p0))*h, np.sum(v)*h
