"""
Reference implementation (reconstructed verbatim from the investigation session).
Damped, dc-driven Frenkel-Kontorova lattice: traveling waves via Fourier-spectral
collocation of the advance-delay equation.

Model (Vainchtein, Cuevas-Maraver, Kevrekidis & Xu, CNSNS 85, 105236 (2020), eq. (1)):
    u_n'' + gamma u_n' = u_{n+1} - 2 u_n + u_{n-1} + mu (sigma - sin u_n)
Traveling wave u_n(t) = phi(xi), xi = n - c t:
    c^2 phi'' - gamma c phi' = phi(xi+1) - 2 phi(xi) + phi(xi-1) + mu(sigma - sin phi)
    phi -> arcsin(sigma)+2pi (xi->-inf), arcsin(sigma) (xi->+inf), pinned phi(0)=pi.
Unknown: periodic remainder psi = phi - A(xi) on [-L,L), plus sigma.
"""
import numpy as np
from scipy.linalg import circulant


def make_grid(L, N):
    h = 2.0 * L / N
    xi = -L + h * np.arange(N)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=h)
    j0 = N // 2
    assert abs(xi[j0]) < 1e-12
    return xi, k, h, j0


def circ_from_symbol(s, N):
    col = np.fft.ifft(s)
    assert np.max(np.abs(col.imag)) < 1e-10 * max(1.0, np.max(np.abs(col.real)))
    return circulant(col.real)


def sym_d1(k, N):
    s = 1j * k.copy()
    s[N // 2] = 0.0            # Nyquist mode removed for the odd symbol
    return s


def sym_lin(k, N, c, gamma):
    """symbol of c^2 d^2 - gamma c d - Delta_1,  Delta_1 f = f(.+1)+f(.-1)-2f"""
    return -(c ** 2) * k ** 2 - gamma * c * sym_d1(k, N) - (2.0 * np.cos(k) - 2.0)


def sym_M1(k, N, c, gamma):
    """symbol of M1 = -2c d/dxi + gamma"""
    return -2.0 * c * sym_d1(k, N) + gamma


def spec_d1(f, k, N):
    return np.real(np.fft.ifft(sym_d1(k, N) * np.fft.fft(f)))


def template(xi, w):
    """A = 2pi - 4 arctan(exp(xi/w)) (2pi -> 0); returns A, A', A'', Delta_1 A (closed form)."""
    def A_of(x):
        return 2.0 * np.pi - 4.0 * np.arctan(np.exp(np.clip(x / w, -600, 600)))
    A = A_of(xi)
    sech = 1.0 / np.cosh(np.clip(xi / w, -600, 600))
    Ap = -(2.0 / w) * sech
    App = (2.0 / w ** 2) * sech * np.tanh(xi / w)
    D1A = A_of(xi + 1.0) + A_of(xi - 1.0) - 2.0 * A
    return A, Ap, App, D1A


class FKSolver:
    def __init__(self, L, N, mu=1.0, gamma=0.1, tw=2.0):
        self.L, self.N, self.mu, self.gamma, self.tw = L, N, mu, gamma, tw
        self.xi, self.k, self.h, self.j0 = make_grid(L, N)

    def build(self, c):
        N, k = self.N, self.k
        self.Lin = circ_from_symbol(sym_lin(k, N, c, self.gamma), N)
        self.M1 = circ_from_symbol(sym_M1(k, N, c, self.gamma), N)
        self.A, self.Ap, self.App, self.D1A = template(self.xi, self.tw)
        self.c = c

    def residual(self, psi, sigma, c):
        mu, gamma = self.mu, self.gamma
        lin_A = c ** 2 * self.App - gamma * c * self.Ap - self.D1A
        return lin_A + self.Lin @ psi - mu * sigma + mu * np.sin(self.A + psi)

    def jac_psi(self, psi):
        return self.Lin + np.diag(self.mu * np.cos(self.A + psi))

    def solve(self, c, psi0, sigma0, tol=1e-12, maxit=40):
        """Damped Newton on (psi, sigma) at fixed c; pinning psi(0)=0 <=> phi(0)=pi."""
        self.build(c)
        N, mu, j0 = self.N, self.mu, self.j0
        psi, sigma = psi0.copy(), float(sigma0)
        for it in range(maxit):
            R = self.residual(psi, sigma, c)
            g = psi[j0]
            res = max(np.max(np.abs(R)), abs(g))
            if res < tol:
                return psi, sigma, res, it
            J = np.zeros((N + 1, N + 1))
            J[:N, :N] = self.jac_psi(psi); J[:N, N] = -mu; J[N, j0] = 1.0
            dx = np.linalg.solve(J, np.concatenate([-R, [-g]]))
            lam = 1.0
            for _ in range(20):
                psin, sign = psi + lam * dx[:N], sigma + lam * dx[N]
                Rn = self.residual(psin, sign, c)
                if max(np.max(np.abs(Rn)), abs(psin[j0])) < res or lam < 1e-3:
                    break
                lam *= 0.5
            psi, sigma = psin, sign
        R = self.residual(psi, sigma, c)
        return psi, sigma, max(np.max(np.abs(R)), abs(psi[j0])), maxit


def init_branch(L, N, c0=0.88, sigma_guess=0.6, tw=2.0, mu=1.0, gamma=0.1):
    """Initialisation used throughout the investigation: solve at c0 with the
    physical-width template w = sqrt((1-c0^2)/mu) from psi = 0, then transfer the
    profile to the fixed-width (w = tw) template and re-converge."""
    Sp = FKSolver(L=L, N=N, mu=mu, gamma=gamma, tw=np.sqrt((1 - c0 ** 2) / mu))
    o = Sp.solve(c0, np.zeros(N), sigma_guess)
    phi = Sp.A + o[0]
    S = FKSolver(L=L, N=N, mu=mu, gamma=gamma, tw=tw); S.build(c0)
    o2 = S.solve(c0, phi - S.A, o[1])
    return S, o2[0], o2[1]
