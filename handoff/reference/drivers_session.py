"""
Drivers reconstructed from the original investigation session.

These reproduce, function by function, the computations behind each table of the
research note.  fk.py / spectral.py / arc.py / gen.py are verbatim; the drivers were
consolidated here from the session's throwaway scripts.  The subset re-run after a
sandbox reset (verify_core.py, verify_extra.py) matched the session values exactly;
the remaining values are transcribed in ../claims.yaml.

Run:  python drivers_session.py <name>     name in {convergence, threshold, fold,
                                           fold_invariant, scope, decay, conformal,
                                           mscan, spectrum, all}
"""
import sys, json
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.linalg import lu_factor, lu_solve, expm
from fk import FKSolver, init_branch, spec_d1
from spectral import scalars, pencil_eigs, real_nontrivial, left_null
from arc import Branch
from gen import Gen

MU, GAM = 1.0, 0.1


def _march(S, psi, sig, cs):
    for c in cs:
        psi, sig, r, _ = S.solve(round(float(c), 6), psi, sig)
        assert r < 1e-9, (c, r)
    return psi, sig


# --------------------------------------------------------------- Table 3 ---
def convergence():
    """Relative error of kappa = -mu sigma' <phat,1> at c=0.89 vs (L,N)."""
    out = {}
    for (L, N) in [(200, 1024), (200, 2048), (200, 4096), (300, 3072), (300, 6144), (150, 2048)]:
        S, psi, sig = init_branch(float(L), N)
        psi, sig = _march(S, psi, sig, np.arange(0.881, 0.8901, 0.001))
        d = scalars(S, psi, sig, 0.89)
        out[f"L{L}_N{N}"] = dict(relerr=d['ident_relerr'], res_M0p0=d['res_M0p0'])
        print(f"L={L} N={N} h={2*L/N:.4f} relerr={d['ident_relerr']:.2e} "
              f"res={d['res_M0p0']:.1e}", flush=True)
    return out


# --------------------------------------------------------------- Table 4 ---
THRESHOLD_CS = [0.89500, 0.89800, 0.89880, 0.89892, 0.89900, 0.89920, 0.89950, 0.89980, 0.90000]

def threshold(L=200.0, N=4096):
    S, psi, sig = init_branch(L, N)
    psi, sig = _march(S, psi, sig, np.arange(0.881, 0.8951, 0.001))
    rows = {}
    for c in THRESHOLD_CS:
        psi, sig, r, _ = S.solve(c, psi, sig); assert r < 1e-9
        d = scalars(S, psi, sig, c)
        shift = max(d['nu2_pred'], 0.0) + 0.004
        cand = [x for x in real_nontrivial(pencil_eigs(d['M0'], d['M1'], shift, k=24), GAM)
                if abs(x - d['nu2_pred']) < 0.05]
        nu2 = cand[0] if cand else None
        rows[f"{c:.5f}"] = dict(sigma=sig, sigma_prime=d['sigp'], kappa=d['kappa'], m=d['m'],
                                nu2_pred=d['nu2_pred'], nu2=nu2,
                                rho2=(np.exp(nu2 / c) if nu2 is not None else None))
        print(f"c={c:.5f} sp={d['sigp']:+.7f} kappa={d['kappa']:+.7f} m={d['m']:9.2f} "
              f"pred={d['nu2_pred']:+.6f} nu2={nu2}", flush=True)
    return rows


# --------------------------------------------------------------- Table 5 ---
def fold(L=200.0, N=2048, ds=0.03, nsteps=120):
    """Pseudo-arclength through c_max, with kappa, <phat,1> and the unstable eigenvalue."""
    B = Branch(L=L, N=N, gamma=GAM)
    S, psi, sig = init_branch(L, N)
    B.S = S
    psi, sig = _march(S, psi, sig, list(np.arange(0.885, 0.8981, 0.005)) + [0.8990, 0.8995, 0.8998])
    X = np.concatenate([psi, [sig], [0.8998]])
    F, J, M0, M1, p0 = B.F_and_J(X[:N], X[N], X[N + 1])
    t = np.zeros(N + 2); t[N + 1] = 1.0
    tau = B.tangent(J, t)
    if tau[N + 1] < 0: tau = -tau
    rows = {}
    for i in range(nsteps):
        X, J, M0, M1, p0, res = B.step(X, tau, ds)
        tau = B.tangent(J, tau)
        kap, one, _ = B.scalars(M0, M1, p0)
        c = X[N + 1]
        ev = []
        if i % 12 == 0 or abs(tau[N + 1]) < 3e-4:
            for s0 in (0.010, 0.035):
                ev += [x for x in real_nontrivial(pencil_eigs(M0, M1, s0, k=16), GAM) if x > 0]
            ev = sorted(set(np.round(ev, 8)), reverse=True)
            rows[i] = dict(c=c, sigma=X[N], cdot=tau[N + 1], kappa=kap, phat_one=one,
                           nu=(ev[0] if ev else None), n_unstable=len(ev))
            print(f"{i:4d} c={c:.6f} sigma={X[N]:.6f} cdot={tau[N+1]:+.2e} kappa={kap:+.5f} "
                  f"<p,1>={one:+.6f} nu={ev}", flush=True)
        if tau[N + 1] < 0 and i > 60 and c < 0.8998:
            break
    return rows


def fold_invariant(L=200.0, N=2048, ds=0.02, nsteps=150):
    """max |cdot*kappa + mu*sigmadot*<phat,1>| along the branch (ds=0.02 run from c=0.897)."""
    B = Branch(L=L, N=N, gamma=GAM)
    S, psi, sig = init_branch(L, N); B.S = S
    psi, sig = _march(S, psi, sig, list(np.arange(0.885, 0.8971, 0.005)))
    X = np.concatenate([psi, [sig], [0.897]])
    F, J, M0, M1, p0 = B.F_and_J(X[:N], X[N], X[N + 1])
    t = np.zeros(N + 2); t[N + 1] = 1.0
    tau = B.tangent(J, t)
    if tau[N + 1] < 0: tau = -tau
    worst = 0.0
    for i in range(nsteps):
        X, J, M0, M1, p0, res = B.step(X, tau, ds)
        tau = B.tangent(J, tau)
        kap, one, _ = B.scalars(M0, M1, p0)
        inv = abs(tau[N + 1] * kap + MU * tau[N] * one)
        if i >= 105: worst = max(worst, inv)
        if X[N + 1] < 0.870: break
    print("invariant max (steps >= 105):", worst, flush=True)
    return worst


# --------------------------------------------------------------- Table 7 ---
SCOPE = [("fk", dict(mu=1.0, mu2=0.0, C1=1.0, C2=0.0), 0.1),
         ("mu2m05", dict(mu=1.0, mu2=-0.5, C1=1.0, C2=0.0), 0.1),
         ("nnn", dict(mu=1.0, mu2=0.0, C1=1.0, C2=0.25), 0.1),
         ("mu2nnn", dict(mu=1.0, mu2=0.3, C1=1.0, C2=0.2), 0.1),
         ("soft", dict(mu=0.6, mu2=0.0, C1=1.0, C2=0.0), 0.2)]

def scope(L=200.0, N=2048):
    rows = {}
    for name, kw, gam in SCOPE:
        G = Gen(L=L, N=N, gamma=gam, **kw)
        p = np.zeros(N); f = 0.5
        for c in [0.55, 0.65, 0.75, 0.82]:
            p, f, r = G.solve(c, p, f)
        fp, kap, one = G.scalars(p, 0.82)
        rows[name] = dict(f=f, fprime=fp, kappa=kap, pred=-fp * one,
                          relerr=abs(kap + fp * one) / abs(fp * one), res=r)
        print(name, rows[name], flush=True)
    return rows


# --------------------------------------------------------------- Table 6 ---
def decay(L=200.0, N=4096, c=0.89):
    S, psi, sig = init_branch(L, N)
    psi, sig = _march(S, psi, sig, np.arange(0.881, 0.8901, 0.001))
    S.build(c); M0 = S.jac_psi(psi); p0 = S.Ap + spec_d1(psi, S.k, N)
    ph = left_null(M0)
    p0n, phn = p0 / np.max(np.abs(p0)), ph / np.max(np.abs(ph))
    out = {}
    for x in [0, 10, 20, 40, 80, 120, 199]:
        j, jm = np.argmin(np.abs(S.xi - x)), np.argmin(np.abs(S.xi + x))
        out[str(x)] = dict(p0=float(max(abs(p0n[j]), abs(p0n[jm]))),
                           phat=float(max(abs(phn[j]), abs(phn[jm]))))
    print(out, flush=True)
    return out


def conformal(n=8, gam=0.37, T=0.9, dt=1e-4, seed=1):
    """M^T J M = e^{-gamma T} J for an arbitrary symmetric K(t), plus the one-site shift."""
    rng = np.random.default_rng(seed)
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    Phi = np.eye(2 * n); t = 0.0
    while t < T - 1e-12:
        K = rng.standard_normal((n, n)); K = (K + K.T) / 2
        A = np.block([[np.zeros((n, n)), np.eye(n)], [K, -gam * np.eye(n)]])
        Phi = expm(A * dt) @ Phi; t += dt
    P = np.zeros((n, n)); P[np.arange(n), (np.arange(n) + 1) % n] = 1.0
    M = np.block([[P, np.zeros((n, n))], [np.zeros((n, n)), P]]) @ Phi
    err = np.max(np.abs(M.T @ J @ M - np.exp(-gam * T) * J)) / np.max(np.abs(J))
    print("conformal symplectic relerr:", err, flush=True)
    return float(err)


def mscan(L=200.0, N=2048):
    """Sign of m along the branch (orientation <phat,1> = -2 pi)."""
    S, psi, sig = init_branch(L, N)
    cs = list(np.round(np.arange(0.875, 0.199, -0.005), 5))
    vals = {}
    for c in cs:
        psi, sig, r, _ = S.solve(c, psi, sig)
        if r > 1e-9: break
        if (abs(round(c / 0.05) * 0.05 - c) < 1e-9 and c <= 0.80):
            vals[f"{c:.4f}"] = scalars(S, psi, sig, c)['m']
    S2, psi, sig = init_branch(L, N)
    for c in np.round(np.arange(0.885, 0.8951, 0.005), 5):
        psi, sig, r, _ = S2.solve(c, psi, sig)
        vals[f"{c:.4f}"] = scalars(S2, psi, sig, c)['m']
    print(vals, flush=True)
    return vals


def spectrum(L=200.0, N=2048, c=0.89):
    S, psi, sig = init_branch(L, N)
    psi, sig = _march(S, psi, sig, np.arange(0.881, 0.8901, 0.001))
    d = scalars(S, psi, sig, c)
    nus = pencil_eigs(d['M0'], d['M1'], shift=-0.045, k=40)
    pair = max(min(abs(x + GAM + y) for y in nus) for x in nus)
    print("n real nontrivial:", len(real_nontrivial(nus, GAM)), " pairing err:", pair, flush=True)
    return dict(nus=[[x.real, x.imag] for x in nus], pairing_max_err=float(pair))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    reg = dict(convergence=convergence, threshold=threshold, fold=fold,
               fold_invariant=fold_invariant, scope=scope, decay=decay,
               conformal=conformal, mscan=mscan, spectrum=spectrum)
    res = {k: f() for k, f in reg.items()} if name == "all" else {name: reg[name]()}
    json.dump(res, open(f"session_{name}.json", "w"), indent=1, default=float)
