"""Re-verification of the key numbers quoted in the note (run after sandbox reset)."""
import numpy as np, time, json, warnings; warnings.filterwarnings("ignore")
from fk import FKSolver, init_branch
from spectral import scalars, pencil_eigs, real_nontrivial
out = {}
t0 = time.time()
# (a) boundary-condition check, c = 0.88, physical template, N=2048, L=200 (as in session)
Sp = FKSolver(L=200.0, N=2048, tw=np.sqrt(1 - 0.88**2)); o = Sp.solve(0.88, np.zeros(2048), 0.6)
phi = Sp.A + o[0]; sg = o[1]
out['a'] = dict(sigma088=sg, res=o[2], phiL=phi[0], phiR=phi[-1],
                bcL=np.arcsin(sg) + 2*np.pi, bcR=np.arcsin(sg))
print("(a)", out['a'], flush=True)
# (b) N=4096, L=200, tw=2: c=0.89 scalars
S, psi, sig = init_branch(200.0, 4096)
for c in np.round(np.arange(0.881, 0.8901, 0.001), 6):
    psi, sig, r, _ = S.solve(c, psi, sig)
d = scalars(S, psi, sig, 0.89)
out['b'] = {k: d[k] for k in ['sigma','sigp','kappa','one','orth','m','nu2_pred','ident_relerr','sigma_pb','res_M0p0','res_phat','dcpsi_bdry']}
out['b']['pb_relerr'] = abs(d['sigma'] - d['sigma_pb']) / d['sigma']
out['b']['dinf_pred'] = d['sigp'] / np.sqrt(1 - d['sigma']**2)
print("(b)", out['b'], "t=%.0f" % (time.time()-t0), flush=True)
# (c) threshold: exact zero of sigma'(c) by secant, N=4096
rows = {}
for c in [0.8988, 0.89892, 0.8990]:
    psi, sig, r, _ = S.solve(c, psi, sig)
    d = scalars(S, psi, sig, c)
    rows[c] = (sig, d['sigp'], d['kappa'], d['m'], d['nu2_pred'], r)
    print("(c)", c, rows[c], flush=True)
a, b = 0.89892, 0.8990
fa, fb = rows[a][1], rows[b][1]
psi_b, sig_b = psi.copy(), sig
for _ in range(6):
    cm = b - fb * (b - a) / (fb - fa)
    psi, sig, r, _ = S.solve(cm, psi, sig)
    fm = scalars(S, psi, sig, cm)['sigp']
    a, fa, b, fb = b, fb, cm, fm
    if abs(fm) < 1e-10: break
out['c_hat1'] = cm; out['sigma_hat1'] = sig
print("(c) c_hat1 = %.7f  sigma_hat1 = %.7f  sigma'=%.1e" % (cm, sig, fm), flush=True)
# (d) Arnoldi at c = 0.8990 and 0.8988
for c in [0.8988, 0.8990]:
    psi, sig, r, _ = S.solve(c, psi, sig)
    d = scalars(S, psi, sig, c)
    nus = pencil_eigs(d['M0'], d['M1'], shift=max(d['nu2_pred'], 0.0) + 0.004, k=24)
    rr = real_nontrivial(nus, 0.1)
    cand = [x for x in rr if abs(x - d['nu2_pred']) < 0.05]
    pair = [abs((x + y.conjugate()*0 ) ) for x in []]
    out['d%.4f' % c] = dict(nu2_pred=d['nu2_pred'], nu2=cand[0] if cand else None,
                            has_zero=bool(np.min(np.abs(nus)) < 1e-6),
                            has_minus_gamma=bool(np.min(np.abs(nus + 0.1)) < 1e-6))
    print("(d)", c, out['d%.4f' % c], "t=%.0f" % (time.time()-t0), flush=True)
json.dump(out, open('verify_core.json', 'w'), indent=1, default=float)
print("done", time.time() - t0)
