# Numerical method

What the code actually does, in the order it does it. The mathematics it implements is in
[`../handoff/SCIENCE_BRIEF.md`](../handoff/SCIENCE_BRIEF.md); the choices that are easy to get
wrong are in [`conventions.md`](conventions.md).

## 1. Discretization

The traveling wave solves the advance--delay equation

    c^2 phi'' - gamma c phi' = sum_j C_j (phi(xi+j) - 2 phi(xi) + phi(xi-j)) + f - V'(phi)

on `xi in [-L, L)` with `N` Fourier collocation points, `h = 2L/N`, `xi_j = -L + j h`, and `N`
even so that `xi_{N/2} = 0` is a grid point (the pinning condition lives there).

Fourier space makes both operations in that equation diagonal: `d/dxi` has symbol `i k`, and
the unit shift has symbol `e^{i k}`, so `Delta_j` has symbol `2 cos(j k) - 2`. The whole linear
part is therefore one real circulant matrix, built from `real(ifft(symbol))` -- which is the
reason for working in `xi` rather than integrating the lattice in time. `grid.py` builds the
operators and asserts the symbol is conjugate-symmetric, which it is only after the Nyquist
mode of the odd symbol is zeroed.

The `2 pi` topological jump is carried by an analytic template `A(xi) = 2 pi - 4 arctan(e^{xi/w})`
with fixed `w = 2`. The unknown is the periodic remainder `psi = phi - A`, which also carries
the constant `arcsin(sigma)`. `A`, `A'`, `A''` and `Delta_j A` are evaluated in closed form;
`A` is not periodic and must never go through the FFT.

## 2. The traveling wave

Unknowns `(psi, drive)` in `R^{N+1}`; equations the `N` collocated residuals plus the pinning
row `psi(0) = 0`, i.e. `phi(0) = pi`. The Jacobian is exact,

    [[M0, -1], [e_{j0}^T, 0]],      M0 = Lin + diag(V''(A + psi)),

with the drive column `-1` because `drive = mu sigma` is the additive constant. Damped Newton:
backtracking by halving until the max-norm residual decreases, minimum factor `1e-3`, tolerance
`1e-12`, and a `ConvergenceError` rather than a silent return if the final residual exceeds
`1e-9`. Achieved residuals on the branch are `~1e-13`.

**Initialization is two-stage and the branch depends on it.** At `c0 = 0.88`, converge from
`psi = 0`, `sigma = 0.6` with the *physical* template width `sqrt((1-c0^2)/mu)`; then transfer
the converged profile to the fixed `w = 2` template and re-converge. Experiment 08 deliberately
uses a different, single-stage initialization, because its rows are the reference values of
Table 7 and that is the path they were computed along.

## 3. Continuation

* **Natural**, in `c`, previous solution as the initial guess, step `1e-3` refined to `1e-4`
  near `c_hat1`. This is what every experiment except 04 uses.
* **Pseudo-arclength** in `(psi, drive, c)` for experiment 04, because `c` stops being a valid
  parameter at the turning point. `dR/dc = -M1 phi'` (tested against a finite difference); the
  tangent comes from the bordered system with the previous tangent as the last row; corrector
  tolerance `1e-10`, `ds = 0.02` or `0.03`.

The bordered Newton matrix is singular exactly when `<phat,1> = 0`, which by Corollary 5.2 is
exactly at a turning point in `c`. The failure of natural continuation there is the same fact
as the theorem, not an unrelated numerical nuisance.

## 4. Branch derivatives and biorthogonal scalars

* `sigma'(c)` from the bordered solve `[[M0, -mu 1], [e_{j0}^T, 0]] [d_c psi; sigma'] = [M1 phi'; 0]`.
  Never a finite difference -- see `conventions.md`.
* `phi' = A' + d/dxi psi` spectrally.
* `phat` by inverse iteration on `M0^T + 1e-14 I`, 80 iterations from a seeded Gaussian start,
  renormalized each step. Deterministic. Then normalized by `<phat,1> = -2 pi` or to Euclidean
  unit norm, depending on the experiment.
* `kappa = <phat, M1 phi'>_h`, `phat_one = <phat,1>_h`, `phat_dot_Vpp = <phat, V''(phi)>_h`
  (zero by Lemma 4.2, so a free check).
* `m = <phat, phi'>_h - <phat, M1 q1>_h` with `q1` from the bordered solve
  `[[M0, phat], [phi'^T, 0]] [q1; beta] = [M1 phi'; 0]`.
* Power balance `f = gamma c <phi',phi'>_h / (2 pi)`, nowhere imposed, hence a free check on
  the profile.

## 5. Eigenvalues

Shift-invert Arnoldi (ARPACK) on the companion linearization of the quadratic pencil
`Q(nu) = nu^2 + nu M1 + M0`. The `2N x 2N` matrix is never formed: one `N x N` LU of
`Q(s) = M0 + s M1 + s^2 I` gives the shifted inverse through

    x1 = -Q(s)^{-1} (b2 + (M1 + s I) b1),      x2 = b1 + s x1,

and `nu = s + 1/val`. The start vector is fixed, so runs are reproducible. Shifts, `k` and
tolerances are per-experiment config. Classification of what comes back -- structural,
isolated, or discretized continuum -- is `dlkin.spectral.classify_real_eigenvalues`, described
in `conventions.md`.

## 6. The lattice monodromy (experiment 07 only)

An independent cross-check that leaves this formulation entirely. The profile is sampled at the
integer sites by trigonometric interpolation plus the closed-form template; the nonlinear
lattice is integrated once with RK4 over `T = 1/c` with `V''(u)` cached at the stage times; each
matrix-vector product then integrates the linearization against that cached orbit and applies
the one-site shift. The monodromy is never formed, so

* `M^T J M = e^{-gamma T} J` is tested as `(Mx)^T J (My) = e^{-gamma T} x^T J y` on random pairs;
* `rho = 1` is tested by applying `M` to `z = (phi'(n), -c phi''(n))` -- one product, decisive;
* the dominant multiplier comes from power iteration, not Arnoldi, because all but a handful of
  the `2S` multipliers sit on `|rho| = e^{-gamma/2c} = 0.945`, a ratio too close to 1 for a
  restarted Krylov method to separate quickly but perfectly manageable for power iteration.

The number to read first is the drift: after one period the lattice state should equal the
initial one shifted by a site, and it does to `~4e-6`, which bounds everything else measured
there.

## 7. Resolutions

`L = 200`, `N = 4096` (`h ~ 0.098`) for the identity, threshold and null-vector work;
`N = 2048` (`h ~ 0.195`) for arclength continuation, the spectrum snapshots, the SVD and the
model survey. The convergence study spans `(L, N)` from `(200, 1024)` to `(300, 6144)`. Dense
linear algebra throughout, cost `O(N^3)`; see [`../reports/RUNTIME.md`](../reports/RUNTIME.md).
