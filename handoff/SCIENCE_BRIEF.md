# Science brief

Self-contained statement of the problem, the results, the conventions and the numerical
recipe behind the research note `note.tex`. Read this in full before writing code.

---

## 1. Model

Damped, dc-driven Frenkel–Kontorova lattice (Vainchtein, Cuevas-Maraver, Kevrekidis & Xu,
*Commun. Nonlinear Sci. Numer. Simul.* **85** (2020) 105236, eq. (1)):

```
u_n'' + gamma u_n' = u_{n+1} - 2 u_n + u_{n-1} + mu (sigma - sin u_n)
```

Traveling kink `u_n(t) = phi(xi)`, `xi = n - c t`, solves the **advance–delay equation**

```
c^2 phi''(xi) - gamma c phi'(xi) = phi(xi+1) - 2 phi(xi) + phi(xi-1) + mu (sigma - sin phi(xi))   (TW)
phi -> arcsin(sigma) + 2*pi  (xi -> -inf),   phi -> arcsin(sigma)  (xi -> +inf),   phi(0) = pi
```

For `gamma > 0` the solutions form a one-parameter family; the driving force as a function of
the velocity is the **kinetic relation** `sigma = sigma(c)`. It is multivalued: the first branch
rises to a maximum at `c_hat1`, falls to a turning point in `c` at `c_max`, and continues on a
second branch. Default parameters throughout: `mu = 1`, `gamma = 0.1`.

Generalized model (scope test):

```
u_n'' + gamma u_n' = C1*(u_{n+1}-2u_n+u_{n-1}) + C2*(u_{n+2}-2u_n+u_{n-2}) + f - V'(u_n)
V'(u) = mu sin(u) + mu2 sin(2u),    f = additive drive (plays the role of mu*sigma)
```

## 2. Operators

Linearizing (TW) at fixed `(sigma, c)`, and writing `Delta_1 p = p(.+1) + p(.-1) - 2p`:

```
M0 = c^2 d_xi^2 - gamma c d_xi - Delta_1 + mu cos(phi)        (linearization)
M1 = -2 c d_xi + gamma                                        (d/d nu of the pencil)
Q(nu) = nu^2 + nu M1 + M0                                     (co-traveling quadratic pencil)
```

Floquet solutions `xi_n(t) = e^{nu t} p(n - c t)` of the linearized lattice satisfy
`Q(nu) p = 0`, and the Floquet multiplier of the monodromy ("evolve one period `T = 1/c`, then
shift one site") is `rho = e^{nu/c}`. Instability means `Re nu > 0`, i.e. `|rho| > 1`.

Transpose with respect to the **bilinear** pairing `<f,g> = int f g dxi` (no conjugation):
`d_xi^T = -d_xi`, `Delta_1^T = Delta_1`, so

```
M0^T = c^2 d_xi^2 + gamma c d_xi - Delta_1 + mu cos(phi)      (= M0 with gamma -> -gamma)
```

`phi'` spans `ker M0`; `phat` spans `ker M0^T`.

## 3. The results the repository must support

All quantities below are computed, not assumed.

**(R1) Conformal symplecticity.** With `J = [[0,I],[-I,0]]` and `A(t) = [[0,I],[K(t),-gamma I]]`,
`K = K^T`: `A^T J + J A = -gamma J`, hence `Phi(T)^T J Phi(T) = e^{-gamma T} J` and
`M^T J M = e^{-gamma/c} J`. Consequence: multipliers pair as `rho * rho_hat = e^{-gamma/c}`.

**(R2) Pencil duality.** `Q(-gamma-nu) = Q(nu)^T` exactly. Hence the spectrum is invariant
under `nu -> -gamma - nu`, and the **left** null vector at `nu` is the **right** null vector at
`-gamma - nu`. At `nu = 0` this says the left neutral vector is `phat`, the null vector of the
anti-damped equation, and `nu = -gamma` (`rho = e^{-gamma/c}`) is always an eigenvalue.

**(R3) Essential spectrum.** Replacing `cos phi` by its limit `sqrt(1-sigma^2)` gives the
symbol `s(nu,k) = nu^2 + nu(gamma - 2 i c k) + L(k)` with
`L(k) = -c^2 k^2 - i gamma c k + 4 sin^2(k/2) + mu sqrt(1-sigma^2)`. Writing `nu = lambda + i c k`
gives `lambda^2 + gamma lambda + omega_k^2` with `omega_k^2 = 4 sin^2(k/2) + mu sqrt(1-sigma^2)`,
so the constant-coefficient spectrum is exactly `Re nu = -gamma/2` when
`mu sqrt(1-sigma^2) > gamma^2/4` — i.e. the multiplier circle `|rho| = e^{-gamma/(2c)}`.
`L(k) != 0` for all real `k` when `gamma > 0`, which is why `M0` is Fredholm of index zero.

**(R4) Jordan-chain identity (the central result).** Define

```
kappa(c) = <phat, M1 phi'>
```

Then, from `M0 [d_c phi] = M1 phi' + mu sigma'(c) * 1` (differentiate (TW) along the branch)
paired with `phat`:

```
kappa(c) = -mu * sigma'(c) * <phat, 1>            and        <phat, cos phi> = 0
```

With the normalization `<phat,1> = <phi',1> = -2*pi` this is `kappa = 2*pi*mu*sigma'(c)`.
`nu = 0` is an algebraically simple eigenvalue of the pencil iff `kappa != 0`; the Jordan
partner, where it exists, is `-d_c phi`.

The second identity is free: `M0 * 1 = mu cos(phi)`, so `<phat, mu cos phi> = <M0^T phat, 1> = 0`.

**(R5) Arclength form and the fold dichotomy.** Parametrizing the branch by arclength `s`:

```
cdot(s) * kappa(s) + mu * sigmadot(s) * <phat,1> = 0
```

So at an extremum of `sigma` (`sigmadot = 0`, `cdot != 0`) we get `kappa = 0`, while at a
turning point in `c` (`cdot = 0`, `sigmadot != 0`) we get `<phat,1> = 0` with `kappa != 0` —
no multiplier reaches `+1` and the stability does not change there.

**(R6) Power balance.** `mu * sigma = (gamma c / 2 pi) * int phi'(xi)^2 dxi`. (General model:
`f = (gamma c / 2 pi) int phi'^2`.) Nowhere imposed; a free check on the profile.

**(R7) Local crossing.** With `Pi f = phat <phat,f>/<phat,phat>`, let `q1` solve
`M0 q1 = (I - Pi) M1 phi'` with `<phi', q1> = 0`, and set

```
m(c) = <phat, phi'> - <phat, M1 q1>
```

Then near an extremum the non-trivial eigenvalue is `nu2 ~ -kappa/m`, and
`nu2'(c_hat) = mu sigma''(c_hat) <phat,1> / m(c_hat)`. `m(c_hat) = 0` iff the Jordan chain has
length three.

**(R8) Generality.** For the generalized model, `kappa = -f'(c) <phat,1>`,
`<phat, V''(phi)> = 0`, same power balance. Requires: additive drive, translation invariance,
symmetric finite-range coupling, uniform scalar damping.

## 4. Numerical recipe (this is the method that produced the reported numbers)

**Discretization.** Fourier spectral collocation on `xi in [-L, L)`, `N` points,
`h = 2L/N`, `xi_j = -L + j h`; `N` even so `xi_{N/2} = 0` is a grid point. Wavenumbers
`k = 2 pi fftfreq(N, d=h)`. Symbols: `d_xi -> i k` with the **Nyquist mode zeroed**;
`d_xi^2 -> -k^2`; `Delta_1 -> 2 cos k - 2`; `Delta_2 -> 2 cos 2k - 2`. Each operator is built
as a real circulant matrix from `real(ifft(symbol))`.

**Topological jump.** `phi = A + psi` with the analytic template
`A(xi) = 2 pi - 4 arctan(exp(xi/w))`, fixed width `w = 2`. `A'`, `A''` and `Delta_1 A` are
evaluated in closed form (never spectrally — `A` is not periodic). The unknown `psi` is
periodic and carries the constant `arcsin(sigma)`.

**Newton.** Unknowns `(psi, sigma) in R^{N+1}`; equations: the `N` collocated residuals plus
the pinning row `psi(0) = 0` (equivalent to `phi(0) = pi`). Jacobian
`[[M0, -mu*1],[e_{j0}^T, 0]]`. Backtracking line search halving the step until the max-norm
residual decreases, minimum factor `1e-3`. Tolerance `1e-12` on
`max(||R||_inf, |psi(0)|)`; achieved residuals are `~1e-13`.

**Initialization (exact procedure — reproduce it).** At `c0 = 0.88`: solve from `psi = 0`,
`sigma = 0.6` with the *physical-width* template `w0 = sqrt((1-c0^2)/mu)`; then transfer the
converged profile to the `w = 2` template (`psi = phi - A_{w=2}`) and re-converge. Then
continue in `c`.

**Continuation.** Natural continuation in `c`, previous solution as the initial guess,
step `1e-3` (refined to `1e-4` near `c_hat1`). Through the turning point: pseudo-arclength in
`(psi, sigma, c)`; `dR/dc = -M1 phi'`; tangent from the bordered system with the previous
tangent as the last row; corrector tolerance `1e-10`; `ds = 0.02` or `0.03`.

**Exact `sigma'(c)`** (never finite differences — a centred difference with step `1e-3` has a
grid-independent error of `1.6e-4` near the curvature maximum): solve the bordered system

```
[[M0, -mu*1],[e_{j0}^T, 0]] [d_c psi ; sigma'] = [M1 phi' ; 0]
```

**`phi'`** is `A' + (spectral derivative of psi)`.

**Left null vector `phat`.** Inverse iteration on `M0^T + 1e-14 * I`, 70–80 iterations,
seeded Gaussian start (fixed seed), renormalized each step. Then either
`<phat,1>_h = -2 pi` (used for `kappa`, `m` in the threshold tables) or `||phat||_2 = 1`
(Euclidean norm of the grid vector) with `<phat, phi'>_h > 0` (used for the fold and scope
tables). **Which normalization is in force changes `kappa` and `<phat,1>` individually but
not their relation or any ratio — always record it.**

**Discrete pairing.** `<f,g>_h = h * sum_j f_j g_j`.

**`m`.** Bordered solve `[[M0, phat],[phi'^T, 0]] [q1; beta] = [M1 phi'; 0]`, then
`m = <phat, phi'>_h - <phat, M1 q1>_h`.

**Eigenvalues.** Shift-invert Arnoldi (ARPACK) on the companion linearization
`[[0, I], [-M0, -M1]]`. One `N x N` LU of `Q(s) = M0 + s M1 + s^2 I` suffices per shift:

```
x1 = -Q(s)^{-1} ( b2 + (M1 + s I) b1 ),     x2 = b1 + s x1
```

`k = 14..24` eigenvalues, `tol = 1e-10`, `maxiter = 8000`, **fixed starting vector `v0`** for
determinism. Recover `nu = s + 1/val`. Real if `|Im nu| < 1e-7`; discard `|nu| < 1e-6` (the
translation mode) and `|nu + gamma| < 1e-6` (its conformal partner). Shifts used:
`max(nu2_pred, 0) + 0.004` for the threshold table; `0.010` and `0.035` for the fold;
`-0.045` for the full spectrum snapshot. Multipliers: `rho = exp(nu/c)`.

**Resolution.** `L = 200`, `N = 4096` (`h ~ 0.098`) for the identity and threshold tables;
`N = 2048` (`h ~ 0.195`) for arclength continuation, the scope survey and the singular values.
Dense linear algebra, cost `O(N^3)`; `N = 4096` needs roughly 1 GB.

## 5. Numbers that must come out

See `claims.yaml` for the full list with tolerances. The critical ones:

| quantity | value | reference |
|---|---|---|
| `c_hat1` (zero of exact `sigma'`) | 0.8989297 | 0.8989 in the 2020 paper |
| `sigma(c_hat1)` | 0.6501918 | 0.65019 |
| `c_max` (turning point) | 0.900196 | 0.9002 |
| `sigma(0.89)`, `sigma'(0.89)` | 0.6373006910, 1.9251320610 | |
| `kappa(0.89)` at `<phat,1> = -2 pi` | 12.0959614798 | `= 2 pi mu sigma'` |
| identity relative error, `N = 4096` | `<= 1e-12` | `3.05e-4` at `N=1024`, `1.26e-8` at `N=2048` |
| power balance relative error | `4e-14` | |
| `nu2(0.899)` (Arnoldi) | +0.001931 | unstable, `rho = 1.00215` |
| `nu2(0.8988)` | -0.003441 | stable |
| smallest singular values of `M0`, `N=2048` | 2.66e-9, 9.89e-2, 1.91e-1 | kernel is simple |

## 6. What is proved and what is not — do not blur this in code comments or docs

Proved: R1, R2, R3, the Fredholm index-zero lemma, R6, and — under stated hypotheses on the
branch and on the decay of `phat` — R4, R5, R7, R8.
Not proved: simplicity of `ker M0` on the infinite lattice; exponential decay (hence `L^1`
membership) of `phat`; the sign of `m`; anything outside a neighbourhood of each extremum;
crossings through `rho = -1` or by complex pairs; the weakly damped (`gamma = 0.01`)
resonance regime, which is untested here.
