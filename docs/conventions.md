# Conventions

Short list of the choices that are easy to get wrong, each with the reason it was made. Every
one of them has a test or a claim behind it.

## Sign and pairing

* **Bilinear pairing, not sesquilinear.** `<f,g> = int f g dxi`, no conjugation, and `T^T` is
  the formal transpose with respect to it. The identity `Q(-gamma-nu) = Q(nu)^T` is a
  statement about transposes; with a conjugating pairing it would carry a stray complex
  conjugate and fail for complex `nu`. Tested in `tests/test_monodromy.py::test_pencil_duality_is_exact`.
* **Discrete pairing** is `<f,g>_h = h * sum_j f_j g_j`. The grid weight matters for `kappa`
  and `<phat,1>` individually; it cancels in every ratio.
* **`M0^T` is `M0` with `gamma -> -gamma`.** That is why the left null vector `phat` is the
  null vector of the *anti-damped* advance--delay equation, and why `nu = -gamma` is always in
  the spectrum.

## Normalization of `phat`

`phat` spans a one-dimensional space, so every quantity linear in it -- `kappa`, `<phat,1>`,
`m` -- depends on how it is scaled. Two conventions are in use and `normalization` is a
**required argument** with no default, because mixing them silently changes numbers by a
factor of a few:

| mode | definition | used by | consequence |
|---|---|---|---|
| `minus2pi` | `<phat,1>_h = -2 pi`, matching `<phi',1> = phi(+inf) - phi(-inf)` | experiments 01--03, 05, 06 | `kappa = 2 pi mu sigma'(c)` reads directly; resolution-independent |
| `unit` | Euclidean norm of the grid vector `= 1`, sign fixed by `<phat,phi'>_h > 0` | experiments 04, 08 | `kappa` and `<phat,1>` scale like `sqrt(N)`; only their relation is meaningful |

`minus2pi` is unavailable exactly where `<phat,1> = 0` -- which, by Corollary 5.2, is exactly at
a turning point of the branch in `c`. That is why the fold experiment uses `unit`: the
normalization that makes the identity most readable is the one that breaks at the fold, and
that is a theorem, not an inconvenience.

## Discretization

* **The Nyquist mode of the first-derivative symbol is set to zero.** Otherwise the symbol
  fails `s(-k) = conj(s(k))` at `k = k_Nyquist` and the circulant matrix is not real. Tested in
  `tests/test_operators.py`.
* **The template is never differentiated spectrally.** `A(xi) = 2 pi - 4 arctan(e^{xi/w})` is
  not periodic; `A'`, `A''` and `Delta_j A` are closed-form. Only the periodic remainder `psi`
  goes through the FFT.
* **Template width `w = 2`, fixed**, not the physical kink width. The two-stage `init_branch`
  converges first with the physical width `sqrt((1-c^2)/mu)` -- which is what Newton likes at
  `c0 = 0.88` -- then transfers the profile to the fixed template. The reported numbers lie on
  the branch that procedure reaches.

## Derivatives along the branch

`sigma'(c)` is **always** the bordered solve

    [[M0, -mu*1], [e_{j0}^T, 0]] [d_c psi; sigma'] = [M1 phi'; 0]

never a finite difference. A centred difference with spacing `1e-3` is wrong by `1.6e-4` in
relative terms at `c = 0.89` -- a truncation error that does not shrink with `N` or `L`, and
that would put a floor ten orders of magnitude above the identity's measured residual.
Experiment 02 computes and records both, for exactly this reason.

## Eigenvalues

* `rho = exp(nu/c)`; instability is `Re nu > 0`, i.e. `|rho| > 1`.
* `nu = 0` (translation) and `nu = -gamma` (its conformal dual) are always present and are
  excluded before counting isolated modes. The cut scales with the accuracy actually achieved,
  not a fixed constant.
* A real candidate is classified as discretized essential spectrum when it is **both** half of
  a dual pair summing to `-gamma` **and** within `line_factor` spreads of `Re nu = -gamma/2`.
  Neither test alone suffices: every eigenvalue has a dual partner, so the pairing test alone
  would discard genuine isolated modes -- which it did, in an early version of experiment 06,
  throwing away the unstable eigenvalue at `c = 0.8995`. One implementation,
  `dlkin.spectral.classify_real_eigenvalues`, is shared by experiments 03 and 06.
