# Results

What each experiment shows, with the numbers from the committed `data/`. Every value here is
also a claim in [`../manuscript/claims.yaml`](../manuscript/claims.yaml) unless marked
*cross-check*, and `make check` re-verifies all of them.

## 01 -- the kinetic relation

`sigma(c)` rises on the first branch to a maximum at `c_hat1 = 0.8989297`,
`sigma_hat1 = 0.6501918`, located as the zero of the exactly computed
`sigma'(c)` by secant. The 2020 lattice computation reported 0.8989 and 0.65019. The two are
obtained by different methods -- advance-delay collocation here, Newton on the lattice period
map there -- so the agreement is an independent check. ![](../figures/fig1_kinetic_relation.pdf)

## 02 -- the identity

At `c = 0.89`, `L = 200`, `N = 4096`: `sigma' = 1.9251320610`,
`kappa = 12.0959614798` with `<phat,1> = -2 pi`, so `kappa = 2 pi mu sigma'` to
machine precision; `<phat, cos phi> = 1.2e-13` (Lemma 4.2); power balance
to `1e-13`. Under refinement the relative error goes
`3.0e-04` -> `1.3e-08`
-> `4.2e-14` as `h` halves, and is the same at `L = 150, 200, 300`:
spectral convergence, no truncation effect. A centred finite difference for `sigma'` would have
been wrong by `1.6e-04` -- which is why it is never used.
![](../figures/fig2_jordan_identity.pdf) ![](../figures/fig6_convergence.pdf)

## 03 -- the first extremum

`kappa` and `sigma'` cross zero together, and so does the eigenvalue `nu2`: by linear
interpolation `0.8989293` and `0.8989296`.
The leading-order prediction `-kappa/m` converges to `nu2` as the threshold is approached
(`10.3%` off at `c = 0.898`, `1.5%` at `0.8988`),
and `m(c_hat1) = 283.4 > 0`, which is the sign Corollary 6.2 needs for
the slope criterion to hold locally. `m` changes sign between `c = 0.8999` and `0.9000`, where
`nu2` is no longer small; that marks the edge of the reduction's validity, not a change of
stability -- `nu2` stays positive.

## 04 -- the first turning point

Pseudo-arclength through `c_max = 0.900196`: `cdot` and `<phat,1>` change sign in the
same continuation step (at fractions `0.336` and
`0.342` of it), while `kappa` passes through at
`-2.533` and exactly one unstable eigenvalue (`nu = 0.0658`) is
present before, at and after. The invariant `cdot kappa + mu sigmadot <phat,1>` stays at the
discretization level of the grid. ![](../figures/fig3_fold_dichotomy.pdf)

## 05 -- the null vectors

Singular values of `M0` at `N = 2048`: `2.66e-09`, `9.887e-02`,
`1.909e-01` -- a gap of `3.7e+07`, which is what "the kernel is
one-dimensional" means numerically. Both null vectors fall by nine orders of magnitude across
the domain, which is the numerical content of the decay hypothesis.

They are not proportional, and *found during the final validation pass*: their tails are
mirror images. The envelope of `|phat|` at `+xi` matches that of `|phi'|` at `-xi` to within
`0.7%`--`4.7%`
at `|xi| = 50, 100, 150, 190`. The anti-damped adjoint radiates on the opposite side of the kink.
The exponential weight `e^(-gamma xi/c)` that would make the weighted-space argument work
requires the log-envelope ratio to fall with slope `-gamma/c = -0.112`;
it rises with slope `+0.097`. ![](../figures/fig4_null_vectors.pdf)

## 06 -- the spectrum

At `c = 0.89` the computed spectrum contains `nu = 7.3e-09` and
`nu = -0.100000007` -- the structural pair `0, -gamma` -- and no other
isolated real eigenvalue; every computed eigenvalue has a dual partner with `nu + nu' = -gamma`
to `7e-15`. At `c = 0.8995` there is exactly one unstable real
eigenvalue, `+0.017591`. ![](../figures/fig5_spectrum.pdf)

## 07 -- conformal symplecticity

For a symmetric `K(t)` resampled at random every step, `|M^T J M - e^(-gamma T) J| / |J| =
1.9e-14`, and at most `2.7e-14` over the
`(n, dt)` sweep.

*Cross-check, not a manuscript claim:* going back to the lattice in `(n, t)` -- the method of the
2020 paper -- the sampled profile is a genuine lattice traveling wave to
`3.7e-06` per period; the actual FK monodromy satisfies `M^T J M = e^(-gamma/c) J`
to `6e-14`; it maps the translation mode to itself to
`6.5e-06`; and its dominant multiplier at `c = 0.8995` is
`1.02000` against `1.01975` from the pencil.
Two completely different methods agree on the unstable multiplier to four figures.

## 08 -- other lattices

`kappa = -f'(c) <phat,1>` at `c = 0.82` for FK, FK with a second harmonic `mu2 = -0.5`, with
next-nearest-neighbour coupling, with both, and with a softer substrate at stronger damping:
relative errors `2e-10`, `4e-06`,
`1e-14`, `2e-11`,
`2e-13`. The `mu2 = -0.5` row is a resolution effect: at `N = 4096`
it falls to `2e-12`. `mu2 = +0.35` was attempted and does
not converge from `psi = 0`; that is a failure of the initialization, recorded as such.
