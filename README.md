# damped-lattice-kinetic

Computational companion to a research note on **traveling kinks in damped, driven lattices**.

In the damped, dc-driven Frenkel–Kontorova lattice

    u_n'' + gamma u_n' = u_{n+1} - 2 u_n + u_{n-1} + mu (sigma - sin u_n)

traveling kinks come in families along which the driving force is a function of the
velocity, the *kinetic relation* `sigma(c)`. Vainchtein, Cuevas-Maraver, Kevrekidis and Xu
(*CNSNS* **85**, 105236, 2020) found numerically that a kink acquires an unstable direction
exactly where `sigma(c)` has an extremum, and identified the obstruction to a proof: damping
destroys the generalized eigenvector at the neutral Floquet multiplier that the Hamiltonian
argument relies on.

The note shows that this obstruction can be measured exactly. With `phat` the null vector of
the transpose of the linearization, the scalar that decides whether the neutral eigenvalue is
simple satisfies

    kappa(c) = <phat, M1 phi'> = -mu sigma'(c) <phat, 1>,      M1 = -2c d/dxi + gamma,

and, along the branch parametrized by arclength, `cdot kappa + mu sigmadot <phat,1> = 0`. A
multiplier can therefore reach `+1` only at an extremum of `sigma`; at a turning point of the
branch in `c` it is `<phat,1>` that vanishes instead and the stability does not change.

This repository computes every number in the note, checks each one against a manifest, and
regenerates every figure and table.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps   # pinned versions; ~1 min
# (unpinned alternative: pip install -e ".[dev]" -- results agree, figure bytes may not)

make test                        # fast tests, ~30 s
make check                       # resolve all 129 manuscript claims against data/ -- seconds
make figures tables              # regenerate figures/ and tables/ from data/ -- seconds
```

`data/` is committed, so `make check`, `make figures` and `make tables` work immediately from
a fresh clone. To regenerate the data itself:

```bash
make data-quick                  # every experiment at reduced resolution, < 1 min
make data                        # every experiment at production resolution, ~1 h on one core
make check                       # then confirm nothing moved
```

`make status` shows which commit produced each result file and flags any that predate `HEAD`.

## What reproduces what

| manuscript | script | result file | claims |
|---|---|---|---|
| Table 2 (`c_hat1`, `sigma_hat1`; validation against the 2020 paper) | `scripts/01_kinetic_relation.py` | `data/01_kinetic.json`, `01_kinetic_curve.csv` | `VAL-*` |
| Obs. 9.2, Table 3 (the identity, its convergence) | `scripts/02_identity_convergence.py` | `data/02_identity.json` | `ID-*`, `CV-*` |
| Obs. 9.3, Table 4 (the first extremum) | `scripts/03_threshold.py` | `data/03_threshold.json` | `TH-*` |
| Obs. 9.4, Table 5 (the first turning point) | `scripts/04_fold_arclength.py` | `data/04_fold.json` | `FD-*`, `VAL-cmax` |
| Obs. 9.5, Table 6 (null vectors, kernel, reflection) | `scripts/05_null_vectors.py` | `data/05_nullvec.json`, `05_null_profiles.csv` | `NV-*` |
| Obs. 9.6 (spectrum, pairing) | `scripts/06_spectrum.py` | `data/06_spectrum.json`, `06_spectrum_*.csv` | `SP-*` |
| Obs. 9.6 (conformal symplecticity) | `scripts/07_conformal_symplectic.py` | `data/07_conformal.json` | `CS-*` |
| Obs. 9.7, Table 7 (other lattices) | `scripts/08_general_lattices.py` | `data/08_general.json` | `GN-*` |
| Figures 1–6 | `scripts/09_figures.py` | reads all of the above | — |
| table fragments, manuscript diff | `scripts/10_tables.py` | `tables/*.tex`, `tables/DIFF_REPORT.md` | — |

`scripts/check_claims.py` resolves every number in [`manuscript/claims.yaml`](manuscript/claims.yaml)
— the key path that produces it, the tolerance it must meet — and writes
[`reports/validation_report.md`](reports/validation_report.md). A missing file or key is a
failure, never a skip, and data written by a `--quick` run cannot satisfy a claim under
`--strict`. `scripts/10_tables.py` then checks the other direction: that every value in the
typeset tables matches the data at the precision the manuscript prints.

## Layout

| path | contents |
|---|---|
| `src/dlkin/grid.py` | spectral grid on `[-L, L)`, Fourier symbols, real circulant operators |
| `src/dlkin/model.py` | `LatticeModel` (damping, onsite potential, finite-range coupling) and the analytic kink template |
| `src/dlkin/solver.py` | `TravelingWaveSolver`: damped Newton on the collocated advance–delay equation; `init_branch` |
| `src/dlkin/continuation.py` | natural continuation in `c`, pseudo-arclength through the fold |
| `src/dlkin/spectral.py` | `phat`, the exact `sigma'(c)`, `kappa`, `m`; eigenvalues of the co-traveling pencil and their classification |
| `src/dlkin/monodromy.py` | conformal-symplectic tests and the direct lattice monodromy (independent cross-check) |
| `src/dlkin/config.py`, `io.py` | YAML configs with a `quick:` overlay; result files with a provenance block |
| `configs/` | every numerical setting of every experiment — nothing is hard-coded in a script |
| `scripts/` | experiments `01`–`08`, figures `09`, tables `10`, `check_claims.py` |
| `data/` | result JSON and CSV, committed; each records the commit, config and runtime that produced it |
| `figures/`, `tables/`, `reports/` | generated output and the validation report |
| `manuscript/` | `note.tex`, its claims manifest, build instructions |
| `docs/` | [numerical method](docs/numerics.md), [conventions](docs/conventions.md), [results](docs/results.md) |
| `handoff/` | provenance: the science brief and the reference implementation the package is tested against |
| `tests/` | identities to machine precision, regression against the reference, convergence, end-to-end |

## What is established and what is not

Proved in the note: the conformal symplecticity of the monodromy, the operator duality
`Q(-gamma-nu) = Q(nu)^T`, the Fredholm property of the linearization, and the power balance.
Proved *under hypotheses*: the identity above, its arclength form and the fold dichotomy, and
the local crossing law at an extremum. The hypotheses are that `ker M0` is one-dimensional and
that `phat` decays exponentially; both are supported numerically here (a singular-value gap of
`3.7e7`, and nine orders of decay across the domain) but not proved.

Verified numerically, at `mu = 1`, `gamma = 0.1`: the identity to `~1e-13` with spectral
convergence and no dependence on the domain size; `c_hat1 = 0.8989297`, `sigma_hat1 = 0.6501918`
and `c_max = 0.900196` (the 2020 paper: 0.8989, 0.65019, 0.9002); the stability transition at
the zero of `sigma'` to `1e-6` in `c`; the fold dichotomy at the first turning point; and the
identity in four modified lattices.

**Not** established: the sign of the coefficient `m` that makes the slope criterion
sufficient (positive wherever computed, not proved); anything beyond a neighbourhood of each
extremum; crossings through `rho = -1` or by complex pairs; and the weakly damped
`gamma = 0.01` resonance regime, which is untested.

## Reproducibility notes

* Deterministic: fixed seeds for the inverse iteration, a fixed ARPACK start vector, no
  wall-clock-dependent branching.
* Figures are byte-identical across repeated runs **in the pinned environment**
  (`requirements.txt`, matplotlib 3.11.2), which is what produced the committed PDFs. Other
  matplotlib versions draw the same data with different glyph placement -- 3.10.8, for
  instance, differs in about a tenth of the pixels -- so the PDF bytes are only comparable
  within one environment. The data, the claims and the tables do not depend on it.
* Dense linear algebra throughout, `O(N^3)`; production runs use `N = 4096` (~1 GB). The one
  `L = 300, N = 6144` case (~2 GB) has its own selector:
  `python scripts/02_identity_convergence.py --only convergence --cases L300_N6144`.
* Numbers at the round-off floor (relative errors near `1e-14`) are platform-dependent in their
  last digits; the claims bound them rather than pinning them.
* See [`reports/RUNTIME.md`](reports/RUNTIME.md) for timings.

## Citation and licence

Code under the MIT licence (`LICENSE`); the manuscript text is not covered by it. See
`CITATION.cff`.
