# damped-lattice-kinetic

Reproducibility repository for a research note on **traveling kinks in damped, driven
lattices**: the kinetic relation `sigma(c)` of the damped, dc-driven Frenkel–Kontorova
lattice, the Jordan-chain identity `kappa = 2 pi mu sigma'(c)` that ties it to the
spectrum of the co-traveling linearization, and the fold dichotomy at `c_max`.

The model, the conventions, the identities and the numerical recipe are stated in
[`handoff/SCIENCE_BRIEF.md`](handoff/SCIENCE_BRIEF.md); the numbers the code must
reproduce are listed in [`handoff/claims.yaml`](handoff/claims.yaml).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
make install          # pip install -e ".[dev]"
make test             # pytest -q -m "not slow"
```

## Reproducing everything

```bash
make data             # experiments 01-08 at production settings   (~50 min, up to ~2 GB)
make check            # resolve every claims.yaml claim, strictly, and write the report
make all              # data figures tables check
```

`make data-quick` runs the same eight experiments at reduced `L`/`N` in about a minute;
those results are **not** expected to reproduce `claims.yaml` and the data files they write
say so, which `make check` then refuses to count as passes.

Nothing reuses stale data in silence. Every result file records the git sha and the wall
clock it was produced at, `make data` prints them before and after the run via
[`scripts/provenance.py`](scripts/provenance.py), and any file whose script, config or
library changed since it was written is called out. That check is per data file and
deliberately conservative: it reports a changed dependency without judging whether the
change was substantive, so it fires on the commit that records a run (that commit can also
touch the library) — there, a re-run is optional. What it will never do is stay silent. [`reports/RUNTIME.md`](reports/RUNTIME.md) is regenerated from the
same metadata and says which parts of the pipeline are impractical in CI.

## Checking the manuscript against the data

[`scripts/check_claims.py`](scripts/check_claims.py) resolves all 125 claims of the
manifest against `data/*.json` and writes
[`reports/validation_report.md`](reports/validation_report.md) plus a machine-readable
`reports/validation.json`. The manifest lives in two places on purpose:
`handoff/claims.yaml` is where it arrived, and `manuscript/claims.yaml` is the copy that
travels with the paper — the checker prefers the manuscript copy, and
`--compare-manifests` reports if the two have drifted apart.

Three rules, all of them there because a claim that quietly goes unchecked is worse than
one that visibly fails:

- a missing file or an unresolvable key is a **failure**, never a skip;
- a data file marked `quick` has its claims reported `SKIPPED-QUICK`, loudly, and
  `--strict` (which `make check` uses) turns those into failures;
- a claim marked `exact_replication` that fails gets its own section, *procedural drift*,
  because the fix there is never to widen the tolerance.

## Layout

| path | contents |
|---|---|
| `src/dlkin/grid.py` | spectral grid on `[-L, L)`, Fourier symbols, real circulant operators |
| `src/dlkin/model.py` | `LatticeModel` (damping, substrate, finite-range coupling) and the analytic kink template |
| `src/dlkin/solver.py` | `TravelingWaveSolver`: damped Newton on the collocated advance–delay equation; `init_branch` |
| `src/dlkin/continuation.py` | natural continuation in `c` and pseudo-arclength through the fold |
| `src/dlkin/spectral.py` | `phat`, the exact `sigma'(c)`, `kappa`, `m`, and the eigenvalues of the co-traveling pencil |
| `src/dlkin/monodromy.py` | direct time integration of the linearized lattice: conformal symplecticity and the Floquet multipliers, without the traveling-wave ansatz |
| `src/dlkin/config.py` | YAML config loading, resolution and the `quick:` overlay |
| `src/dlkin/io.py` | result files: the metadata block, key-path access, merged partial runs |
| `configs/`, `scripts/` | experiment configurations and drivers |
| `scripts/check_claims.py` | resolves `claims.yaml` against `data/*.json`; writes the validation report |
| `manuscript/claims.yaml` | the copy of the manifest that travels with the paper |
| `Makefile`, `scripts/reproduce_all.sh` | the reproduction pipeline (the Makefile is primary) |
| `data/`, `figures/`, `tables/`, `reports/` | generated output (tracked, not ignored) |
| `handoff/` | science brief, claims manifest and the reference implementation |

## Experiments

Each driver takes `--config configs/NN_name.yaml`, writes `data/NN_name.json` through
`dlkin.io.save_result`, and prints its tables to stdout.  Every numerical setting --
`L`, `N`, the velocity lists, step sizes, tolerances, seeds and iteration counts -- lives
in the YAML, not in the script.  `--quick` re-runs the same code at the reduced `L`/`N`
of the config's `quick:` section in well under a minute; such a run records
`quick: true` in its metadata and writes to `data/NN_name_quick.json`, because its
results are *not* expected to reproduce `claims.yaml`.

| script | output | what it establishes | runtime |
|---|---|---|---|
| `01_kinetic_relation.py` | `data/01_kinetic.json`, `data/01_kinetic_curve.csv` | the far-field boundary check, the kinetic relation over `c in [0.20, 0.8995]`, and `c_hat1 = 0.8989297` as the zero of the exact `sigma'(c)` | ~19 min |
| `02_identity_convergence.py` | `data/02_identity.json` | the Jordan-chain identity `kappa = -mu sigma' <phat,1>` at `c = 0.89`, its spectral convergence over six `(L, N)`, its maximum error along the branch, and the 1.6e-4 error of a finite-difference `sigma'` | ~11 min |
| `03_threshold.py` | `data/03_threshold.json` | `-kappa/m` against the pencil's own eigenvalue across `c_hat1`; the zeros of `sigma'` and of `nu2` coincide to 3e-7 | ~5 min |
| `04_fold_arclength.py` | `data/04_fold.json` | pseudo-arclength through `c_max = 0.900196`: `<phat,1>` changes sign while `kappa != 0`, so the fold is not a stability change | ~8 min |
| `05_null_vectors.py` | `data/05_nullvec.json`, `data/05_null_profiles.csv` | decay and localization of `phi'` and `phat`; the singular-value gap that says `ker M0` is one-dimensional; and that no exponential weight relates the two | ~50 s |
| `06_spectrum.py` | `data/06_spectrum.json`, `data/06_spectrum_*.csv` | the pencil spectrum either side of the threshold: no real non-trivial eigenvalue at `c = 0.89`, one at `+0.0176` at `c = 0.8995` | ~20 s |
| `07_conformal_symplectic.py` | `data/07_conformal.json` | `M^T J M = e^{-gamma T} J` on a random symmetric `K(t)`, and on the actual FK monodromy by direct integration | ~40 s |
| `08_general_lattices.py` | `data/08_general.json` | `kappa = -f'(c) <phat,1>` in five lattices with different potentials, coupling ranges, substrate strengths and damping | ~50 s |

The heaviest single computation in the repository is the `L300_N6144` convergence case of
experiment 02 -- 6145 x 6145 dense matrices, ~300 MB each, a few minutes on its own.  It
has its own selector so it can be run separately and merged into the result file:

```bash
python scripts/02_identity_convergence.py --only convergence --cases L300_N6144
```

## Status

The numerical core (grid, operators, model, Newton solver, continuation) and the
spectral / biorthogonal layer (`phat`, `kappa`, `m`, pencil eigenvalues) are in place and
are checked against `handoff/reference/{fk,spectral}.py` in `tests/test_reference_smoke.py`
and `tests/test_spectral.py` -- the restructured code reproduces the reference bit for
bit.  Experiments 01-08 are written and their `data/*.json` are committed; all 125
`claims.yaml` claims pass.  Three findings from running them
have been carried back into the handoff bundle -- the relative error of the identity near
`c_hat1`, the spacing convention of the finite-difference comparison, and the rule that
separates an isolated eigenvalue from the discretized essential spectrum -- so `note.tex`,
`SCIENCE_BRIEF.md` and `claims.yaml` now agree with the committed data.

Experiment 07 also carries an **independent cross-check that the manuscript does not
claim**: the monodromy of the linearized lattice, built by integrating in time rather than
through the co-traveling pencil, so that neither computation assumes the other.  It
reproduces `rho = 1`, its conformal partner `e^{-gamma/c}`, and — at `c = 0.8995` — the
unstable multiplier `1.019749` that the pencil predicts from `nu_2 = +0.017591`, to 5e-8.

`scripts/check_claims.py`, the `Makefile` pipeline and the validation report are in
place. Figures and tables are not built yet (`make figures` and `make tables` are
placeholders).
