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
pip install -e .
pytest -q
```

## Layout

| path | contents |
|---|---|
| `src/dlkin/grid.py` | spectral grid on `[-L, L)`, Fourier symbols, real circulant operators |
| `src/dlkin/model.py` | `LatticeModel` (damping, substrate, finite-range coupling) and the analytic kink template |
| `src/dlkin/solver.py` | `TravelingWaveSolver`: damped Newton on the collocated advance–delay equation; `init_branch` |
| `src/dlkin/continuation.py` | natural continuation in `c` and pseudo-arclength through the fold |
| `src/dlkin/spectral.py` | `phat`, the exact `sigma'(c)`, `kappa`, `m`, and the eigenvalues of the co-traveling pencil |
| `src/dlkin/config.py` | YAML config loading, resolution and the `quick:` overlay |
| `src/dlkin/io.py` | result files: the metadata block, key-path access, merged partial runs |
| `configs/`, `scripts/` | experiment configurations and drivers |
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
bit.  Experiments 01-04 are written and their `data/*.json` are committed.  Experiments
05-08 (null-vector decay, the full spectrum snapshot, conformal symplecticity, the scope
survey) and `scripts/check_claims.py` are not written yet.
