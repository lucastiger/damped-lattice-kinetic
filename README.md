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
| `src/dlkin/spectral.py` | left null vector `phat`, exact `sigma'(c)`, `kappa`, `m`, the quadratic pencil and its eigenvalues |
| `src/dlkin/monodromy.py` | conformal symplecticity (R1) and an independent lattice-monodromy cross-check |
| `src/dlkin/io.py` | JSON results with provenance, CSV tables, `claims.yaml`-style key lookup |
| `src/dlkin/config.py` | YAML config loading and resolution |
| `configs/`, `scripts/` | experiment configurations and drivers |
| `data/`, `figures/`, `tables/`, `reports/` | generated output (tracked, not ignored) |
| `handoff/` | science brief, claims manifest and the reference implementation |

## Tests

```bash
pytest -q                 # everything, ~2 min
pytest -q -m "not slow"   # the fast suite, ~20 s
pytest -q -m slow         # the N=4096 ground-truth cases, ~2 min
```

`tests/test_reference_smoke.py` and `tests/test_reference_equivalence.py` gate everything
downstream: they run `handoff/reference/*.py` side by side with `dlkin` and require
agreement, and reproduce the ground-truth values in `handoff/reference/verify_*.json`.

## The `phat` normalization

`kappa` and `<phat, 1>` depend on which normalization of the left null vector is in force
— `"minus2pi"` (`<phat,1>_h = -2 pi`, used for the identity and threshold tables) or
`"unit"` (Euclidean norm 1, used for the fold and scope tables, and grid-dependent).
Ratios such as `-kappa/m` do not. `normalize_phat` and `biorthogonal_scalars` therefore
take `mode` / `phat_mode` as a **required** argument, and every record carries it back.

## Status

The numerical core and the spectral / biorthogonal layer are in place. The experiment
scripts under `scripts/` and the configurations under `configs/` are not written yet.
