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
| `src/dlkin/config.py` | YAML config loading and resolution |
| `configs/`, `scripts/` | experiment configurations and drivers |
| `data/`, `figures/`, `tables/`, `reports/` | generated output (tracked, not ignored) |
| `handoff/` | science brief, claims manifest and the reference implementation |

## Status

The numerical core (grid, operators, model, Newton solver, continuation) is in place and
is checked against `handoff/reference/fk.py` in `tests/test_reference_smoke.py`. The
spectral / biorthogonal layer (`phat`, `kappa`, `m`, pencil eigenvalues) and the
experiment scripts are not written yet.
