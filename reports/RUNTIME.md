# Runtime and hardware

Measured on one core (the container used for the final validation run); the `elapsed_sec`
field of each result file carries the figure from the run that produced it.

| script | production | notes |
|---|---|---|
| `01_kinetic_relation.py` | ~19 min | the kinetic curve over `c` in [0.20, 0.8995] at `N=2048` plus the secant for `c_hat1` at `N=4096` |
| `02_identity_convergence.py` | ~8 min | without `L300_N6144`; that case alone is the heaviest computation here (6145x6145 dense, ~2 GB resident) and has its own `--cases` selector |
| `03_threshold.py` | ~5--7 min | nine `N=4096` points, each a dense LU plus shift-invert Arnoldi, and the `N=2048` scan of `m` |
| `04_fold_arclength.py` | ~8 min | two pseudo-arclength runs at `N=2048` plus eigenvalues at the sampled steps |
| `05_null_vectors.py` | ~2 min | measured 118 s; dominated by the `N=2048` SVD |
| `06_spectrum.py` | ~20 s | measured 21 s; two `N=2048` Arnoldi windows |
| `07_conformal_symplectic.py` | ~3 min | measured 187 s; the random-`K` sweep is seconds; the rest is the direct lattice monodromy, whose power iteration is ~250 matrix-vector products per velocity, each one an integration of the linearized lattice over a period |
| `08_general_lattices.py` | ~1.5 min | measured 82 s; five lattices at `N=2048` plus one at `N=4096` |

Total for `make data`: roughly 50 minutes on one core, plus ~10 minutes if `L300_N6144` is
included. Memory peaks at about 1 GB for the `N=4096` work and ~2 GB for `L300_N6144`.

`make data-quick` runs all eight scripts in about a minute at reduced `L`/`N`. Quick results do
not reproduce the claims and are recorded with `quick: true` so the checker refuses to accept
them as evidence.
