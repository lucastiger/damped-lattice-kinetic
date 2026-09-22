# Handoff bundle

Everything Claude Code needs to build the companion repository for the research note.

| file | what it is |
|---|---|
| `note.tex` | the manuscript. Compiles standalone (pdflatex, 18 pp). Figures are drawn as placeholder boxes until `figures/*.pdf` exist. |
| `claims.yaml` | 125 machine-checkable claims: every number quoted in the note, with the JSON key path the experiment scripts must emit, and a tolerance. |
| `SCIENCE_BRIEF.md` | self-contained statement of the model, the identities, the conventions and the full numerical recipe. Read this before writing any code. |
| `reference/fk.py` | Fourier-spectral advance--delay solver for the traveling wave (verbatim from the investigation). |
| `reference/spectral.py` | null vectors, exact `sigma'(c)`, `kappa`, `m`, power balance, shift-invert Arnoldi on the quadratic pencil. |
| `reference/arc.py` | pseudo-arclength continuation in `(psi, sigma, c)` with the scalars tracked through folds. |
| `reference/gen.py` | generalized lattice (second harmonic, next-nearest coupling) used for the scope test. |
| `reference/drivers_session.py` | the drivers behind each table, reconstructed from the session. |
| `reference/verify_core.py`, `verify_extra.py` (+ `.json`, `.log`) | the subset re-run after a sandbox reset; **these outputs are ground truth** and match the note exactly. |

## Provenance and trust

`fk.py`, `spectral.py`, `arc.py`, `gen.py` are the code that produced the reported numbers
and must be treated as the reference definition of the numerical method. `drivers_session.py`
is a consolidation of throwaway driver scripts and has *not* been run end to end; where its
output and `claims.yaml` disagree, `claims.yaml` (and `verify_*.json`) win.

The values in `verify_core.json` / `verify_extra.json` were recomputed from scratch and agree
with the manuscript to all quoted digits. Any new implementation must reproduce them.
