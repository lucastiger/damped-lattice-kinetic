# Building the note

```bash
make manuscript-figures   # copy figures/*.pdf into manuscript/figures/
cd manuscript
pdflatex note.tex && pdflatex note.tex    # twice, for the table of contents and refs
```

`note.tex` is standalone: it needs no local class or style files, only a TeX Live
installation with `amsmath`, `booktabs`, `hyperref` and `graphicx`. It compiles whether or not
the figures exist -- `\figinclude` falls back to a labelled placeholder box -- so a missing
figure is visible in the PDF rather than a compilation failure.

The numbers in the tables are checked against `../data/` by `python scripts/check_claims.py`
(every value, with a tolerance, listed in `claims.yaml`) and against the regenerated table
fragments by `python scripts/10_tables.py` (which writes `../tables/DIFF_REPORT.md`).
