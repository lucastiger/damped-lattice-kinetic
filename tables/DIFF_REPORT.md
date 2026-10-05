# Table diff: manuscript against regenerated data

Manuscript: `manuscript/note.tex`.

Each table's numeric cells are extracted from the typeset `tabular` and from the freshly generated fragment, then compared as sets at the precision the manuscript prints. The claims checker verifies that the data is right; this verifies that what was typed into the paper matches the data.

**Every regenerated value has a counterpart in the typeset tables.**

## tab_valid (`tab:valid`)

- regenerated cells: 3; numbers found in the typeset table: 6
- regenerated values with no match in the manuscript: **0**

## tab_conv (`tab:conv`)

- regenerated cells: 30; numbers found in the typeset table: 30
- regenerated values with no match in the manuscript: **0**

## tab_threshold (`tab:threshold`)

- regenerated cells: 36; numbers found in the typeset table: 60
- regenerated values with no match in the manuscript: **0**

## tab_fold (`tab:fold`)

- regenerated cells: 30; numbers found in the typeset table: 40
- regenerated values with no match in the manuscript: **0**

## tab_decay (`tab:decay`)

- regenerated cells: 14; numbers found in the typeset table: 14
- regenerated values with no match in the manuscript: **0**

## tab_gen (`tab:gen`)

- regenerated cells: 25; numbers found in the typeset table: 36
- regenerated values with no match in the manuscript: **0**
