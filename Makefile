# Reproduction targets for the damped-lattice kinetic-relation study.
#
#   make install    editable install plus the test/figure extras
#   make test       the fast test suite
#   make test-slow  the slow tests too (reference equivalence at N=4096, lattice monodromy)
#   make data       every experiment at production settings  (hours; see reports/RUNTIME.md)
#   make data-quick every experiment at the reduced resolution of each config's `quick:` block
#   make figures    the six manuscript figures from data/
#   make tables     the table fragments and the manuscript diff
#   make check      resolve every claim in the manifest against data/
#   make all        data figures tables check
#
# Every target is idempotent.  `make status` prints the provenance of each result file so a
# stale one is visible before it is trusted.

PY ?= python3
SCRIPTS := scripts
DATA := data

.PHONY: all install test test-slow data data-quick figures tables check status clean-quick help

help:
	@sed -n '2,20p' Makefile

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q -m "not slow"

test-slow:
	$(PY) -m pytest -q

# ---------------------------------------------------------------------- data --
# 01 and 02 are the long ones; 02's L300_N6144 case is the single heaviest computation here
# and is split out so it can be run on its own.
data:
	$(PY) $(SCRIPTS)/01_kinetic_relation.py
	$(PY) $(SCRIPTS)/02_identity_convergence.py
	$(PY) $(SCRIPTS)/03_threshold.py
	$(PY) $(SCRIPTS)/04_fold_arclength.py
	$(PY) $(SCRIPTS)/05_null_vectors.py
	$(PY) $(SCRIPTS)/06_spectrum.py
	$(PY) $(SCRIPTS)/07_conformal_symplectic.py
	$(PY) $(SCRIPTS)/08_general_lattices.py

data-quick:
	$(PY) $(SCRIPTS)/01_kinetic_relation.py --quick
	$(PY) $(SCRIPTS)/02_identity_convergence.py --quick
	$(PY) $(SCRIPTS)/03_threshold.py --quick
	$(PY) $(SCRIPTS)/04_fold_arclength.py --quick
	$(PY) $(SCRIPTS)/05_null_vectors.py --quick
	$(PY) $(SCRIPTS)/06_spectrum.py --quick
	$(PY) $(SCRIPTS)/07_conformal_symplectic.py --quick
	$(PY) $(SCRIPTS)/08_general_lattices.py --quick

figures:
	$(PY) $(SCRIPTS)/09_figures.py

tables:
	$(PY) $(SCRIPTS)/10_tables.py

check:
	$(PY) $(SCRIPTS)/check_claims.py --strict \
	    --report reports/validation_report.md --json reports/validation.json

status:
	@$(PY) -c "import json,glob,os,subprocess;\
head=subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip();\
print('HEAD', head);\
[print(f\"{f:34s} quick={m.get('quick')!s:<5} commit={m.get('git_commit')} {m.get('created','')[:19]} {m.get('elapsed_sec',0):8.1f}s\" + ('   <-- commit differs from HEAD' if m.get('git_commit') not in (head,None) else ''))\
 for f in sorted(glob.glob('data/*.json')) for m in [json.load(open(f)).get('metadata',{})]]"

all: data figures tables check

clean-quick:
	rm -f $(DATA)/*_quick.json $(DATA)/*_quick.csv

.PHONY: manuscript-figures
manuscript-figures: figures
	@mkdir -p manuscript/figures
	cp figures/*.pdf manuscript/figures/
	@echo "copied $$(ls figures/*.pdf | wc -l) figures into manuscript/figures/"
