# Reproduction pipeline for the research note.
#
#   make install     editable install with the development extras
#   make test        the fast test suite
#   make test-slow   the slow-marked tests (dense eigensolves, time integration)
#   make data        experiments 01-08 at production settings   (~46 min)
#   make data-quick  the same at reduced L/N                    (~1 min)
#   make figures     (P6)
#   make tables      (P6)
#   make check       resolve every claims.yaml claim, strictly, and write the report
#   make all         data figures tables check
#
# Every target is idempotent.  `make data` prints the git sha and mtime recorded in each
# data file BEFORE and AFTER the run and warns when a file was produced at a different
# commit than HEAD, so a stale file cannot be reused in silence.

# bash with pipefail, so a failing script is not masked by the `tee` it is piped into
SHELL       := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

PYTHON      ?= python
PIP         ?= $(PYTHON) -m pip
PYTEST      ?= $(PYTHON) -m pytest
SCRIPTS     := scripts
DATA        := data
REPORTS     := reports
FIGURES     := figures
TABLES      := tables
REPORT      := $(REPORTS)/validation_report.md
VALIDATION  := $(REPORTS)/validation.json

# experiment -> its primary data file, in run order
EXPERIMENTS := 01_kinetic_relation 02_identity_convergence 03_threshold 04_fold_arclength \
               05_null_vectors 06_spectrum 07_conformal_symplectic 08_general_lattices

.PHONY: all install test test-slow data data-quick figures tables check provenance clean-quick help

help:
	@awk '/^#/{sub(/^# ?/, ""); print; next} {exit}' Makefile

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -q -m "not slow"

test-slow:
	$(PYTEST) -q -m slow

## ---------------------------------------------------------------- data ----
# Each script is deterministic and overwrites its own output, so re-running is safe and
# produces the same numbers; only the metadata timestamp and git sha move.
# Each script's stdout is teed to reports/<name>.log, so the logs never fall out of step
# with the data files they describe.
data: provenance
	@echo "== running experiments 01-08 at production settings =="
	@mkdir -p $(REPORTS)
	@for e in $(EXPERIMENTS); do \
	  echo ""; echo "-- $$e"; \
	  $(PYTHON) -u $(SCRIPTS)/$$e.py 2>&1 | tee $(REPORTS)/$$e.log; \
	done
	@$(MAKE) --no-print-directory provenance

data-quick:
	@echo "== running experiments 01-08 at REDUCED L/N -- results do NOT reproduce claims.yaml =="
	@mkdir -p $(REPORTS)
	@for e in $(EXPERIMENTS); do \
	  echo ""; echo "-- $$e --quick"; \
	  $(PYTHON) -u $(SCRIPTS)/$$e.py --quick 2>&1 | tee $(REPORTS)/$$e.quick.log; \
	done

# Print what is on disk and whether it was produced at this commit.
provenance:
	@$(PYTHON) $(SCRIPTS)/provenance.py

## ------------------------------------------------------- figures/tables ---
figures:
	@echo "make figures: not implemented yet (P6 fills this in); $(FIGURES)/ untouched"

tables:
	@echo "make tables: not implemented yet (P6 fills this in); $(TABLES)/ untouched"

## --------------------------------------------------------------- check ----
check:
	$(PYTHON) $(SCRIPTS)/check_claims.py --strict --compare-manifests \
	  --report $(REPORT) --json $(VALIDATION)

all: data figures tables check

# Quick runs write to data/*_quick.* (gitignored); this removes them.
clean-quick:
	rm -f $(DATA)/*_quick.json $(DATA)/*_quick.csv $(REPORTS)/*.quick.log
