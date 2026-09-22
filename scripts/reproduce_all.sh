#!/usr/bin/env bash
# Reproduce every number in the note, from a clean checkout.
#
# The Makefile is the primary entry point and this is a thin wrapper over it, for anyone
# who would rather not have make involved.  Each stage is idempotent; the pipeline prints
# the git sha and mtime recorded in every data file before and after, so a stale file
# cannot be reused in silence.
#
#   ./scripts/reproduce_all.sh            install, test, data, check   (~50 min)
#   ./scripts/reproduce_all.sh --quick    the same at reduced L/N      (~2 min, NOT claim-grade)
#   ./scripts/reproduce_all.sh --check    resolve the claims against whatever data is on disk
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAKE=${MAKE:-make}
mode=${1:-full}

case "$mode" in
  --quick)
    echo ">> reduced L/N: the results will NOT reproduce claims.yaml"
    $MAKE install test data-quick
    echo ">> checking (non-strict: quick data is skipped, loudly)"
    python scripts/check_claims.py --compare-manifests \
      --report reports/validation_report.md --json reports/validation.json
    ;;
  --check)
    $MAKE check
    ;;
  full)
    $MAKE install
    $MAKE test
    $MAKE data
    $MAKE check
    ;;
  *)
    echo "usage: $0 [--quick|--check]" >&2
    exit 2
    ;;
esac
