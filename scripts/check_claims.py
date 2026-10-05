#!/usr/bin/env python3
"""Resolve every number quoted in the manuscript against the data this repository produced.

``claims.yaml`` lists, for each number in ``note.tex``, the result file and key path that is
supposed to produce it and the tolerance it must meet.  This script resolves all of them and
exits non-zero if any fails.  It is the single answer to "is this number reproducible?", and
it is deliberately unforgiving:

* a missing file or an unresolvable key is a FAILURE, never a skip -- a claim that quietly
  vanishes is worse than one that visibly breaks;
* claims against a result file whose metadata records ``quick: true`` are reported as
  SKIPPED-QUICK with a warning, and ``--strict`` turns those into failures, because quick runs
  are at reduced resolution and cannot reproduce production numbers;
* claims marked ``exact_replication`` in the manifest depend on reproducing a particular
  continuation path step for step; they are reported in their own section, since a failure
  there means the procedure drifted rather than that the science is wrong.

Kinds of test
    ``value``  ``|x - value| <= abs_tol``, or ``|x - value| <= rel_tol * |value|``
    ``upper``  ``x <= bound`` (for errors and residuals)
    ``order``  ``|log10|x| - log10|value|| <= decades`` (for quantities quoted to an order)
    ``bool``   ``x == value``
    ``null``   ``x is None`` (a quantity deliberately not resolved)

Usage
    python scripts/check_claims.py [--claims PATH] [--strict]
                                   [--report reports/validation_report.md]
                                   [--json reports/validation.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIM_LOCATIONS = ("manuscript/claims.yaml", "handoff/claims.yaml")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def find_claims(explicit: str | None) -> Path:
    """The manuscript's copy wins, so the manifest travels with the paper."""
    if explicit:
        return Path(explicit)
    for candidate in CLAIM_LOCATIONS:
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit(f"no claims manifest found in {CLAIM_LOCATIONS}")


def get_path(data: Any, key: str) -> Any:
    """Nested lookup with ``/`` separators; keys may contain dots (``rows/0.89500/kappa``)."""
    cur = data
    for part in key.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def evaluate(claim: Dict[str, Any], value: Any) -> Tuple[bool, str]:
    """Apply one claim's test; returns ``(ok, how)`` where ``how`` describes the comparison."""
    kind = claim["kind"]
    if kind == "null":
        return value is None, "is null"
    if value is None:
        return False, "value is null"
    if kind == "bool":
        return bool(value) == bool(claim["value"]), f"== {claim['value']}"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False, f"not numeric: {value!r}"
    if kind == "value":
        target = float(claim["value"])
        if "abs_tol" in claim:
            tol = float(claim["abs_tol"])
            return abs(x - target) <= tol, f"|x - {target:g}| <= {tol:g}"
        tol = float(claim["rel_tol"])
        return abs(x - target) <= tol * abs(target), f"|x - {target:g}| <= {tol:g}|{target:g}|"
    if kind == "upper":
        bound = float(claim["bound"])
        return x <= bound, f"x <= {bound:g}"
    if kind == "order":
        target, decades = float(claim["value"]), float(claim["decades"])
        if x == 0.0:
            return False, "x == 0, no order of magnitude"
        return abs(math.log10(abs(x)) - math.log10(abs(target))) <= decades, (
            f"within {decades} decades of {target:g}"
        )
    return False, f"unknown kind {kind!r}"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claims", default=None, help="claims manifest (default: manuscript/ then handoff/)")
    parser.add_argument("--strict", action="store_true", help="treat quick-run data as a failure")
    parser.add_argument("--report", default=None, help="write a Markdown report here")
    parser.add_argument("--json", dest="json_out", default=None, help="write a machine-readable summary here")
    args = parser.parse_args(argv)

    manifest_path = find_claims(args.claims)
    manifest = yaml.safe_load(manifest_path.read_text())
    claims = manifest["claims"]

    files: Dict[str, Any] = {}
    quick_files: Dict[str, bool] = {}
    for claim in claims:
        name = claim["file"]
        if name in files:
            continue
        path = REPO_ROOT / name
        if path.exists():
            payload = json.loads(path.read_text())
            files[name] = payload
            meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            quick_files[name] = bool(meta.get("quick", payload.get("quick", False)))
        else:
            files[name] = None
            quick_files[name] = False

    results: List[Dict[str, Any]] = []
    for claim in claims:
        row = {
            "id": claim["id"],
            "where": claim.get("where", ""),
            "file": claim["file"],
            "key": claim["key"],
            "kind": claim["kind"],
            "expected": claim.get("value", claim.get("bound")),
            "exact_replication": bool(claim.get("exact_replication", False)),
            "note": claim.get("note", ""),
        }
        payload = files[claim["file"]]
        if payload is None:
            row.update(status=FAIL, obtained=None, how="result file does not exist")
            results.append(row)
            continue
        try:
            value = get_path(payload, claim["key"])
        except KeyError:
            row.update(status=FAIL, obtained=None, how="key not present in the result file")
            results.append(row)
            continue
        ok, how = evaluate(claim, value)
        row["obtained"] = value
        row["how"] = how
        if quick_files[claim["file"]] and not ok:
            row["status"] = FAIL if args.strict else SKIP
            row["how"] = how + "  (quick-run data: reduced resolution)"
        else:
            row["status"] = PASS if ok else FAIL
        results.append(row)

    passed = [r for r in results if r["status"] == PASS]
    failed = [r for r in results if r["status"] == FAIL]
    skipped = [r for r in results if r["status"] == SKIP]
    drift = [r for r in failed if r["exact_replication"]]
    science = [r for r in failed if not r["exact_replication"]]

    print(f"claims: {len(results)}   passed: {len(passed)}   failed: {len(failed)}   skipped: {len(skipped)}")
    for row in science:
        print(f"  FAIL {row['id']:<16} {row['file']}:{row['key']}")
        print(f"       expected {row['expected']!r} ({row['how']}), obtained {row['obtained']!r}")
    for row in drift:
        print(f"  DRIFT {row['id']:<15} {row['file']}:{row['key']} -- exact-replication claim")
        print(f"       expected {row['expected']!r} ({row['how']}), obtained {row['obtained']!r}")
    for row in skipped:
        print(f"  SKIP {row['id']:<16} quick-run data at {row['file']}")

    if args.report:
        _write_report(Path(args.report), results, manifest_path, args.strict)
        print(f"wrote {args.report}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": git_commit(),
                    "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                    "counts": {"total": len(results), "passed": len(passed),
                               "failed": len(failed), "skipped": len(skipped)},
                    "claims": results,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"wrote {args.json_out}")

    return 1 if failed else 0


def _write_report(path: Path, results, manifest_path: Path, strict: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = [r for r in results if r["status"] == PASS]
    failed = [r for r in results if r["status"] == FAIL]
    skipped = [r for r in results if r["status"] == SKIP]
    drift = [r for r in failed if r["exact_replication"]]
    science = [r for r in failed if not r["exact_replication"]]

    lines = [
        "# Validation report",
        "",
        f"- generated: {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}",
        f"- commit: `{git_commit()}`",
        f"- manifest: `{manifest_path.relative_to(REPO_ROOT)}`",
        f"- mode: {'strict' if strict else 'normal'}",
        "",
        f"**{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped "
        f"of {len(results)} claims.**",
        "",
    ]
    if science:
        lines += ["## UNRESOLVED", "",
                  "Claims that do not reproduce. Neither the manifest nor the manuscript has "
                  "been edited to accommodate them.", "",
                  "| id | where | key | expected | obtained | test |", "|---|---|---|---|---|---|"]
        lines += [
            f"| `{r['id']}` | {r['where']} | `{r['file']}:{r['key']}` | `{r['expected']}` | "
            f"`{r['obtained']}` | {r['how']} |" for r in science
        ]
        lines.append("")
    if drift:
        lines += ["## Procedural drift -- investigate before editing anything", "",
                  "These claims depend on reproducing a particular continuation path step for "
                  "step. A failure means the procedure moved, not that the identity is wrong.",
                  "", "| id | key | expected | obtained |", "|---|---|---|---|"]
        lines += [f"| `{r['id']}` | `{r['key']}` | `{r['expected']}` | `{r['obtained']}` |" for r in drift]
        lines.append("")
    if skipped:
        lines += ["## Skipped (quick-run data)", "",
                  "Re-run the corresponding script without `--quick` to check these.", ""]
        lines += [f"- `{r['id']}` ({r['file']})" for r in skipped] + [""]

    by_place: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_place.setdefault(r["where"] or "(unplaced)", []).append(r)
    lines += ["## All claims, by manuscript location", ""]
    for place, rows in by_place.items():
        lines += [f"### {place}", "",
                  "| id | key | expected | obtained | test | status |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            obtained = r.get("obtained")
            shown = f"{obtained:.10g}" if isinstance(obtained, float) else repr(obtained)
            lines.append(
                f"| `{r['id']}` | `{r['key']}` | `{r['expected']}` | `{shown}` | {r['how']} | "
                f"{r['status']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip("\n") + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
