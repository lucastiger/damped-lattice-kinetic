#!/usr/bin/env python3
"""Resolve every claim in the manifest against the emitted data and report pass/fail.

This is what makes the manuscript machine-checkable.  ``claims.yaml`` lists, for every
number quoted in ``note.tex``, the JSON key path an experiment script must emit and the
tolerance it must meet; this script reads the manifest, resolves each key path with
:func:`dlkin.io.get`, applies the test, and exits non-zero if anything fails.

The manifest lives in two places on purpose.  ``handoff/claims.yaml`` is where it arrived;
``manuscript/claims.yaml`` is the copy that travels with the paper, and it is the one this
script prefers, so that a manuscript sent to a collaborator carries its own checkable
manifest rather than a reference to a bundle they may not have.  ``--compare-manifests``
reports whether the two have drifted apart.

The tests, one per ``kind``:

``value``   ``|x - value| <= abs_tol``, or ``|x - value| / |value| <= rel_tol``
``upper``   ``x <= bound``                       (``x`` is an error or a residual)
``order``   ``|log10|x| - log10|value|| <= decades``
``bool``    ``x is value``                       (identity, not truthiness)
``null``    ``x is None``                        (deliberately unresolved)

Three rules matter more than the arithmetic, and all three exist because a claim that
quietly does not get checked is worse than one that visibly fails:

* **A missing file or an unresolvable key is a FAILURE, never a skip.**  A script that
  stopped emitting a key would otherwise disappear from the report.
* **A data file marked ``quick`` has its claims reported SKIPPED-QUICK, loudly.**  A quick
  run is at reduced ``L``/``N`` and is not expected to reproduce anything; counting its
  claims as passes would be a lie.  ``--strict`` turns those skips into failures, which is
  what CI and ``make check`` use.
* **Claims marked ``exact_replication`` get their own section** if they fail, headed
  "procedural drift".  Those claims depend on reproducing the reference implementation's
  continuation procedure step for step -- a failure there means the procedure drifted, not
  that the science is wrong, and the fix is never to widen the tolerance.

Usage
-----
    python scripts/check_claims.py
    python scripts/check_claims.py --strict --report reports/validation_report.md
    python scripts/check_claims.py --json reports/validation.json --compare-manifests
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dlkin.io import get  # noqa: E402  -- after the path insert

#: Where the manifest is looked for, in order.  The manuscript copy wins.
MANIFEST_SEARCH = ("manuscript/claims.yaml", "handoff/claims.yaml")

PASS = "PASS"
FAIL = "FAIL"
SKIPPED_QUICK = "SKIPPED-QUICK"


@dataclass
class Result:
    """One adjudicated claim."""

    id: str
    where: str
    file: str
    key: str
    kind: str
    expected: str
    obtained: str
    tolerance: str
    status: str
    detail: str = ""
    exact_replication: bool = False
    raw_expected: Any = None
    raw_obtained: Any = None

    @property
    def ok(self) -> bool:
        return self.status == PASS


# ------------------------------------------------------------------ tests ---
def _fmt(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, (int, float)):
        return f"{x:.10g}"
    return str(x)


def apply_test(claim: Dict[str, Any], x: Any) -> Tuple[bool, str, str, str]:
    """Adjudicate one claim.  Returns ``(ok, expected, tolerance, detail)`` as strings."""
    kind = claim["kind"]

    if kind == "null":
        return (x is None), "null", "exact", ("is null" if x is None else f"got {_fmt(x)}")

    if kind == "bool":
        want = claim["value"]
        return (x is want), _fmt(want), "exact", f"{_fmt(x)} vs {_fmt(want)}"

    if x is None:
        return False, _fmt(claim.get("value", claim.get("bound"))), "", (
            f"value is null, but kind '{kind}' needs a number"
        )
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False, _fmt(claim.get("value", claim.get("bound"))), "", (
            f"value is {type(x).__name__}, not a number"
        )
    x = float(x)
    if not math.isfinite(x):
        return False, _fmt(claim.get("value", claim.get("bound"))), "", f"value is {x}"

    if kind == "upper":
        bound = float(claim["bound"])
        return x <= bound, f"<= {bound:.2g}", f"{bound:.2g}", f"{x:.6e} vs bound {bound:.2g}"

    if kind == "order":
        value, decades = abs(float(claim["value"])), float(claim["decades"])
        if x == 0.0:
            return False, f"~{value:.4g}", f"{decades:g} dec", "value is exactly 0; log10 undefined"
        got = abs(math.log10(abs(x)) - math.log10(value))
        return got <= decades, f"~{value:.4g}", f"{decades:g} dec", f"{got:.3f} decades apart"

    if kind == "value":
        value = float(claim["value"])
        if "abs_tol" in claim:
            tol = float(claim["abs_tol"])
            err = abs(x - value)
            return err <= tol, f"{value:.10g}", f"abs {tol:.2g}", f"|err| {err:.3e}"
        if "rel_tol" in claim:
            tol = float(claim["rel_tol"])
            err = abs(x - value) / abs(value) if value else math.inf
            return err <= tol, f"{value:.10g}", f"rel {tol:.2g}", f"rel err {err:.3e}"
        return False, f"{value:.10g}", "", "claim has neither abs_tol nor rel_tol"

    return False, "", "", f"unknown claim kind {kind!r}"


# ------------------------------------------------------------- the run ------
def load_manifest(path: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    """Load the manifest, preferring the manuscript copy over the handoff bundle."""
    if path is not None:
        chosen = Path(path)
        if not chosen.is_file():
            raise SystemExit(f"manifest not found: {chosen}")
    else:
        for candidate in MANIFEST_SEARCH:
            chosen = REPO_ROOT / candidate
            if chosen.is_file():
                break
        else:
            raise SystemExit(
                "no manifest found; looked for " + " then ".join(MANIFEST_SEARCH)
            )
    with chosen.open("r", encoding="utf-8") as handle:
        return chosen, yaml.safe_load(handle)


def manifests_differ() -> Optional[str]:
    """Whether the manuscript copy and the handoff original have drifted apart."""
    a, b = REPO_ROOT / MANIFEST_SEARCH[0], REPO_ROOT / MANIFEST_SEARCH[1]
    if not a.is_file() or not b.is_file():
        return f"only one of {MANIFEST_SEARCH[0]} / {MANIFEST_SEARCH[1]} exists"
    left, right = yaml.safe_load(a.read_text()), yaml.safe_load(b.read_text())
    if left == right:
        return None
    ids_a = {c["id"] for c in left.get("claims", [])}
    ids_b = {c["id"] for c in right.get("claims", [])}
    only_a, only_b = sorted(ids_a - ids_b), sorted(ids_b - ids_a)
    changed = sorted(
        c["id"]
        for c in left.get("claims", [])
        if c["id"] in ids_b
        and c != next(d for d in right["claims"] if d["id"] == c["id"])
    )
    return (
        f"manuscript-only: {only_a or 'none'}; handoff-only: {only_b or 'none'}; "
        f"differing: {changed or 'none'}"
    )


def _cell(text: Any) -> str:
    """Escape a value for a Markdown table cell.

    Details routinely read ``|err| 4.98e-02``, and an unescaped pipe silently breaks the
    row into extra columns -- which in a report sent to people who did not run the pipeline
    would hide the very failure the row exists to show.
    """
    return str(text).replace("|", "\\|")


def _rel(path: Path) -> str:
    """Repo-relative path when it is inside the repo, absolute when it is not.

    ``--manifest`` may legitimately point outside the checkout (a reviewer's copy, a
    temporary file), so the report must not crash trying to relativize it.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def check(manifest: Dict[str, Any], data_root: Path) -> Tuple[List[Result], Dict[str, Any]]:
    """Adjudicate every claim; returns the results and what was learned about the files."""
    claims = manifest.get("claims", [])
    files: Dict[str, Dict[str, Any]] = {}
    results: List[Result] = []

    for claim in claims:
        rel = claim["file"]
        if rel not in files:
            path = data_root / rel
            entry: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
            if entry["exists"]:
                try:
                    entry["document"] = json.loads(path.read_text(encoding="utf-8"))
                    entry["error"] = None
                except (OSError, ValueError) as exc:
                    entry["document"], entry["error"] = None, str(exc)
                    entry["exists"] = False
                else:
                    meta = entry["document"].get("metadata", {}) or {}
                    entry["quick"] = bool(
                        entry["document"].get("quick", False) or meta.get("quick", False)
                    )
                    entry["metadata"] = meta
                    entry["mtime"] = dt.datetime.fromtimestamp(
                        path.stat().st_mtime, dt.timezone.utc
                    ).isoformat(timespec="seconds")
            else:
                entry["error"] = "file not found"
            files[rel] = entry
        entry = files[rel]

        common = dict(
            id=claim["id"],
            where=str(claim.get("where", "")),
            file=rel,
            key=claim["key"],
            kind=str(claim["kind"]),
            exact_replication=bool(claim.get("exact_replication", False)),
            raw_expected=claim.get("value", claim.get("bound")),
        )

        # A missing file is a failure for every claim that points at it -- never a skip.
        if not entry["exists"]:
            results.append(
                Result(
                    expected=_fmt(common["raw_expected"]),
                    obtained="--",
                    tolerance="",
                    status=FAIL,
                    detail=f"data file missing or unreadable: {entry['error']}",
                    **common,
                )
            )
            continue

        # A quick run is at reduced L/N and reproduces nothing; say so loudly.
        if entry.get("quick"):
            results.append(
                Result(
                    expected=_fmt(common["raw_expected"]),
                    obtained="--",
                    tolerance="",
                    status=SKIPPED_QUICK,
                    detail="data file records quick=true (reduced L/N)",
                    **common,
                )
            )
            continue

        try:
            value = get(entry["document"], claim["key"])
        except KeyError as exc:
            results.append(
                Result(
                    expected=_fmt(common["raw_expected"]),
                    obtained="--",
                    tolerance="",
                    status=FAIL,
                    detail=f"key path did not resolve: {exc.args[0]}",
                    **common,
                )
            )
            continue

        ok, expected, tolerance, detail = apply_test(claim, value)
        results.append(
            Result(
                expected=expected,
                obtained=_fmt(value),
                tolerance=tolerance,
                status=PASS if ok else FAIL,
                detail=detail,
                raw_obtained=value if isinstance(value, (int, float, bool, type(None))) else str(value),
                **common,
            )
        )
    return results, files


# --------------------------------------------------------------- reporting --
def diagnose(r: Result) -> Tuple[str, str]:
    """What kind of failure this is, and what would settle it.

    The report goes to people who did not run the pipeline, so a failure has to arrive with
    enough of an account to be actionable.  What can be inferred mechanically is inferred
    here; anything that needs judgement is marked as needing it, rather than left blank.
    """
    if "data file missing" in r.detail:
        return (
            "The experiment that writes this file has not been run at production settings, "
            "or its output was deleted. Nothing about the claim itself is in question.",
            "`make data` (or the single script that writes this file), then `make check`.",
        )
    if "did not resolve" in r.detail:
        return (
            "The data file exists but does not contain this key path: the script that writes "
            "it has stopped emitting the key, or renamed it. This is a break between the "
            "manifest and the script, not a numerical disagreement.",
            "Compare the key against the script's payload; whichever of the two moved, fix "
            "that one -- do not delete the claim.",
        )
    if "quick=true" in r.detail:
        return (
            "The data file holds a `--quick` run at reduced L/N, which reproduces nothing.",
            "`make data` to rebuild at production settings.",
        )
    if r.kind == "null":
        return (
            "The claim says this quantity is deliberately unresolved, but the run produced a "
            "number for it. Either the computation now resolves something it previously could "
            "not, or it is resolving the wrong object.",
            "Look at what was selected and why before touching either side.",
        )
    if r.exact_replication:
        return (
            "Marked `exact_replication`: the value depends on reproducing the reference "
            "continuation procedure step for step, so a mismatch means the procedure drifted "
            "-- a changed step size, starting point or stopping rule.",
            "Diff the procedure against `handoff/reference/`. Widening the tolerance is not a "
            "fix here.",
        )
    return (
        "Numerical disagreement outside the stated tolerance. Needs a human: work out whether "
        "it is resolution, conditioning, or a genuine difference in what is being computed.",
        "Re-run the owning script; if it reproduces, compare against `handoff/reference/` at "
        "the same settings and report the margin rather than adjusting the tolerance.",
    )


def _summary(results: Sequence[Result], strict: bool) -> Dict[str, int]:
    skipped = sum(r.status == SKIPPED_QUICK for r in results)
    failed = sum(r.status == FAIL for r in results)
    return {
        "total": len(results),
        "passed": sum(r.ok for r in results),
        "failed": failed + (skipped if strict else 0),
        "skipped_quick": 0 if strict else skipped,
        "exact_replication_failures": sum(
            r.status == FAIL and r.exact_replication for r in results
        ),
    }


def write_report(
    path: Path,
    results: Sequence[Result],
    files: Dict[str, Any],
    manifest_path: Path,
    strict: bool,
    drift: Optional[str],
) -> None:
    """The human-readable validation report, grouped by manuscript location."""
    counts = _summary(results, strict)
    unresolved = [r for r in results if r.status == FAIL]
    drifted = [r for r in unresolved if r.exact_replication]
    skipped = [r for r in results if r.status == SKIPPED_QUICK]

    elapsed = sum(
        float((f.get("metadata") or {}).get("elapsed_sec") or 0.0)
        for f in files.values()
        if f.get("exists")
    )
    for f in files.values():
        for previous in ((f.get("metadata") or {}).get("previous_runs") or []):
            elapsed += float(previous.get("elapsed_sec") or 0.0)
    head = git_sha()

    lines: List[str] = []
    lines.append("# Validation report")
    lines.append("")
    lines.append(
        f"`{_rel(manifest_path)}` resolved against `data/*.json`. "
        "Every number quoted in the manuscript is listed here with the key path that "
        "produced it and the tolerance it had to meet."
    )
    lines.append("")

    # -- UNRESOLVED goes at the very top, before anything reassuring ----------------
    if unresolved:
        lines.append("## UNRESOLVED")
        lines.append("")
        lines.append(
            f"**{len(unresolved)} claim(s) did not pass.** The manifest and the manuscript "
            "have NOT been edited to accommodate them. Each is listed with what it expected, "
            "what it got, and what would be needed to settle it."
        )
        lines.append("")
        for r in unresolved:
            lines.append(f"### {r.id} — `{r.file}:{r.key}`")
            lines.append("")
            lines.append(f"- **where**: {r.where or '(unattributed)'}")
            lines.append(f"- **expected**: {r.expected}  (tolerance: {r.tolerance or 'n/a'})")
            lines.append(f"- **obtained**: {r.obtained}")
            lines.append(f"- **detail**: {r.detail}")
            if r.exact_replication:
                lines.append(
                    "- **exact_replication**: yes — see the procedural-drift section below"
                )
            why, needed = diagnose(r)
            lines.append(f"- **diagnosis**: {why}")
            lines.append(f"- **to resolve**: {needed}")
            lines.append("")
    else:
        lines.append("## UNRESOLVED")
        lines.append("")
        lines.append("None. Every claim that was checked passed at its stated tolerance.")
        lines.append("")

    # -- summary -------------------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| claims in manifest | {counts['total']} |")
    lines.append(f"| passed | {counts['passed']} |")
    lines.append(f"| failed | {counts['failed']} |")
    lines.append(f"| skipped (quick data) | {counts['skipped_quick']} |")
    lines.append(f"| accounted for | {counts['passed'] + counts['failed'] + counts['skipped_quick']} |")
    lines.append(f"| strict mode | {'on' if strict else 'off'} |")
    lines.append(f"| manifest | `{_rel(manifest_path)}` |")
    lines.append(f"| manifest drift | {drift or 'none — manuscript copy matches the handoff original'} |")
    lines.append(f"| git sha (HEAD) | `{head}` |")
    lines.append(
        f"| generated | {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} |"
    )
    lines.append(f"| pipeline wall time (from data metadata) | {elapsed / 60.0:.1f} min |")
    lines.append("")

    # -- the data files it read ----------------------------------------------------
    lines.append("### Data files")
    lines.append("")
    lines.append("| file | exists | quick | git sha | matches HEAD | mtime (UTC) | wall time | peak RSS |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rel in sorted(files):
        f = files[rel]
        meta = f.get("metadata") or {}
        sha = meta.get("git_commit") or "--"
        secs = meta.get("elapsed_sec")
        rss = meta.get("peak_rss_mb")
        lines.append(
            f"| `{rel}` | {'yes' if f['exists'] else '**NO**'} | "
            f"{'**yes**' if f.get('quick') else 'no'} | `{sha}` | "
            f"{'yes' if sha == head else '**no**'} | {f.get('mtime', '--')} | "
            f"{f'{secs:.0f} s' if secs else '--'} | {f'{rss:.0f} MB' if rss else '--'} |"
        )
    lines.append("")

    # -- procedural drift ----------------------------------------------------------
    if drifted:
        lines.append("## Procedural drift, investigate before editing anything")
        lines.append("")
        lines.append(
            "These claims are marked `exact_replication: true`. They depend on reproducing "
            "the reference implementation's continuation procedure step for step. A failure "
            "here means the procedure drifted — a changed step size, a changed starting "
            "point, a changed stopping rule — **not** that the science is wrong, and the "
            "fix is never to widen the tolerance."
        )
        lines.append("")
        lines.append("| id | where | key | expected | obtained | detail |")
        lines.append("|---|---|---|---|---|---|")
        for r in drifted:
            lines.append(
                f"| {r.id} | {_cell(r.where)} | `{r.file}:{r.key}` | {_cell(r.expected)} | "
                f"{_cell(r.obtained)} | {_cell(r.detail)} |"
            )
        lines.append("")

    # -- skipped -------------------------------------------------------------------
    if skipped:
        lines.append("## Skipped — quick data")
        lines.append("")
        lines.append(
            f"**{len(skipped)} claim(s) were not checked** because the data file they point "
            "at records `quick: true`. A quick run is at reduced `L`/`N` and is not expected "
            "to reproduce anything. Re-run the pipeline at production settings (`make data`) "
            "and check again; `--strict` (which `make check` uses) turns these into failures."
        )
        lines.append("")
        for rel in sorted({r.file for r in skipped}):
            ids = ", ".join(r.id for r in skipped if r.file == rel)
            lines.append(f"- `{rel}`: {ids}")
        lines.append("")

    # -- the full table, grouped by manuscript location ----------------------------
    lines.append("## All claims, by manuscript location")
    lines.append("")
    order: List[str] = []
    for r in results:
        if r.where not in order:
            order.append(r.where)
    for where in order:
        group = [r for r in results if r.where == where]
        bad = sum(r.status != PASS for r in group)
        heading = where or "(unattributed)"
        noun = "claim" if len(group) == 1 else "claims"
        lines.append(
            f"### {heading}  ({len(group)} {noun}"
            f"{', ' + str(bad) + ' NOT PASSING' if bad else ''})"
        )
        lines.append("")
        lines.append("| id | file / key | expected | obtained | tolerance | status |")
        lines.append("|---|---|---|---|---|---|")
        for r in group:
            mark = {PASS: "PASS", FAIL: "**FAIL**", SKIPPED_QUICK: "_skip-quick_"}[r.status]
            lines.append(
                f"| {r.id} | `{r.file.replace('data/', '')}:{r.key}` | {_cell(r.expected)} | "
                f"{_cell(r.obtained)} | {_cell(r.tolerance)} | {mark} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, results: Sequence[Result], files: Dict[str, Any], strict: bool) -> None:
    """Machine-readable results, for CI."""
    payload = {
        "schema": 1,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "strict": strict,
        "summary": _summary(results, strict),
        "files": {
            rel: {
                "exists": f["exists"],
                "quick": f.get("quick"),
                "git_commit": (f.get("metadata") or {}).get("git_commit"),
                "elapsed_sec": (f.get("metadata") or {}).get("elapsed_sec"),
                "peak_rss_mb": (f.get("metadata") or {}).get("peak_rss_mb"),
                "mtime": f.get("mtime"),
            }
            for rel, f in files.items()
        },
        "claims": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="claims manifest (default: " + " then ".join(MANIFEST_SEARCH) + ")",
    )
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT), help="directory the claims' `file` paths are relative to"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat SKIPPED-QUICK as failures (what CI and `make check` use)",
    )
    parser.add_argument("--report", default=None, help="write the Markdown report here")
    parser.add_argument("--json", dest="json_out", default=None, help="write machine-readable results here")
    parser.add_argument(
        "--compare-manifests",
        action="store_true",
        help="also check the manuscript copy against the handoff original",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = parser.parse_args(argv)

    manifest_path, manifest = load_manifest(args.manifest)
    drift = manifests_differ() if args.compare_manifests else None
    results, files = check(manifest, Path(args.data_root))
    counts = _summary(results, args.strict)

    if not args.quiet:
        for r in results:
            if r.status == PASS:
                continue
            tag = "FAIL " if r.status == FAIL else "SKIP "
            print(f"{tag} {r.id:16s} {r.file}:{r.key}  {r.detail}")

    skipped = [r for r in results if r.status == SKIPPED_QUICK]
    if skipped:
        bad = sorted({r.file for r in skipped})
        print()
        print("!" * 78)
        print(
            f"WARNING: {len(skipped)} claim(s) were NOT CHECKED because their data file "
            "records quick=true."
        )
        print(f"         Files: {', '.join(bad)}")
        print("         A quick run is at reduced L/N and reproduces nothing. Run `make data`.")
        if args.strict:
            print("         --strict is on, so these count as FAILURES.")
        print("!" * 78)

    drifted = [r for r in results if r.status == FAIL and r.exact_replication]
    if drifted:
        print()
        print("PROCEDURAL DRIFT, INVESTIGATE BEFORE EDITING ANYTHING")
        for r in drifted:
            print(f"  {r.id:16s} {r.file}:{r.key}  expected {r.expected}, got {r.obtained}")

    if args.report:
        write_report(Path(args.report), results, files, manifest_path, args.strict, drift)
        print(f"\nwrote {args.report}")
    if args.json_out:
        write_json(Path(args.json_out), results, files, args.strict)
        print(f"wrote {args.json_out}")

    if drift:
        print(f"\nmanifest drift: {drift}")

    print(
        f"\n{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['skipped_quick']} skipped (quick) of {counts['total']} claims "
        f"[{_rel(manifest_path)}]"
    )
    accounted = counts["passed"] + counts["failed"] + counts["skipped_quick"]
    if accounted != counts["total"]:
        print(f"INTERNAL ERROR: {accounted} accounted for, {counts['total']} in manifest")
        return 2
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
