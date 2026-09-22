#!/usr/bin/env python3
"""Print what data is on disk and whether it was produced at the current commit.

`make data` runs this before and after the pipeline.  The point is that a stale data file
must never be reused in silence: each result file records the git sha it was produced at
and when, and this prints both next to ``HEAD`` so a mismatch is visible without anyone
having to think to look.

A differing sha is not by itself a problem, and the interesting question is not whether the
data is behind ``HEAD`` -- it always is, by one commit, because the commit that records the
data necessarily comes after the run that produced it.  The question is whether anything
that *affects the numbers* changed in between.  So for each file this asks git what changed
under ``src/dlkin/``, ``scripts/`` and ``configs/`` between the sha the file records and
``HEAD``, and separates the two cases:

``behind``   the data predates HEAD but nothing it depends on changed since -- fine
``RECHECK``  something it depends on changed since it was written -- re-run to be sure

Only the second is worth acting on, and "depends on" is meant narrowly: each data file
depends on the library, on ``scripts/_common.py``, and on its own script and config -- not
on the other seven experiments, and not on the tooling (this file, the claims checker, the
runtime report), which consumes data and cannot produce it.  ``src/dlkin/io.py`` is excluded
for the same reason: it serializes results, so it can move a file's ``metadata`` but not any
number a claim is made about.

The guard is deliberately conservative and does not try to judge whether a change was
*substantive*: a dependency that changed is reported, even if the change was a re-export or
a docstring.  It therefore fires by design on the commit that records a run, since that
commit can also touch the library -- the answer there is that a re-run is optional, not that
the guard is wrong.  Silence is the only thing this must never do.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

#: data file -> the script (and config) stem that writes it
EXPECTED = {
    "01_kinetic.json": "01_kinetic_relation",
    "02_identity.json": "02_identity_convergence",
    "03_threshold.json": "03_threshold",
    "04_fold.json": "04_fold_arclength",
    "05_nullvec.json": "05_null_vectors",
    "06_spectrum.json": "06_spectrum",
    "07_conformal.json": "07_conformal_symplectic",
    "08_general.json": "08_general_lattices",
}


#: Shared code every experiment's numbers depend on.  ``io.py`` is excluded: it writes the
#: result file, so it can move the metadata but not the numbers inside it.
SHARED_PATHS = ("src/dlkin", ":!src/dlkin/io.py", "scripts/_common.py")


def deps_changed_since(sha: str, stem: str) -> list[str] | None:
    """What this data file depends on that changed between ``sha`` and ``HEAD``.

    ``None`` when the question cannot be answered -- an unknown sha, a shallow clone, no
    git at all -- in which case the caller treats the file as suspect rather than quietly
    assuming it is fine.
    """
    paths = [*SHARED_PATHS, f"scripts/{stem}.py", f"configs/{stem}.yaml"]
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", f"{sha}..HEAD", "--", *paths],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [line for line in out.stdout.splitlines() if line.strip()]


def head() -> str:
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


def main() -> int:
    sha = head()
    print(f"data provenance (HEAD = {sha})")
    print(f"  {'file':22s} {'sha':>9s} {'mtime (UTC)':>20s} {'wall':>8s} {'peak':>9s}  state")
    stale, missing, quick, behind, unknown = [], [], [], [], []
    for name, stem in EXPECTED.items():
        path = DATA / name
        if not path.is_file():
            missing.append(name)
            print(f"  {name:22s} {'--':>9s} {'MISSING':>20s}")
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8")).get("metadata", {}) or {}
        except (OSError, ValueError) as exc:
            missing.append(name)
            print(f"  {name:22s}  unreadable: {exc}")
            continue
        file_sha = meta.get("git_commit") or "--"
        mtime = dt.datetime.fromtimestamp(
            path.stat().st_mtime, dt.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        secs = meta.get("elapsed_sec")
        rss = meta.get("peak_rss_mb")
        is_quick = bool(meta.get("quick"))

        if file_sha == sha:
            state = "at HEAD"
        elif file_sha == "--":
            state, _ = "no sha", unknown.append(name)
        else:
            changed = deps_changed_since(file_sha, stem)
            if changed is None:
                state, _ = "UNKNOWN", unknown.append(name)
            elif changed:
                state = f"RECHECK ({len(changed)} dependenc{'y' if len(changed) == 1 else 'ies'} changed)"
                stale.append((name, file_sha, changed))
            else:
                state = "behind HEAD, numerics unchanged"
                behind.append(name)
        if is_quick:
            quick.append(name)
            state = "QUICK DATA -- " + state
        print(
            f"  {name:22s} {file_sha:>9s} {mtime:>20s} "
            f"{(f'{secs:.0f}s' if secs else '--'):>8s} "
            f"{(f'{rss:.0f}MB' if rss else '--'):>9s}  {state}"
        )

    if missing:
        print(f"\n  WARNING: {len(missing)} data file(s) missing: {', '.join(missing)}")
    if quick:
        print(
            f"\n  WARNING: {len(quick)} file(s) hold QUICK data (reduced L/N) and reproduce "
            f"nothing: {', '.join(quick)}"
        )
    if unknown:
        print(
            f"\n  WARNING: cannot tell whether {len(unknown)} file(s) are current "
            f"({', '.join(unknown)}); treat them as suspect and re-run."
        )
    if stale:
        print(
            f"\n  WARNING: {len(stale)} file(s) have a dependency that changed after they were "
            "written. The\n           numbers may or may not have moved -- this guard does not "
            "judge whether a change\n           was substantive, only that there was one:"
        )
        for name, file_sha, changed in stale:
            shown = ", ".join(changed[:4]) + (" ..." if len(changed) > 4 else "")
            print(f"           {name} (written at {file_sha}); changed since: {shown}")
        print("           Re-run `make data` to be sure.")
    if behind and not stale:
        print(
            f"\n  {len(behind)} file(s) predate HEAD, but nothing they depend on (the library, "
            "their own script and config) changed since, so they are still valid."
        )
    if not (missing or quick or stale or unknown):
        print(f"\n  all {len(EXPECTED)} data files present, production settings, still valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
