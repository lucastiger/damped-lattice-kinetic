#!/usr/bin/env python3
"""Print what data is on disk and whether it was produced at the current commit.

`make data` runs this before and after the pipeline.  The point is that a stale data file
must never be reused in silence: each result file records the git sha it was produced at
and when, and this prints both next to ``HEAD`` so a mismatch is visible without anyone
having to think to look.

A differing sha is a warning, not an error.  It is normal immediately after a commit that
did not touch the numerics, and it is exactly what you want to see if you have just changed
a script and not yet re-run it.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

EXPECTED = [
    "01_kinetic.json",
    "02_identity.json",
    "03_threshold.json",
    "04_fold.json",
    "05_nullvec.json",
    "06_spectrum.json",
    "07_conformal.json",
    "08_general.json",
]


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
    print(f"  {'file':22s} {'sha':>9s} {'mtime (UTC)':>20s} {'wall':>8s} {'peak':>9s}  quick")
    stale, missing, quick = [], [], []
    for name in EXPECTED:
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
        flag = ""
        if file_sha != sha:
            stale.append(name)
            flag = "  <- produced at a different commit"
        if is_quick:
            quick.append(name)
            flag += "  <- QUICK DATA"
        print(
            f"  {name:22s} {file_sha:>9s} {mtime:>20s} "
            f"{(f'{secs:.0f}s' if secs else '--'):>8s} "
            f"{(f'{rss:.0f}MB' if rss else '--'):>9s}  {'yes' if is_quick else 'no'}{flag}"
        )

    if missing:
        print(f"\n  WARNING: {len(missing)} data file(s) missing: {', '.join(missing)}")
    if quick:
        print(
            f"\n  WARNING: {len(quick)} file(s) hold QUICK data (reduced L/N) and reproduce "
            f"nothing: {', '.join(quick)}"
        )
    if stale:
        print(
            f"\n  WARNING: {len(stale)} file(s) were produced at a commit other than HEAD "
            f"({sha}): {', '.join(stale)}"
        )
        print("           Run `make data` if the numerics changed since then.")
    if not (missing or quick or stale):
        print(f"\n  all {len(EXPECTED)} data files present, production settings, built at HEAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
