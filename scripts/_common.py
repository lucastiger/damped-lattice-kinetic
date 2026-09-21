"""Shared plumbing for the experiment drivers: CLI, config overlay, c-paths, tables.

Nothing numerical lives here.  Every ``L``, ``N``, velocity list, step size, tolerance,
seed and iteration count is read from the YAML config named by ``--config``; this module
only turns those settings into the argument lists the ``dlkin`` API wants, and prints the
result tables.

The ``--quick`` flag is implemented by :func:`load`, which deep-merges the config's
top-level ``quick:`` section over the rest of the file (see
:func:`dlkin.config.apply_overlay`).  A quick run therefore differs from a production run
only in what the YAML says, and the resulting JSON records ``quick: true`` so that the
claim checker skips it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from dlkin import ResolvedConfig, apply_overlay, load_config, resolve

__all__ = [
    "build_parser",
    "load",
    "section",
    "output_path",
    "crange",
    "path_to",
    "march",
    "print_table",
    "fmt",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- CLI ----
def build_parser(description: str, default_config: str, default_out: str) -> argparse.ArgumentParser:
    """The common ``--config`` / ``--quick`` / ``--out`` command line."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=default_config, help=f"YAML configuration (default: {default_config})"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run at the reduced L/N of the config's `quick:` section; the results are "
        "NOT expected to reproduce claims.yaml and the JSON records quick=true",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"JSON result file (default: {default_out}, or the same name with a "
        "'_quick' suffix under --quick, so that a quick run never overwrites the "
        "production record)",
    )
    parser.set_defaults(_default_out=default_out)
    return parser


def output_path(args: argparse.Namespace) -> Path:
    """Where the result JSON goes: ``--out`` if given, else the script's default.

    Under ``--quick`` the default gains a ``_quick`` suffix.  A quick run is at reduced
    ``L``/``N`` and does not reproduce the claims, so letting it silently replace
    ``data/NN_name.json`` would poison the reproducibility record; an explicit ``--out``
    still does exactly what it says.
    """
    if args.out is not None:
        return Path(args.out)
    default = Path(args._default_out)
    if args.quick:
        return default.with_name(f"{default.stem}_quick{default.suffix}")
    return default


def load(args: argparse.Namespace) -> Tuple[Dict[str, Any], ResolvedConfig]:
    """Read ``args.config``, apply the ``quick`` overlay if asked, and resolve it.

    Returns both the raw mapping (recorded verbatim in the result metadata, so the file
    says exactly what produced it) and the resolved grid/model/run.
    """
    raw = load_config(args.config)
    if args.quick:
        raw = apply_overlay(raw, "quick")
    else:
        raw = {k: v for k, v in raw.items() if k != "quick"}
    return raw, resolve(raw)


def section(run: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Fetch a required ``run`` subsection, with a message that names the config key."""
    value = run.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"config section 'run.{name}' is required and must be a mapping, "
            f"got {type(value).__name__}."
        )
    return dict(value)


# --------------------------------------------------------------- c paths ----
def crange(start: float, stop: float, step: float, decimals: int = 7) -> List[float]:
    """``[start, start+step, ..., stop]``, inclusive, rounded to ``decimals``.

    Computed as ``start + i*step`` from an integer count rather than by accumulation, so
    the endpoint is hit exactly and no drift creeps into the velocities that label the
    output rows.  ``step`` may be negative for a descending sweep.
    """
    if step == 0.0:
        raise ValueError("crange needs a non-zero step.")
    span = stop - start
    if span != 0.0 and (span > 0.0) != (step > 0.0):
        raise ValueError(
            f"crange({start}, {stop}, {step}) would never terminate: the step points away "
            "from the endpoint."
        )
    n = int(round(span / step))
    return [round(start + i * step, decimals) for i in range(n + 1)]


def path_to(c_from: float, c_to: float, max_step: float, decimals: int = 7) -> List[float]:
    """Velocities from ``c_from`` (exclusive) to ``c_to`` (inclusive), steps <= ``max_step``.

    Used to walk between two points of a sweep whose targets are further apart than the
    continuation can jump in one Newton solve.  The steps are equal, so the walk is
    determined by the two endpoints and the bound alone.
    """
    span = c_to - c_from
    if span == 0.0:
        return []
    n = max(1, int(math.ceil(abs(span) / abs(max_step))))
    return [round(c_from + span * (i + 1) / n, decimals) for i in range(n)]


def march(solver, psi, drive, c_values: Iterable[float], tol: float, maxit: int):
    """Natural continuation that keeps only the last point -- the common "get there" step.

    Thin wrapper over :func:`dlkin.continuation.natural_continuation`; returns
    ``(psi, drive, residual)`` at the final velocity.  An empty ``c_values`` is a no-op.
    """
    from dlkin import natural_continuation  # noqa: PLC0415 -- keeps import cost off --help

    values = [float(c) for c in c_values]
    if not values:
        return psi, drive, float("nan")
    records = natural_continuation(solver, psi, drive, values, tol=tol, maxit=maxit)
    last = records[-1]
    return last.psi, last.drive, last.residual


# ---------------------------------------------------------------- output ----
def fmt(value: Any, spec: str = "") -> str:
    """Format a number for a table cell; ``None`` prints as ``--``."""
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def print_table(
    title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]], specs: Sequence[str] | None = None
) -> None:
    """Print a compact right-aligned table with a rule under the header."""
    specs = list(specs) if specs is not None else [""] * len(headers)
    if len(specs) != len(headers):
        raise ValueError(f"{len(headers)} headers but {len(specs)} format specs.")
    cells = [[fmt(v, s) for v, s in zip(row, specs)] for row in rows]
    widths = [
        max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
        for i, h in enumerate(headers)
    ]
    print(f"\n{title}")
    print("  ".join(str(h).rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
