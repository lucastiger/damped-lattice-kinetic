"""Result I/O: JSON records with provenance, CSV tables, and claims-style key lookup.

Every JSON file this package writes carries a ``"meta"`` block that says exactly what
produced it -- when, from which commit, on which machine, with which library versions, and
under which resolved configuration (grid, model, run parameters, tolerances, ``phat``
normalization mode, solver settings).  A result without that block is not reproducible, so
:func:`save_result` refuses to write one.

The remaining top-level keys are the payload, addressed by ``claims.yaml``'s ``"a/b/c"``
path syntax through :func:`get`.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import scipy

__all__ = [
    "git_commit",
    "collect_meta",
    "save_result",
    "load_result",
    "save_table_csv",
    "get",
    "timer",
    "to_jsonable",
]

logger = logging.getLogger(__name__)


def git_commit(repo: str | Path | None = None) -> str | None:
    """The current commit sha, or ``None`` if git or the repository is unavailable."""
    directory = Path(repo) if repo is not None else Path(__file__).resolve().parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _git_dirty(repo: str | Path | None = None) -> bool | None:
    directory = Path(repo) if repo is not None else Path(__file__).resolve().parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, complex numbers and dataclasses to JSON types."""
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def collect_meta(meta: Mapping[str, Any], wall_time: float | None = None) -> Dict[str, Any]:
    """Provenance block: environment fields, plus everything the caller supplies.

    ``meta`` must carry a ``"config"`` entry holding the full resolved configuration --
    grid, model, run parameters, tolerances, ``phat`` normalization mode and solver
    settings.  Caller-supplied keys win over the automatic ones, so a replay can pin the
    timestamp or commit if it needs to.
    """
    if not isinstance(meta, Mapping):
        raise TypeError(f"meta must be a mapping, got {type(meta).__name__}.")
    if "config" not in meta:
        raise ValueError(
            "meta must include a 'config' entry with the full resolved configuration "
            "(grid, model, run parameters, tolerances, phat normalization mode, solver "
            "settings). ResolvedConfig.as_meta() produces one."
        )
    automatic: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "wall_time_seconds": wall_time,
    }
    return to_jsonable({**automatic, **dict(meta)})


def save_result(
    path: str | Path,
    payload: Mapping[str, Any],
    meta: Mapping[str, Any],
    wall_time: float | None = None,
    indent: int = 1,
) -> Path:
    """Write ``{"meta": ..., **payload}`` as JSON, creating parent directories."""
    if "meta" in payload:
        raise ValueError("payload must not contain a 'meta' key; pass it as `meta`.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"meta": collect_meta(meta, wall_time=wall_time), **to_jsonable(payload)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=indent, sort_keys=False)
        handle.write("\n")
    logger.debug("wrote %s (%d top-level keys)", path, len(document))
    return path


def load_result(path: str | Path) -> Dict[str, Any]:
    """Read back a file written by :func:`save_result`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_table_csv(
    path: str | Path, rows: Iterable[Mapping[str, Any] | Sequence[Any]], columns: Sequence[str]
) -> Path:
    """Write ``rows`` as CSV with header ``columns``.

    ``rows`` may be mappings (looked up by column name; a missing column is written empty)
    or plain sequences (written positionally, and required to match ``columns`` in length).
    """
    columns = list(columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, row in enumerate(rows):
            if isinstance(row, Mapping):
                writer.writerow([to_jsonable(row.get(name, "")) for name in columns])
            else:
                row = list(row)
                if len(row) != len(columns):
                    raise ValueError(
                        f"row {index} has {len(row)} entries but there are "
                        f"{len(columns)} columns."
                    )
                writer.writerow([to_jsonable(item) for item in row])
    logger.debug("wrote %s", path)
    return path


def get(payload: Mapping[str, Any], key_path: str) -> Any:
    """Nested lookup with ``"/"`` separators, as used by ``claims.yaml``.

    Only ``"/"`` separates levels, so keys may freely contain dots -- ``claims.yaml``
    addresses rows by velocity, e.g. ``"rows/0.89500/kappa"``.  A missing level raises a
    ``KeyError`` naming the full path, the level that failed, and what was available there.
    """
    if not key_path:
        raise KeyError("empty key path")
    current: Any = payload
    walked: list[str] = []
    for part in key_path.split("/"):
        if isinstance(current, Mapping):
            if part not in current:
                where = "/".join(walked) or "<root>"
                available = ", ".join(sorted(map(str, current))) or "<nothing>"
                raise KeyError(
                    f"{key_path!r}: no key {part!r} at {where}; available: {available}"
                )
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                where = "/".join(walked) or "<root>"
                raise KeyError(
                    f"{key_path!r}: {part!r} is not a valid index into the "
                    f"{len(current)}-element sequence at {where}"
                ) from exc
        else:
            where = "/".join(walked) or "<root>"
            raise KeyError(
                f"{key_path!r}: cannot descend into {type(current).__name__} at {where}"
            )
        walked.append(part)
    return current


@contextmanager
def timer():
    """Context manager yielding a zero-argument callable for elapsed seconds.

    ::

        with timer() as elapsed:
            ...
        save_result(path, payload, meta, wall_time=elapsed())
    """
    start = perf_counter()
    finished: list[float] = []

    def elapsed() -> float:
        return finished[0] if finished else perf_counter() - start

    try:
        yield elapsed
    finally:
        finished.append(perf_counter() - start)
