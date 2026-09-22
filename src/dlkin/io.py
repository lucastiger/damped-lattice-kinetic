"""JSON result files: the metadata block, deep-merged partial runs, and key-path access.

Every experiment script writes its results through :func:`save_result`, which wraps the
payload in a file of the shape::

    {
      "metadata": {"schema": 1, "script": ..., "quick": false, ...},
      "<payload keys at the top level>": ...
    }

``handoff/claims.yaml`` addresses results by a ``/``-separated **key path** into that
file -- ``critical/c_hat1``, ``rows/0.89500/kappa``, ``c_max`` -- so the payload sits at
the top level next to ``metadata`` and nothing may be nested under a wrapper key.
:func:`get_path` resolves exactly those paths.

Two conventions the checker depends on:

``metadata.quick``
    ``true`` when the file was produced by a ``--quick`` run at reduced ``L``/``N``.  Such
    a file is *not* expected to reproduce the claims and must be skipped.  It is
    duplicated at the top level as ``quick`` so that a checker can find it either way.
``metadata.sections``
    Which parts of the script produced this file.  Expensive sections can be run
    separately (``--only``) and merged into an existing file by :func:`save_result` with
    ``merge=True``; the metadata of the runs that were merged over is kept under
    ``metadata.previous_runs`` so the provenance of a mixed file stays readable.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import resource
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "save_result",
    "load_result",
    "build_metadata",
    "deep_merge",
    "get",
    "get_path",
    "set_path",
    "to_jsonable",
]

#: Version of the result-file layout described in the module docstring.
SCHEMA_VERSION = 1

#: Reserved top-level keys that a payload may not use (they carry file-level state).
_RESERVED = ("metadata", "quick")


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python so that :mod:`json` can write them."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` over ``base``; mappings merge, everything replaces."""
    out: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def get_path(data: Mapping[str, Any], key: str, separator: str = "/") -> Any:
    """Resolve a ``/``-separated key path, e.g. ``"rows/0.89500/kappa"``.

    Raises
    ------
    KeyError
        If any component is missing, naming the prefix that did resolve -- which is what
        tells a claim check *where* the emitted structure diverged.
    """
    node: Any = data
    walked: list[str] = []
    for part in key.split(separator):
        if not isinstance(node, Mapping) or part not in node:
            where = separator.join(walked) or "<root>"
            available = (
                ", ".join(sorted(map(str, node.keys()))[:12])
                if isinstance(node, Mapping)
                else f"a {type(node).__name__}"
            )
            raise KeyError(
                f"key path {key!r} is missing component {part!r}; {where} holds {available}."
            )
        node = node[part]
        walked.append(part)
    return node


#: Short alias for :func:`get_path`; ``dlkin.io.get(document, "critical/c_hat1")`` is how
#: ``scripts/check_claims.py`` resolves a claim's key path.
get = get_path


def set_path(
    data: MutableMapping[str, Any], key: str, value: Any, separator: str = "/"
) -> None:
    """Assign into a ``/``-separated key path, creating intermediate dicts as needed."""
    parts = key.split(separator)
    node: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise TypeError(
                f"cannot descend into {part!r} of key path {key!r}: it holds a "
                f"{type(child).__name__}, not a mapping."
            )
        node = child
    node[parts[-1]] = value


def _git_commit() -> str | None:
    """Short commit of the working tree, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def build_metadata(
    script: str,
    quick: bool = False,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    sections: Sequence[str] | None = None,
    elapsed_sec: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """The provenance block written next to every payload.

    Records what was run (script, config file and its sha256 prefix, which sections),
    under what (``quick``), with what (interpreter and library versions, git commit) and
    what it cost (wall time and peak resident set).  The timestamp is provenance only: no
    code path anywhere in this package branches on wall-clock time.

    ``peak_rss_mb`` is the high-water mark of the whole process, so it covers the dense
    matrices the run allocated and is what ``reports/RUNTIME.md`` is built from.  It is
    read from ``getrusage``, which reports kibibytes on Linux and bytes on macOS; only the
    Linux reading is converted, so the number is right on the platform the repository is
    run on and is labelled with that platform in ``versions``.
    """
    import scipy  # noqa: PLC0415  -- imported here only to report its version

    meta: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "script": str(script),
        "quick": bool(quick),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sections": list(sections) if sections is not None else None,
        "elapsed_sec": None if elapsed_sec is None else round(float(elapsed_sec), 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "config": None if config_path is None else str(config_path),
        "config_sha256_16": None if config_path is None else _digest(Path(config_path)),
        "versions": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "git_commit": _git_commit(),
    }
    if config is not None:
        meta["config_contents"] = to_jsonable(config)
    if extra:
        meta.update(to_jsonable(extra))
    return meta


def load_result(path: str | Path) -> Dict[str, Any]:
    """Load a result file written by :func:`save_result`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_result(
    path: str | Path,
    payload: Mapping[str, Any],
    script: str,
    quick: bool = False,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    sections: Sequence[str] | None = None,
    elapsed_sec: float | None = None,
    extra: Mapping[str, Any] | None = None,
    merge: bool = False,
    indent: int = 1,
) -> Dict[str, Any]:
    """Write ``payload`` to ``path`` with a metadata block; return what was written.

    Parameters
    ----------
    merge:
        When ``True`` and ``path`` already exists, the payload is deep-merged over the
        file's existing content instead of replacing it, and the sections and
        ``quick`` flags of the two runs are combined (``quick`` is sticky: a file that
        contains *any* quick section is a quick file, because it no longer reproduces the
        claims as a whole).  This is how an expensive ``--only`` section is run on its own
        and folded back into a complete result file.
    """
    path = Path(path)
    for key in payload:
        if key in _RESERVED:
            raise ValueError(
                f"{key!r} is reserved for the result-file wrapper and cannot be a payload "
                "key; claims address the payload directly at the top level."
            )
    body = to_jsonable(payload)
    sections = list(sections) if sections is not None else None
    previous: Iterable[Any] = ()

    if merge and path.is_file():
        existing = load_result(path)
        old_meta = existing.pop("metadata", None)
        existing.pop("quick", None)
        body = deep_merge(existing, body)
        if isinstance(old_meta, Mapping):
            previous = list(old_meta.get("previous_runs") or []) + [
                {k: v for k, v in old_meta.items() if k != "previous_runs"}
            ]
            quick = bool(quick or old_meta.get("quick", False))
            if sections is not None:
                old_sections = old_meta.get("sections") or []
                sections = sorted({*sections, *old_sections})

    meta = build_metadata(
        script=script,
        quick=quick,
        config_path=config_path,
        config=config,
        sections=sections,
        elapsed_sec=elapsed_sec,
        extra=extra,
    )
    if previous:
        meta["previous_runs"] = list(previous)

    document = {"metadata": meta, "quick": bool(quick), **body}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=indent, sort_keys=False)
        handle.write("\n")
    return document
