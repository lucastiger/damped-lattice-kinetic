"""YAML configuration loading and resolution.

A configuration file is a plain nested mapping with (optional) ``grid``, ``model`` and
``run`` sections::

    grid:  {L: 200.0, N: 4096}
    model: {gamma: 0.1, mu: 1.0, mu2: 0.0, couplings: [[1, 1.0]], template_width: 2.0}
    run:
      name: identity
      c0: 0.88
      c_values: {start: 0.881, stop: 0.8901, step: 0.001, round: 6}
      phat_mode: minus2pi
      tolerances: {newton: 1.0e-12, corrector: 1.0e-10}
      arnoldi: {k: 24, shift: 0.004, tol: 1.0e-10, maxiter: 8000}

:func:`load_config` returns the raw mapping; :func:`resolve` turns ``grid`` and ``model``
into :class:`~dlkin.grid.Grid` and :class:`~dlkin.model.LatticeModel`, and ``run`` into a
:class:`RunConfig` with the velocity list expanded.  :meth:`ResolvedConfig.as_meta` renders
the whole thing as the JSON-safe ``config`` block that :func:`dlkin.io.save_result` demands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml

from .grid import Grid
from .model import LatticeModel

__all__ = [
    "load_config",
    "resolve",
    "resolve_c_values",
    "ResolvedConfig",
    "RunConfig",
]

#: Admissible ``phat`` normalization modes (mirrors ``dlkin.spectral.PHAT_MODES``; kept
#: here as a literal so that config resolution does not import the spectral layer).
_PHAT_MODES = ("minus2pi", "unit")


def resolve_c_values(spec: Any) -> Tuple[float, ...]:
    """Expand a velocity specification into a tuple of floats.

    Accepts a plain list, ``{values: [...]}``, or ``{start, stop, step, round}`` which is
    expanded with :func:`numpy.arange` and rounded (default 6 decimals) -- the same recipe
    the reference drivers use, so that ``c`` keys land on exact decimal strings.
    """
    if spec is None:
        return ()
    if isinstance(spec, Mapping):
        if "values" in spec:
            return tuple(float(value) for value in spec["values"])
        try:
            start, stop = float(spec["start"]), float(spec["stop"])
        except KeyError as exc:
            raise ValueError(
                f"c_values mapping needs 'values', or 'start'/'stop'/'step'; "
                f"missing {exc.args[0]!r}."
            ) from exc
        step = float(spec.get("step", 1e-3))
        digits = int(spec.get("round", 6))
        return tuple(float(value) for value in np.round(np.arange(start, stop, step), digits))
    if isinstance(spec, (str, bytes)):
        raise ValueError(f"c_values must be a list or mapping, got {spec!r}.")
    if isinstance(spec, Sequence):
        return tuple(float(value) for value in spec)
    raise ValueError(f"cannot interpret c_values specification {spec!r}.")


@dataclass(frozen=True)
class RunConfig:
    """The ``run`` section, with the fields the experiment scripts share made explicit.

    Anything not named here survives verbatim in :attr:`params`, so a script can add its
    own keys without touching this class -- and they still end up in the provenance block.
    """

    name: str = ""
    c0: float = 0.88
    c_values: Tuple[float, ...] = ()
    phat_mode: str = "minus2pi"
    tolerances: Dict[str, float] = field(default_factory=dict)
    arnoldi: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phat_mode not in _PHAT_MODES:
            raise ValueError(
                f"run.phat_mode must be one of {_PHAT_MODES}, got {self.phat_mode!r}."
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Look up an unnamed run parameter from :attr:`params`."""
        return self.params.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe rendering, for the provenance block."""
        return {
            "name": self.name,
            "c0": self.c0,
            "c_values": list(self.c_values),
            "phat_mode": self.phat_mode,
            "tolerances": dict(self.tolerances),
            "arnoldi": dict(self.arnoldi),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ResolvedConfig:
    """A configuration with its ``grid``, ``model`` and ``run`` sections instantiated."""

    grid: Grid
    model: LatticeModel
    run: Dict[str, Any] = field(default_factory=dict)
    run_config: RunConfig = field(default_factory=RunConfig)

    def as_meta(self) -> Dict[str, Any]:
        """The full resolved configuration, JSON-safe, for :func:`dlkin.io.save_result`."""
        return {
            "grid": {"L": self.grid.L, "N": self.grid.N, "h": self.grid.h},
            "model": {
                "gamma": self.model.gamma,
                "mu": self.model.mu,
                "mu2": self.model.mu2,
                "couplings": [list(pair) for pair in self.model.couplings],
                "template_width": self.model.template_width,
            },
            "run": self.run_config.as_dict(),
            "run_raw": dict(self.run),
        }


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(data).__name__}."
        )
    return dict(data)


def _resolve_run(run_cfg: Mapping[str, Any]) -> RunConfig:
    known = {"name", "c0", "c_values", "phat_mode", "tolerances", "arnoldi"}
    defaults = RunConfig()
    return RunConfig(
        name=str(run_cfg.get("name", defaults.name)),
        c0=float(run_cfg.get("c0", defaults.c0)),
        c_values=resolve_c_values(run_cfg.get("c_values")),
        phat_mode=str(run_cfg.get("phat_mode", defaults.phat_mode)),
        tolerances={
            str(key): float(value)
            for key, value in dict(run_cfg.get("tolerances", {}) or {}).items()
        },
        arnoldi=dict(run_cfg.get("arnoldi", {}) or {}),
        params={key: value for key, value in run_cfg.items() if key not in known},
    )


def resolve(config: Mapping[str, Any]) -> ResolvedConfig:
    """Instantiate the ``grid``, ``model`` and ``run`` sections."""
    grid_cfg = dict(config.get("grid", {}) or {})
    try:
        grid = Grid(L=float(grid_cfg["L"]), N=int(grid_cfg["N"]))
    except KeyError as exc:
        raise ValueError(f"config section 'grid' is missing key {exc.args[0]!r}.") from exc

    model_cfg = dict(config.get("model", {}) or {})
    defaults = LatticeModel()
    couplings = model_cfg.get("couplings", defaults.couplings)
    model = LatticeModel(
        gamma=float(model_cfg.get("gamma", defaults.gamma)),
        mu=float(model_cfg.get("mu", defaults.mu)),
        mu2=float(model_cfg.get("mu2", defaults.mu2)),
        couplings=tuple((int(j), float(coeff)) for j, coeff in couplings),
        template_width=float(model_cfg.get("template_width", defaults.template_width)),
    )

    run = dict(config.get("run", {}) or {})
    return ResolvedConfig(
        grid=grid, model=model, run=run, run_config=_resolve_run(run)
    )
