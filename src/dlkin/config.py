"""Minimal YAML configuration loading for the experiment scripts.

A configuration file is a plain nested mapping with (optional) ``grid``, ``model`` and
``run`` sections::

    grid:  {L: 200.0, N: 4096}
    model: {gamma: 0.1, mu: 1.0, mu2: 0.0, couplings: [[1, 1.0]], template_width: 2.0}
    run:   {c0: 0.88, c_values: [...]}

:func:`load_config` returns the raw mapping; :func:`resolve` turns the ``grid`` and
``model`` sections into :class:`~dlkin.grid.Grid` and :class:`~dlkin.model.LatticeModel`
objects and passes ``run`` through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .grid import Grid
from .model import LatticeModel

__all__ = ["load_config", "resolve", "ResolvedConfig"]


@dataclass(frozen=True)
class ResolvedConfig:
    """A configuration with its ``grid`` and ``model`` sections instantiated."""

    grid: Grid
    model: LatticeModel
    run: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}.")
    return dict(data)


def resolve(config: Mapping[str, Any]) -> ResolvedConfig:
    """Instantiate the ``grid`` / ``model`` sections; leave ``run`` as a plain dict."""
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
    return ResolvedConfig(grid=grid, model=model, run=run)
