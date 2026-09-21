"""dlkin -- traveling kinks in damped, driven lattices.

Reproducibility package for the research note on the kinetic relation ``sigma(c)`` of a
damped, dc-driven Frenkel-Kontorova lattice and its generalizations.

This module is the numerical core: the spectral grid and operators (:mod:`dlkin.grid`),
the lattice model and kink template (:mod:`dlkin.model`), the traveling-wave Newton
solver (:mod:`dlkin.solver`), and continuation in the velocity (:mod:`dlkin.continuation`).
"""

from __future__ import annotations

import logging

from .config import ResolvedConfig, load_config, resolve
from .continuation import (
    ArclengthSystem,
    ContinuationError,
    ContinuationRecord,
    PseudoArclength,
    natural_continuation,
)
from .grid import (
    Grid,
    circulant_from_symbol,
    spectral_d1,
    sym_d1,
    sym_d2,
    sym_shift_laplacian,
)
from .model import LatticeModel, Template, template_arrays
from .solver import ConvergenceError, NewtonResult, TravelingWaveSolver, init_branch

__version__ = "0.1.0"

# A library must not configure logging for its host; attach a no-op handler so that
# `logging.lastResort` never prints from dlkin's own loggers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    # grid / operators
    "Grid",
    "circulant_from_symbol",
    "sym_d1",
    "sym_d2",
    "sym_shift_laplacian",
    "spectral_d1",
    # model
    "LatticeModel",
    "Template",
    "template_arrays",
    # solver
    "TravelingWaveSolver",
    "NewtonResult",
    "ConvergenceError",
    "init_branch",
    # continuation
    "natural_continuation",
    "ContinuationRecord",
    "ContinuationError",
    "PseudoArclength",
    "ArclengthSystem",
    # config
    "load_config",
    "resolve",
    "ResolvedConfig",
]
