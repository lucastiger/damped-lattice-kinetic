"""dlkin -- traveling kinks in damped, driven lattices.

Reproducibility package for the research note on the kinetic relation ``sigma(c)`` of a
damped, dc-driven Frenkel-Kontorova lattice and its generalizations.

The numerical core is the spectral grid and operators (:mod:`dlkin.grid`), the lattice
model and kink template (:mod:`dlkin.model`), the traveling-wave Newton solver
(:mod:`dlkin.solver`) and continuation in the velocity (:mod:`dlkin.continuation`).

On top of it sit the biorthogonal / Jordan-chain quantities and the quadratic pencil
(:mod:`dlkin.spectral`), the conformal-symplectic and lattice-monodromy checks
(:mod:`dlkin.monodromy`), result I/O with provenance (:mod:`dlkin.io`) and configuration
handling (:mod:`dlkin.config`).
"""

from __future__ import annotations

import logging

from .config import (
    ResolvedConfig,
    RunConfig,
    load_config,
    resolve,
    resolve_c_values,
)
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
from .io import get, load_result, save_result, save_table_csv, timer
from .model import LatticeModel, Template, template_arrays
from .monodromy import (
    conformal_symplectic_flow_test,
    lattice_monodromy,
    symplectic_form,
)
from .solver import ConvergenceError, NewtonResult, TravelingWaveSolver, init_branch
from .spectral import (
    PHAT_MODES,
    biorthogonal_scalars,
    classify_eigs,
    drop_arrays,
    exact_drive_derivative,
    left_null,
    normalize_phat,
    pencil_eigs,
    pencil_matrix,
    scalar_keys,
)

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
    # spectral / biorthogonal
    "PHAT_MODES",
    "left_null",
    "normalize_phat",
    "exact_drive_derivative",
    "biorthogonal_scalars",
    "scalar_keys",
    "drop_arrays",
    "pencil_matrix",
    "pencil_eigs",
    "classify_eigs",
    # monodromy
    "symplectic_form",
    "conformal_symplectic_flow_test",
    "lattice_monodromy",
    # io
    "save_result",
    "load_result",
    "save_table_csv",
    "get",
    "timer",
    # config
    "load_config",
    "resolve",
    "resolve_c_values",
    "ResolvedConfig",
    "RunConfig",
]
