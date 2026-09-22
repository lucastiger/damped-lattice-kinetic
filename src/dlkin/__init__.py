"""dlkin -- traveling kinks in damped, driven lattices.

Reproducibility package for the research note on the kinetic relation ``sigma(c)`` of a
damped, dc-driven Frenkel-Kontorova lattice and its generalizations.

This module is the numerical core: the spectral grid and operators (:mod:`dlkin.grid`),
the lattice model and kink template (:mod:`dlkin.model`), the traveling-wave Newton
solver (:mod:`dlkin.solver`), and continuation in the velocity (:mod:`dlkin.continuation`).

On top of it sit the biorthogonal / spectral layer (:mod:`dlkin.spectral`) -- ``phat``,
the exact ``sigma'(c)``, ``kappa``, ``m`` and the eigenvalues of the co-traveling pencil
-- the direct time integration of the linearized lattice (:mod:`dlkin.monodromy`), and
the result-file writer the experiment scripts emit through (:mod:`dlkin.io`).
"""

from __future__ import annotations

import logging

from .config import ResolvedConfig, apply_overlay, load_config, resolve
from .continuation import (
    ArclengthSystem,
    ContinuationError,
    ContinuationRecord,
    PseudoArclength,
    natural_continuation,
)
from .io import (
    SCHEMA_VERSION,
    build_metadata,
    deep_merge,
    get,
    get_path,
    load_result,
    save_result,
    set_path,
    to_jsonable,
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
from .monodromy import (
    LatticeMonodromy,
    conformal_symplectic_flow_test,
    conformal_symplectic_residual,
    interpolate_psi,
    lattice_monodromy,
    shift_matrix,
    symplectic_form,
)
from .spectral import (
    PHAT_NORMALIZATIONS,
    BranchScalars,
    EigClassification,
    KappaScalars,
    classify_eigs,
    branch_scalars,
    kappa_scalars,
    left_null,
    normalize_phat,
    pencil_eigs,
    positive_real_union,
    real_nontrivial,
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
    "PHAT_NORMALIZATIONS",
    "left_null",
    "normalize_phat",
    "KappaScalars",
    "kappa_scalars",
    "BranchScalars",
    "branch_scalars",
    "pencil_eigs",
    "real_nontrivial",
    "positive_real_union",
    "EigClassification",
    "classify_eigs",
    # monodromy
    "symplectic_form",
    "shift_matrix",
    "conformal_symplectic_residual",
    "conformal_symplectic_flow_test",
    "interpolate_psi",
    "LatticeMonodromy",
    "lattice_monodromy",
    # config
    "load_config",
    "resolve",
    "apply_overlay",
    "ResolvedConfig",
    # result files
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
