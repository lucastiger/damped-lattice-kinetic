#!/usr/bin/env python3
"""Experiment 08 -- the identity beyond Frenkel-Kontorova (Proposition 7.1).

For a damped lattice with ADDITIVE drive ``f``, a smooth ``2 pi``-periodic onsite potential
``V'(u) = mu sin u + mu2 sin 2u`` and any symmetric finite-range coupling
``sum_j C_j Delta_j``, the derivation of R4 goes through unchanged and gives

    kappa = <phat, M1 phi'>_h = -f'(c) <phat, 1>_h .

The proof uses translation invariance, the symmetry of the coupling and uniform damping --
and nothing else.  Changing the substrate (a second harmonic, of either sign) and the coupling
range (next-nearest neighbours) is therefore the cheapest sharp test of its scope: if the
identity held only for ``sin u`` with nearest-neighbour coupling, something in the derivation
would be wrong.

Conventions, which differ deliberately from the other experiments:

* ``phat`` is normalized in the *Euclidean* norm of the grid vector, not by ``<phat,1> = -2 pi``.
  So ``kappa`` and ``<phat,1>`` separately depend on ``N`` -- halving the grid spacing scales
  both by ``sqrt(2)`` -- while their relation does not.  The reported ``relerr`` is
  normalization-free.
* The branch is reached by the *single-stage* initialization of the reference ``gen.py``
  (``psi = 0``, ``f = 0.5`` at ``c = 0.55``, fixed template width), not the two-stage
  ``init_branch``.  These rows are the reference values of Table 7 of the note and that is the
  path they were computed along.

``mu2 = +0.35`` is attempted and is expected to fail: naive continuation from ``psi = 0`` does
not converge for it.  That is a failure of the initialization, not of the identity, and it is
recorded with its residual rather than rescued -- the manuscript reports it the same way.

``refinement`` repeats the worst row (``mu2 = -0.5``, relative error ~4e-6 at ``N = 2048``) at
``N = 4096``.  If that error is a resolution effect it must collapse; if it were a failure of
the identity it would not.

Usage
    python scripts/08_general_lattices.py [--config ...] [--quick] [--cases fk,nnn]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, load, output_path, print_table, section

from dlkin import ConvergenceError, Grid, LatticeModel, TravelingWaveSolver, save_result
from dlkin.spectral import branch_scalars

SCRIPT = "scripts/08_general_lattices.py"


def _model(cfg, spec) -> LatticeModel:
    """A model for one case: the case's `model:` block over the file's defaults."""
    base = {
        "mu": cfg.model.mu,
        "gamma": cfg.model.gamma,
        "mu2": cfg.model.mu2,
        "couplings": cfg.model.couplings,
        "template_width": cfg.model.template_width,
    }
    base.update({k: v for k, v in dict(spec.get("model", {})).items()})
    base["couplings"] = tuple((int(j), float(a)) for j, a in base["couplings"])
    return LatticeModel(**base)


def _solve_case(cfg, run, spec, N: int) -> Dict[str, Any]:
    """Single-stage continuation from psi = 0, exactly as the reference gen.py does."""
    newton = section(run, "newton")
    phat_cfg = section(run, "phat")
    model = _model(cfg, spec)
    grid = Grid(L=cfg.grid.L, N=int(N))
    solver = TravelingWaveSolver(grid, model)
    psi = np.zeros(grid.N)
    drive = float(run["drive0"])
    residual = float("nan")
    for c in run["approach"]:
        result = solver.newton(float(c), psi, drive, tol=newton["tol"], maxit=newton["maxit"])
        psi, drive, residual = result.psi, result.drive, result.residual

    c = float(run["c"])
    sc = branch_scalars(
        solver, psi, drive, c,
        normalization=phat_cfg["normalization"], phat_seed=phat_cfg["seed"],
        phat_iters=phat_cfg["iters"], phat_shift=phat_cfg["shift"],
    )
    predicted = -sc.drive_prime * sc.phat_one
    return {
        "c": c,
        "L": grid.L,
        "N": grid.N,
        "mu": model.mu,
        "mu2": model.mu2,
        "gamma": model.gamma,
        "couplings": [list(p) for p in model.couplings],
        "f": sc.drive,
        "fprime": sc.drive_prime,
        "kappa": sc.kappa,
        "phat_one": sc.phat_one,
        "pred": float(predicted),
        "relerr": float(abs(sc.kappa - predicted) / abs(predicted)),
        "abserr": float(abs(sc.kappa - predicted)),
        "residual": residual,
        "power_balance_relerr": sc.power_balance_relerr,
        "phat_dot_Vpp": sc.phat_dot_Vpp,
        "normalization": sc.normalization,
    }


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/08_general_lattices.yaml", "data/08_general.json")
    parser.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run
    cases = section(run, "cases")
    wanted = list(cases) if args.cases is None else [c.strip() for c in args.cases.split(",")]
    unknown = [c for c in wanted if c not in cases]
    if unknown:
        parser.error(f"unknown case(s) {unknown}; choose from {list(cases)}")

    started = time.perf_counter()
    rows: Dict[str, Any] = {}
    for name in wanted:
        rows[name] = _solve_case(cfg, run, cases[name], cfg.grid.N)
        print(f"  {name:8s} f={rows[name]['f']:.6f}  relerr={rows[name]['relerr']:.2e}", flush=True)

    attempted: Dict[str, Any] = {}
    for name, spec in dict(run.get("attempted", {}) or {}).items():
        try:
            attempted[name] = _solve_case(cfg, run, spec, cfg.grid.N)
            attempted[name]["status"] = "converged after all"
        except ConvergenceError as exc:
            attempted[name] = {"status": "continuation from psi=0 did not converge", "detail": str(exc)}
            print(f"  {name:8s} did not converge (expected): {exc}", flush=True)

    refinement: Dict[str, Any] = {}
    ref_spec = dict(run.get("refinement", {}) or {})
    if ref_spec:
        case = ref_spec["case"]
        fine = _solve_case(cfg, run, cases[case], int(ref_spec["N"]))
        key = ref_spec["key"]
        refinement[f"{key}_relerr"] = fine["relerr"]
        refinement[f"{key}_fprime"] = fine["fprime"]
        refinement[f"{key}_detail"] = fine
        coarse = rows.get(case)
        if coarse is not None:
            refinement[f"{key}_improvement_factor"] = float(coarse["relerr"] / fine["relerr"])
        print(f"  refinement {case} at N={ref_spec['N']}: relerr={fine['relerr']:.2e}", flush=True)
    elapsed = time.perf_counter() - started

    print_table(
        f"kappa = -f'(c) <phat,1> at c = {run['c']} (phat: Euclidean unit norm, so kappa scales with N)",
        ["case", "f", "f'(c)", "kappa", "-f' <phat,1>", "rel. err"],
        [[k, v["f"], v["fprime"], v["kappa"], v["pred"], v["relerr"]] for k, v in rows.items()],
        ["", ".6f", ".6f", ".6f", ".6f", ".2e"],
    )

    out = output_path(args)
    save_result(
        out,
        {"rows": rows, "attempted": attempted, "refinement": refinement},
        script=SCRIPT, quick=args.quick, config_path=args.config, config=raw,
        sections=["rows", "attempted", "refinement"], elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"], "cases": wanted},
        merge=args.cases is not None,
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
