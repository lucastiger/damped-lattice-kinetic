#!/usr/bin/env python3
"""Experiment 08 -- the identity outside Frenkel-Kontorova: five lattices, one identity.

Proposition 7.1 says the Jordan-chain identity is not a property of the sine potential or
of nearest-neighbour coupling.  For

    u_n'' + gamma u_n' = sum_j C_j (u_{n+j} - 2 u_n + u_{n-j}) + f - V'(u_n),
    V'(u) = mu sin u + mu2 sin 2u,

any symmetric finite-range coupling and any smooth ``2 pi``-periodic ``V`` gives

    kappa = <phat, M1 phi'> = -f'(c) <phat,1> ,

because the only structural facts the proof uses are that the drive is additive and
undifferentiated, that the coupling is symmetric (so ``M0^T`` is ``M0`` with
``gamma -> -gamma``), and that the damping is a uniform scalar.  This experiment changes
the potential, the coupling range, the substrate strength and the damping, and checks the
identity in each.

The five rows are deliberately run the naive way: from a flat ``psi = 0`` with ``f = 0.5``,
walking ``c = 0.55, 0.65, 0.75, 0.82``.  No ``init_branch``, no physical-width template, no
profile transfer -- if the identity needed the careful initialization it would not be an
identity.  Note the Newton tolerance here is ``1e-11``, which is what the quoted rows were
computed against, not the ``1e-12`` of experiments 01-04.

``phat`` carries the ``||phat||_2 = 1`` normalization, NOT ``<phat,1> = -2 pi``: here
``<phat,1>`` is one of the two sides being compared, so fixing it by fiat would make the
test circular.

Two results are not simply "it holds":

``refinement``
    the ``mu2 = -0.5`` row comes out at ``relerr ~ 4e-6``, three to four orders worse than
    the others.  That is resolution, not a failure of the identity: the second harmonic
    sharpens the kink, and rerunning that one row at ``N = 4096`` drops the error to
    ``~2e-12`` while ``f'`` moves in the seventh digit.  The row is kept at ``N = 2048``
    with the refinement recorded beside it, rather than quietly refined.

``expected_failure``
    ``mu2 = +0.35`` does not converge from the flat start.  It is recorded as not
    converged, with the residual it reached.  This is a statement about the naive
    continuation, not about the lattice or the identity -- a better initialization would
    very likely find the branch -- and it is left as it is because the manuscript reports
    it that way.

Everything numerical is in ``configs/08_general_lattices.yaml``.

Usage
-----
    python scripts/08_general_lattices.py --config configs/08_general_lattices.yaml
    python scripts/08_general_lattices.py --quick      # reduced L/N, ~15 s, not claim-grade

Production runtime: about a minute.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, load, output_path, print_table, section

from dlkin import (
    ConvergenceError,
    Grid,
    LatticeModel,
    TravelingWaveSolver,
    branch_scalars,
    save_result,
)

SCRIPT = "scripts/08_general_lattices.py"


def _model(spec: Dict[str, Any]) -> LatticeModel:
    return LatticeModel(
        gamma=float(spec["gamma"]),
        mu=float(spec["mu"]),
        mu2=float(spec["mu2"]),
        couplings=tuple((int(j), float(a)) for j, a in spec["couplings"]),
    )


def _walk(cfg, run, spec: Dict[str, Any], N: int) -> Tuple[Any, np.ndarray, float, float, str]:
    """Naive continuation from a flat start; returns the state, residual and a status."""
    newton = section(run, "newton")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(run["L"]), N=int(N))
    solver = TravelingWaveSolver(grid, _model(spec))
    psi = np.full(grid.N, float(run["psi0"]))
    drive = float(run["drive0"])

    residual = float("nan")
    for c in [float(x) for x in run["c_values"]]:
        try:
            result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
        except ConvergenceError as exc:
            return solver, exc.psi, exc.drive, float(exc.residual), (
                f"continuation from psi={run['psi0']:g} did not converge at c={c:g}"
            )
        psi, drive, residual = result.psi, result.drive, result.residual
        if not result.converged:
            return solver, psi, drive, residual, (
                f"continuation from psi={run['psi0']:g} did not converge at c={c:g}"
            )
    return solver, psi, drive, residual, "converged"


def _scalars(cfg, run, solver, psi, drive):
    phat = section(run, "phat")
    return branch_scalars(
        solver,
        psi,
        drive,
        float(run["c"]),
        normalization=str(phat["normalization"]),
        phat_seed=int(phat["seed"]),
        phat_iters=int(phat["iters"]),
        phat_shift=float(phat["shift"]),
    )


def run_rows(cfg, run) -> Dict[str, Any]:
    """The five lattices."""
    rows: Dict[str, Any] = {}
    table = []
    for name, spec in dict(run["rows"]).items():
        solver, psi, drive, residual, status = _walk(cfg, run, dict(spec), int(run["N"]))
        if status != "converged":
            rows[name] = {"status": status, "residual": residual, **dict(spec)}
            print(f"    {name}: {status} (residual {residual:.2e})", flush=True)
            continue
        s = _scalars(cfg, run, solver, psi, drive)
        rows[name] = {
            "mu": s.mu,
            "mu2": solver.model.mu2,
            "gamma": solver.model.gamma,
            "couplings": [list(p) for p in solver.model.couplings],
            "c": float(run["c"]),
            "L": solver.grid.L,
            "N": solver.grid.N,
            "f": s.drive,
            "fprime": s.drive_prime,
            "kappa": s.kappa,
            "phat_one": s.phat_one,
            "pred": -s.drive_prime * s.phat_one,
            "relerr": s.identity_relerr,
            "residual": residual,
            "status": status,
        }
        table.append(
            [name, s.drive, s.drive_prime, s.kappa, rows[name]["pred"], s.identity_relerr, residual]
        )
        print(
            f"    {name}: f={s.drive:.6f} f'={s.drive_prime:.6f} kappa={s.kappa:.6f} "
            f"relerr={s.identity_relerr:.2e}",
            flush=True,
        )
    print_table(
        f"kappa = -f'(c) <phat,1>  (c = {run['c']}, L = {run['L']:g}, N = {run['N']}, "
        "||phat||_2 = 1)",
        ["lattice", "f", "f'", "kappa", "-f' <phat,1>", "relerr", "Newton res"],
        table,
        ["", ".6f", ".6f", ".6f", ".6f", ".2e", ".1e"],
    )
    return rows


def run_refinement(cfg, run) -> Dict[str, Any]:
    """One row again at twice the resolution: is its larger error a resolution effect?"""
    opts = section(run, "refinement")
    name = str(opts["row"])
    spec = dict(dict(run["rows"])[name])
    N = int(opts["N"])
    solver, psi, drive, residual, status = _walk(cfg, run, spec, N)
    if status != "converged":
        return {"row": name, "N": N, "status": status, "residual": residual}
    s = _scalars(cfg, run, solver, psi, drive)
    record = {
        "row": name,
        "N": N,
        f"{name}_N{N}_relerr": s.identity_relerr,
        f"{name}_N{N}_fprime": s.drive_prime,
        f"{name}_N{N}_f": s.drive,
        f"{name}_N{N}_kappa": s.kappa,
        "residual": residual,
        "status": status,
    }
    print_table(
        f"refinement: {name} at N = {N}",
        ["quantity", "value"],
        [
            ["f", s.drive],
            ["f'", s.drive_prime],
            ["kappa", s.kappa],
            ["relerr", s.identity_relerr],
            ["Newton residual", residual],
        ],
        ["", ".10g"],
    )
    return record


def run_expected_failure(cfg, run) -> Dict[str, Any]:
    """The naive start that does not find the branch.  Recorded, not rescued."""
    out: Dict[str, Any] = {}
    for name, spec in dict(run["expected_failure"]).items():
        started = time.perf_counter()
        _, _, _, residual, status = _walk(cfg, run, dict(spec), int(run["N"]))
        out[name] = {
            **{k: (v if not isinstance(v, list) else [list(p) for p in v]) for k, v in dict(spec).items()},
            "status": status,
            "residual": residual,
            "converged": status == "converged",
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
        print(
            f"    {name}: {status}; final residual {residual:.3e} "
            f"({out[name]['elapsed_sec']:.1f} s)"
        )
    return out


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(
        __doc__ or "", "configs/08_general_lattices.yaml", "data/08_general.json"
    )
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    payload["rows"] = run_rows(cfg, run)
    payload["refinement"] = run_refinement(cfg, run)
    payload["rows"].update(run_expected_failure(cfg, run))
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=["rows", "refinement", "expected_failure"],
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
