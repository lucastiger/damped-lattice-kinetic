#!/usr/bin/env python3
"""Experiment 01 -- the kinetic relation sigma(c) and its first critical point.

Three pieces, all for the Frenkel-Kontorova lattice at mu = 1, gamma = 0.1, with the
``<phat,1>_h = -2 pi`` normalization of ``phat``:

(a) ``bc_check`` -- a check that the periodic collocation really does reproduce the
    advance-delay problem's far-field conditions.  Only the FIRST stage of
    ``init_branch`` is run: the physical-width template ``w0 = sqrt((1-c0^2)/mu)`` solved
    from a flat start at ``c0 = 0.88``.  The converged profile is compared with
    ``arcsin(sigma) + 2 pi`` at the left edge and ``arcsin(sigma)`` at the right, where
    "edge" means the outermost collocation points ``xi_0 = -L`` and ``xi_{N-1} = L - h``.
    The residual ~5e-9 is the kink's own exponential tail at ``|xi| = L``, not a
    discretization error -- it shrinks with ``L``, not with ``N``.

(b) ``curve`` -- the kinetic relation over ``c in [0.20, 0.8995]`` by natural
    continuation, recording at every velocity the exact ``sigma'(c)``, ``kappa``,
    ``<phat,1>`` and the reduced coefficient ``m``.  Written to CSV (one row per
    velocity); the JSON records the path and a summary, not the 700-odd rows.

(c) ``critical`` -- ``c_hat1``, the zero of the exact ``sigma'(c)``, i.e. the maximum of
    the kinetic relation on the first branch.  Bracketed between two velocities where
    ``sigma'`` is computed by the bordered solve, then refined by secant.  This is the
    0.8989 of the 2020 paper, resolved to 0.8989297.

Everything numerical is in ``configs/01_kinetic_relation.yaml``.

Usage
-----
    python scripts/01_kinetic_relation.py --config configs/01_kinetic_relation.yaml
    python scripts/01_kinetic_relation.py --quick        # reduced L/N, ~10 s, not claim-grade

Production runtime: about 19 minutes on four cores.  The curve dominates: 714 continuation
points at N = 2048, each a Newton solve plus three dense solves (the bordered branch
derivative, the inverse iteration for phat, and the bordered system for m).
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, crange, load, march, output_path, print_table, section

from dlkin import (
    Grid,
    TravelingWaveSolver,
    branch_scalars,
    init_branch,
    save_result,
)

SCRIPT = "scripts/01_kinetic_relation.py"
CURVE_COLUMNS = (
    "c",
    "sigma",
    "sigma_prime",
    "kappa",
    "phat_one",
    "m",
    "identity_relerr",
    "residual",
)


def run_bc_check(cfg, run) -> Dict[str, Any]:
    """(a) Stage one of ``init_branch`` alone, and the far-field errors it leaves."""
    opts = section(run, "bc_check")
    newton = section(run, "newton")
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    c0 = float(opts["c0"])

    # the physical-width template: the sine-Gordon kink width at speed c0
    width = float(np.sqrt((1.0 - c0**2) / cfg.model.mu))
    solver = TravelingWaveSolver(grid, replace(cfg.model, template_width=width))
    result = solver.newton(
        c0,
        np.zeros(grid.N),
        float(opts["sigma_guess"]) * cfg.model.mu,
        tol=float(newton["tol"]),
        maxit=int(newton["maxit"]),
    )
    phi = solver.phi(result.psi)
    sigma = result.sigma
    record = {
        "L": grid.L,
        "N": grid.N,
        "c0": c0,
        "template_width": width,
        "sigma": sigma,
        "residual": result.residual,
        "iters": result.iters,
        "phi_left": float(phi[0]),
        "phi_right": float(phi[-1]),
        "bc_left": float(np.arcsin(sigma) + 2.0 * np.pi),
        "bc_right": float(np.arcsin(sigma)),
        "err_left": float(phi[0] - (np.arcsin(sigma) + 2.0 * np.pi)),
        "err_right": float(phi[-1] - np.arcsin(sigma)),
    }
    print_table(
        f"(a) boundary-condition check  (stage-1 template w0 = {width:.6f}, "
        f"L = {grid.L:g}, N = {grid.N})",
        ["quantity", "value"],
        [
            ["sigma(c0=%.2f)" % c0, record["sigma"]],
            ["Newton residual", record["residual"]],
            ["phi(-L)", record["phi_left"]],
            ["arcsin(sigma) + 2 pi", record["bc_left"]],
            ["err_left", record["err_left"]],
            ["phi(+L)", record["phi_right"]],
            ["arcsin(sigma)", record["bc_right"]],
            ["err_right", record["err_right"]],
        ],
        ["", ".12e"],
    )
    return record


def run_curve(cfg, run) -> Dict[str, Any]:
    """(b) Natural continuation both ways from ``c0``, one CSV row per velocity."""
    opts = section(run, "curve")
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    c0, step = float(opts["c0"]), float(opts["step"])
    decimals = int(opts["decimals"])

    # the coarse step is refined above refine_from, where sigma(c) turns over
    refine_from, refine_step = float(opts["refine_from"]), float(opts["refine_step"])
    up = crange(c0 + step, refine_from, step, decimals)
    up += crange(refine_from + refine_step, float(opts["c_max"]), refine_step, decimals)
    down = crange(c0 - step, float(opts["c_min"]), -step, decimals)

    solver, psi0, drive0 = init_branch(grid, cfg.model, c0=c0, tol=tol, maxit=maxit)
    rows: List[Dict[str, Any]] = []

    def record(c: float, psi: np.ndarray, drive: float, residual: float) -> None:
        s = branch_scalars(
            solver,
            psi,
            drive,
            c,
            normalization=str(phat["normalization"]),
            phat_seed=int(phat["seed"]),
            phat_iters=int(phat["iters"]),
            phat_shift=float(phat["shift"]),
        )
        rows.append(
            {
                "c": c,
                "sigma": s.sigma,
                "sigma_prime": s.sigma_prime,
                "kappa": s.kappa,
                "phat_one": s.phat_one,
                "m": s.m,
                "identity_relerr": s.identity_relerr,
                "residual": residual,
            }
        )

    record(c0, psi0, drive0, 0.0)
    for leg, values in (("up", up), ("down", down)):
        psi, drive = psi0, drive0
        for c in values:
            result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
            psi, drive = result.psi, result.drive
            record(c, psi, drive, result.residual)
        print(f"    curve: {leg} leg done, {len(values)} points, c -> {values[-1] if values else c0}")

    rows.sort(key=lambda r: r["c"])
    csv_path = Path(opts["csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CURVE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CURVE_COLUMNS})

    sigma = np.array([r["sigma"] for r in rows])
    imax = int(np.argmax(sigma))
    summary = {
        "csv": str(csv_path),
        "columns": list(CURVE_COLUMNS),
        "L": grid.L,
        "N": grid.N,
        "c0": c0,
        "c_min": rows[0]["c"],
        "c_max": rows[-1]["c"],
        "step": step,
        "refine_from": refine_from,
        "refine_step": refine_step,
        "n_points": len(rows),
        "max_residual": max(r["residual"] for r in rows),
        "max_identity_relerr": max(r["identity_relerr"] for r in rows),
        "sigma_max_on_grid": float(sigma[imax]),
        "c_at_sigma_max_on_grid": rows[imax]["c"],
    }
    stride = max(1, len(rows) // 12)
    print_table(
        f"(b) kinetic curve  (L = {grid.L:g}, N = {grid.N}, {len(rows)} points "
        f"-> {csv_path}; every {stride}th row shown)",
        ["c", "sigma", "sigma'", "kappa", "<phat,1>", "m", "id relerr"],
        [
            [r["c"], r["sigma"], r["sigma_prime"], r["kappa"], r["phat_one"], r["m"], r["identity_relerr"]]
            for r in rows[::stride]
        ],
        [".7f", ".9f", "+.7f", "+.7f", "+.7f", "9.2f", ".2e"],
    )
    return summary


def run_critical(cfg, run) -> Dict[str, Any]:
    """(c) The zero of the exact sigma'(c): bracket, then secant to |sigma'| < tol."""
    opts = section(run, "critical")
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    approach = dict(opts["approach"])
    secant = dict(opts["secant"])
    probes = [float(c) for c in opts["probe"]]
    bracket = [float(c) for c in opts["bracket"]]

    solver, psi, drive = init_branch(grid, cfg.model, c0=float(opts["c0"]), tol=tol, maxit=maxit)
    psi, drive, _ = march(
        solver,
        psi,
        drive,
        crange(float(approach["start"]), float(approach["stop"]), float(approach["step"])),
        tol,
        maxit,
    )

    def sigma_prime_at(c: float):
        """Solve at ``c`` (warm start from the running iterate) and return the exact sigma'."""
        nonlocal psi, drive
        result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
        psi, drive = result.psi, result.drive
        s = branch_scalars(
            solver,
            psi,
            drive,
            c,
            normalization=str(phat["normalization"]),
            phat_seed=int(phat["seed"]),
            phat_iters=int(phat["iters"]),
            phat_shift=float(phat["shift"]),
        )
        return s.sigma, s.sigma_prime, result.residual

    trace: List[Dict[str, Any]] = []
    probed: Dict[float, float] = {}
    for c in probes:
        sigma, sigp, residual = sigma_prime_at(c)
        probed[c] = sigp
        trace.append({"stage": "probe", "c": c, "sigma": sigma, "sigma_prime": sigp, "residual": residual})

    missing = [c for c in bracket if c not in probed]
    if missing:
        raise ValueError(
            f"the bracket ends {missing} are not among the probed velocities {probes}; "
            "list them in run.critical.probe so sigma' is known there before the secant."
        )
    a, b = bracket
    fa, fb = probed[a], probed[b]
    if fa * fb > 0.0:
        raise ValueError(
            f"sigma' does not change sign on the bracket [{a}, {b}]: "
            f"sigma'({a}) = {fa:.6e}, sigma'({b}) = {fb:.6e}."
        )

    cm, sigma_m, fm = b, None, fb
    iters = 0
    for iters in range(1, int(secant["maxit"]) + 1):
        cm = b - fb * (b - a) / (fb - fa)
        sigma_m, fm, residual = sigma_prime_at(cm)
        trace.append({"stage": "secant", "c": cm, "sigma": sigma_m, "sigma_prime": fm, "residual": residual})
        a, fa, b, fb = b, fb, cm, fm
        if abs(fm) < float(secant["tol"]):
            break

    record = {
        "L": grid.L,
        "N": grid.N,
        "bracket": bracket,
        "bracket_sigma_prime": [probed[a0] for a0 in bracket],
        "secant_iters": iters,
        "secant_tol": float(secant["tol"]),
        "c_hat1": float(cm),
        "sigma_hat1": float(sigma_m),
        "sigma_prime_at_root": float(fm),
        "trace": trace,
    }
    print_table(
        f"(c) c_hat1 = zero of the exact sigma'(c)  (L = {grid.L:g}, N = {grid.N})",
        ["stage", "c", "sigma", "sigma'", "Newton res"],
        [[t["stage"], t["c"], t["sigma"], t["sigma_prime"], t["residual"]] for t in trace],
        ["", ".9f", ".10f", "+.3e", ".2e"],
    )
    print(
        f"    c_hat1 = {record['c_hat1']:.7f}   sigma_hat1 = {record['sigma_hat1']:.7f}   "
        f"sigma'(c_hat1) = {record['sigma_prime_at_root']:+.2e}  "
        f"({record['secant_iters']} secant iterations)"
    )
    return record


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/01_kinetic_relation.yaml", "data/01_kinetic.json")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    payload["bc_check"] = run_bc_check(cfg, run)
    payload["curve"] = run_curve(cfg, run)
    payload["critical"] = run_critical(cfg, run)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=["bc_check", "curve", "critical"],
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
