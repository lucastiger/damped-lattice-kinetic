#!/usr/bin/env python3
"""Experiment 04 -- the fold at c_max, and the dichotomy that says nothing happens there.

Parametrize the branch by arclength ``s``.  Differentiating the traveling-wave equation
along it and pairing with ``phat`` gives the invariant (R5 of the science brief)

    cdot(s) * kappa(s) + mu * sigmadot(s) * <phat,1>_h = 0 ,

which forces a dichotomy.  At an extremum of ``sigma`` (``sigmadot = 0``, ``cdot != 0``)
it says ``kappa = 0``, and ``kappa = 0`` is exactly the condition for ``nu = 0`` to be a
multiple eigenvalue of the pencil -- a stability change (that is ``c_hat1``, experiment
03).  At a turning point in ``c`` (``cdot = 0``, ``sigmadot != 0``) it says instead
``<phat,1> = 0`` with ``kappa != 0``: no multiplier reaches ``+1``, and the stability
does NOT change at the fold.  The kink is already unstable when it arrives at ``c_max``
and stays unstable, with exactly one positive real eigenvalue, on the way back.

This script walks through ``c_max`` by pseudo-arclength continuation in
``X = (psi, drive, c)`` and checks all of it:

(a)-(b) ``rows`` -- one entry per continuation step with ``c``, ``sigma``, ``cdot``,
    ``sigmadot``, ``kappa``, ``<phat,1>`` and the invariant; and at sampled steps the
    positive real eigenvalues of ``Q(nu)``, collected from two shifts.  From the step
    where ``cdot`` changes sign come ``c_max``, and ``kappa`` and ``nu`` interpolated at
    the fold: ``kappa ~ -2.54`` (nonzero, as the dichotomy demands) and ``nu ~ +0.065``
    (positive, i.e. unstable, on both sides and through it).  ``<phat,1>`` changes sign
    within the same step -- the two zeros are 0.336 and 0.342 of the way through it --
    which is the dichotomy's ``<phat,1> = 0`` seen directly.

(c) ``invariant_max`` -- a second run at ``ds = 0.02`` started from ``c = 0.897``,
    reporting the largest violation of the invariant over steps 105-149.  It comes out at
    ~1e-8, and THAT NUMBER IS THE DISCRETIZATION LEVEL OF THE N = 2048 GRID, NOT THE
    NEWTON RESIDUAL.  The run measures it three ways over:

    * The corrector residual ``|F|_inf`` sits at 4.2e-11 at every one of the 150 steps
      while the invariant swings from 1.5e-9 to 1.1e-7 -- two orders of magnitude of
      variation with the Newton residual flat.  So the invariant is not tracking Newton.
    * It is not tracking the inverse iteration either: recomputing ``phat`` with 70, 80
      and 200 iterations gives invariants identical to three significant figures.
    * What it does track, step for step, is ``|M0^T phat| / |phat|`` -- recorded here as
      ``res_phat_rel`` -- at a ratio of about 0.02 along the whole run.  That residual is
      how exactly the *discrete* ``M0`` is singular at that point of the branch: ``M0`` is
      singular only in the continuum, and the identity ``kappa = -mu sigma' <phat,1>``
      holds to first order in how far ``phat`` is from a true null vector.  It rises from
      1.4e-7 to 5.1e-6 over ``c = 0.900196 -> 0.900123`` on the way back down the second
      branch, and the invariant rises with it.

    Compare experiment 02: the same identity at the same ``h`` is good to 1.3e-8, and
    refining to ``N = 4096`` takes it below 1e-12.

    ``window`` is a range of step INDICES, so which stretch of the branch it covers
    depends on where the arclength walk starts -- and, per the point above, the invariant
    is not uniform along the branch.  Two starting procedures are therefore run and both
    reported.  ``reference`` is the one ``handoff/reference/drivers_session.py``'s
    ``fold_invariant()`` performs: its ``arange(0.885, 0.8971, 0.005)`` stops at 0.895 and
    the first arclength corrector converges the ``c = 0.897`` start itself; its window
    covers ``c in [0.899838, 0.900196]`` and gives 7.7e-9.  ``converged_start`` marches
    natural continuation all the way to 0.897 before starting, which puts the same window
    slightly further along, over ``c in [0.899715, 0.900196]``, reaching into the stretch
    where ``res_phat_rel`` peaks; it gives 1.1e-7, with 1.5e-9 to 1.5e-8 either side of
    that one spike.  ``invariant_max`` is the ``primary`` variant of the config, i.e. the
    reference procedure.

The normalization of ``phat`` here is ``||phat||_2 = 1`` with ``<phat, phi'>_h > 0``, NOT
the ``<phat,1>_h = -2 pi`` of scripts 01-03: at the fold ``<phat,1>`` is zero, so it
cannot be used to normalise.  ``kappa`` and ``<phat,1>`` are each scaled by a common
factor relative to the other scripts; their ratio, and the invariant above, are not.

Everything numerical is in ``configs/04_fold_arclength.yaml``.

Usage
-----
    python scripts/04_fold_arclength.py --config configs/04_fold_arclength.yaml
    python scripts/04_fold_arclength.py --quick        # reduced L/N, ~45 s, not claim-grade

Production runtime: about eight minutes -- 62 arclength steps with the spectrum sampled at
19 of them for the fold run, plus 2 x 150 steps for the two invariant runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, load, march, output_path, print_table, section

from dlkin import (
    Grid,
    PseudoArclength,
    init_branch,
    kappa_scalars,
    positive_real_union,
    save_result,
)

SCRIPT = "scripts/04_fold_arclength.py"


def _start_arclength(cfg, run, opts) -> Tuple[PseudoArclength, np.ndarray, np.ndarray]:
    """Land on the branch, march up to ``c_start``, and take the first tangent.

    The initial tangent has no predecessor to continue, so it is seeded with the pure
    ``c`` direction and oriented so that ``cdot > 0``: the run starts on the rising part
    of the branch and walks towards the fold.
    """
    newton = section(run, "newton")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    solver, psi, drive = init_branch(grid, cfg.model, c0=float(opts["c0"]), tol=tol, maxit=maxit)
    psi, drive, _ = march(solver, psi, drive, [float(c) for c in opts["approach"]], tol, maxit)

    arc = PseudoArclength(solver)
    X = arc.pack(psi, drive, float(opts["c_start"]))
    return arc, X, arc.initial_tangent(X, direction=1.0)


def run_fold(cfg, run) -> Dict[str, Any]:
    """(a)-(b) Pseudo-arclength through the fold, with the scalars and the spectrum."""
    opts = section(run, "fold")
    phat = section(run, "phat")
    corrector = section(run, "corrector")
    sample = dict(opts["sample"])
    stop = dict(opts["stop"])
    mu, gamma = cfg.model.mu, cfg.model.gamma

    arc, X, tau = _start_arclength(cfg, run, opts)
    N, h = arc.grid.N, arc.grid.h
    ds = float(opts["ds"])
    c_start = float(opts["c_start"])

    rows: Dict[str, Any] = {}
    ordered: List[Dict[str, Any]] = []
    table = []
    for i in range(int(opts["nsteps"])):
        X, J, M0, M1, p0, residual = arc.step(
            X, tau, ds, tol=float(corrector["tol"]), maxit=int(corrector["maxit"])
        )
        tau = arc.tangent(J, tau)
        ks = kappa_scalars(
            M0,
            M1,
            p0,
            h,
            normalization=str(phat["normalization"]),
            seed=int(phat["seed"]),
            iters=int(phat["iters"]),
            shift=float(phat["shift"]),
        )
        c, drive = float(X[N + 1]), float(X[N])
        cdot, drivedot = float(tau[N + 1]), float(tau[N])

        # sample the spectrum periodically, and always where the branch is near-vertical
        # in c -- which is where the fold is and where the interpolation below needs nu.
        sampled = (i % int(sample["every"]) == 0) or (abs(cdot) < float(sample["cdot_window"]))
        if sampled:
            ev = positive_real_union(
                M0,
                M1,
                [float(s) for s in sample["shifts"]],
                gamma,
                k=int(sample["k"]),
                tol=float(sample["tol"]),
                imtol=float(sample["imtol"]),
                decimals=int(sample["decimals"]),
            )
        else:
            ev = None

        row = {
            "c": c,
            "sigma": drive / mu,
            "cdot": cdot,
            "sigmadot": drivedot / mu,
            "kappa": ks.kappa,
            "phat_one": ks.phat_one,
            "invariant": abs(cdot * ks.kappa + drivedot * ks.phat_one),
            "nu": (ev[0] if ev else None) if sampled else None,
            "n_unstable": (len(ev) if ev is not None else None),
            "sampled": sampled,
            "residual": residual,
        }
        rows[str(i)] = row
        ordered.append({"i": i, **row})
        if sampled:
            table.append([i, c, row["sigma"], cdot, ks.kappa, ks.phat_one, row["nu"], row["n_unstable"]])
            print(
                f"    {i:4d} c={c:.6f} sigma={row['sigma']:.6f} cdot={cdot:+.2e} "
                f"kappa={ks.kappa:+.5f} <p,1>={ks.phat_one:+.6f} nu={ev}",
                flush=True,
            )
        if cdot < 0 and i > int(stop["after_step"]) and c < float(stop.get("below_c", c_start)):
            break

    print_table(
        f"(a) pseudo-arclength through the fold  (L = {arc.grid.L:g}, N = {N}, ds = {ds}, "
        f"||phat||_2 = 1; sampled steps only, {len(ordered)} taken)",
        ["step", "c", "sigma", "cdot", "kappa", "<phat,1>", "nu", "n_unst"],
        table,
        ["d", ".6f", ".6f", "+.3e", "+.5f", "+.6f", "+.6f", "d"],
    )
    return {"rows": rows, "ordered": ordered, "L": arc.grid.L, "N": N, "ds": ds}


def _sign_change(ordered: Sequence[Dict[str, Any]], key: str):
    """First step index where ``key`` changes sign, and the fraction of the step at zero.

    Returns ``(i, t)`` with ``t = f_i / (f_i - f_{i+1}) in [0,1]`` the linear
    interpolation parameter, or ``(None, None)`` if the quantity never changes sign.
    """
    for a, b in zip(ordered, ordered[1:]):
        fa, fb = a[key], b[key]
        if fa == 0.0:
            return a["i"], 0.0
        if fa * fb < 0.0:
            return a["i"], fa / (fa - fb)
    return None, None


def _interp(ordered: Sequence[Dict[str, Any]], index: int, t: float, key: str):
    """Linear interpolation of ``key`` a fraction ``t`` through step ``index``."""
    a = next(r for r in ordered if r["i"] == index)
    b = next(r for r in ordered if r["i"] == index + 1)
    if a[key] is None or b[key] is None:
        return None
    return a[key] + t * (b[key] - a[key])


def analyse_fold(fold: Dict[str, Any]) -> Dict[str, Any]:
    """(b) The fold quantities read off the two steps that bracket ``cdot = 0``."""
    ordered = fold["ordered"]
    i_c, t_c = _sign_change(ordered, "cdot")
    i_p, t_p = _sign_change(ordered, "phat_one")
    sampled = [r for r in ordered if r["sampled"]]

    record: Dict[str, Any] = {
        "fold_step": i_c,
        "zero_fraction_cdot": t_c,
        "zero_fraction_phat_one": t_p,
        "same_step": bool(i_c is not None and i_c == i_p),
        "c_max": None if i_c is None else _interp(ordered, i_c, t_c, "c"),
        "kappa_at_fold": None if i_c is None else _interp(ordered, i_c, t_c, "kappa"),
        "sigma_at_fold": None if i_c is None else _interp(ordered, i_c, t_c, "sigma"),
        "nu_at_fold": None if i_c is None else _interp(ordered, i_c, t_c, "nu"),
        "n_unstable_exactly_one_all_steps": bool(
            bool(sampled) and all(r["n_unstable"] == 1 for r in sampled)
        ),
        "n_steps": len(ordered),
        "n_sampled": len(sampled),
        "invariant_max_fold_run": max(r["invariant"] for r in ordered),
    }
    print_table(
        "(b) the fold",
        ["quantity", "value"],
        [
            ["step bracketing cdot = 0", record["fold_step"]],
            ["zero fraction (cdot)", record["zero_fraction_cdot"]],
            ["zero fraction (<phat,1>)", record["zero_fraction_phat_one"]],
            ["same step", record["same_step"]],
            ["c_max", record["c_max"]],
            ["sigma(c_max)", record["sigma_at_fold"]],
            ["kappa at the fold", record["kappa_at_fold"]],
            ["nu at the fold", record["nu_at_fold"]],
            ["exactly one unstable, all sampled steps", record["n_unstable_exactly_one_all_steps"]],
            ["max invariant over this run", record["invariant_max_fold_run"]],
        ],
        ["", ".9g"],
    )
    return record


def _invariant_run(cfg, run, opts, label: str) -> Dict[str, Any]:
    """One arclength run that records only the invariant (and what it tracks)."""
    phat = section(run, "phat")
    corrector = section(run, "corrector")

    arc, X, tau = _start_arclength(cfg, run, opts)
    N, h = arc.grid.N, arc.grid.h
    ds = float(opts["ds"])
    lo, hi = (int(v) for v in opts["window"])
    c_stop = float(opts["c_stop"])

    steps: List[Dict[str, Any]] = []
    for i in range(int(opts["nsteps"])):
        X, J, M0, M1, p0, residual = arc.step(
            X, tau, ds, tol=float(corrector["tol"]), maxit=int(corrector["maxit"])
        )
        tau = arc.tangent(J, tau)
        ks = kappa_scalars(
            M0,
            M1,
            p0,
            h,
            normalization=str(phat["normalization"]),
            seed=int(phat["seed"]),
            iters=int(phat["iters"]),
            shift=float(phat["shift"]),
        )
        steps.append(
            {
                "i": i,
                "c": float(X[N + 1]),
                "invariant": abs(float(tau[N + 1]) * ks.kappa + float(tau[N]) * ks.phat_one),
                "res_phat_rel": ks.res_phat_rel,
                "corrector_residual": residual,
            }
        )
        if float(X[N + 1]) < c_stop:
            print(f"    [{label}] left the region of interest at step {i} (c < {c_stop})")
            break

    windowed = [r for r in steps if lo <= r["i"] < hi]
    worst = max(windowed, key=lambda r: r["invariant"]) if windowed else None
    record = {
        "L": arc.grid.L,
        "N": N,
        "ds": ds,
        "approach": [float(c) for c in opts["approach"]],
        "c_start": float(opts["c_start"]),
        "window": [lo, hi],
        "n_steps": len(steps),
        "n_in_window": len(windowed),
        "invariant_max": None if worst is None else worst["invariant"],
        "argmax_step": None if worst is None else worst["i"],
        "argmax_c": None if worst is None else worst["c"],
        "window_c_min": min((r["c"] for r in windowed), default=None),
        "window_c_max": max((r["c"] for r in windowed), default=None),
        "invariant_max_all_steps": max((r["invariant"] for r in steps), default=None),
        "corrector_residual_max": max((r["corrector_residual"] for r in steps), default=None),
        "res_phat_rel_max": max((r["res_phat_rel"] for r in steps), default=None),
        "steps": steps,
    }
    print(
        f"    [{label}] {len(steps)} steps, window [{lo},{hi}) holds {len(windowed)}: "
        f"max invariant {record['invariant_max']:.3e} at step {record['argmax_step']} "
        f"(c={record['argmax_c']:.6f}); |F|_inf <= {record['corrector_residual_max']:.1e}",
        flush=True,
    )
    return record


def run_invariant(cfg, run) -> Tuple[Dict[str, Any], str]:
    """(c) The arclength invariant, from each of the configured starting procedures.

    The invariant is a property of the branch point, but ``window`` selects step indices,
    so the two procedures sample different stretches of the branch with the same indices.
    Both are run; ``primary`` says which one ``invariant_max`` comes from.
    """
    opts = section(run, "invariant")
    variants = dict(opts["variants"])
    primary = str(opts["primary"])
    if primary not in variants:
        raise ValueError(
            f"run.invariant.primary = {primary!r} is not one of the variants {list(variants)}."
        )

    runs: Dict[str, Any] = {}
    for name, override in variants.items():
        runs[name] = _invariant_run(cfg, run, {**opts, **dict(override)}, name)

    print_table(
        "(c) arclength invariant |cdot kappa + mu sigmadot <phat,1>|  "
        f"(L = {runs[primary]['L']:g}, N = {runs[primary]['N']}, ds = {runs[primary]['ds']})",
        ["variant", "steps", "window c range", "max in window", "at step", "|F|_inf max", "res_phat max"],
        [
            [
                name + ("  (primary)" if name == primary else ""),
                r["n_steps"],
                f"[{r['window_c_min']:.6f}, {r['window_c_max']:.6f}]"
                if r["window_c_min"] is not None
                else "--",
                r["invariant_max"],
                r["argmax_step"],
                r["corrector_residual_max"],
                r["res_phat_rel_max"],
            ]
            for name, r in runs.items()
        ],
        ["", "d", "", ".4e", "d", ".1e", ".1e"],
    )
    return runs, primary


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/04_fold_arclength.yaml", "data/04_fold.json")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    fold = run_fold(cfg, run)
    payload: Dict[str, Any] = {"rows": fold["rows"]}
    payload.update(analyse_fold(fold))
    payload["fold_run"] = {"L": fold["L"], "N": fold["N"], "ds": fold["ds"]}
    invariant_runs, primary = run_invariant(cfg, run)
    payload["invariant_max"] = invariant_runs[primary]["invariant_max"]
    payload["invariant_primary"] = primary
    payload["invariant_runs"] = invariant_runs
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=["fold", "invariant"],
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
