#!/usr/bin/env python3
"""Experiment 07 -- conformal symplecticity, on paper's terms and on the lattice's.

R1 says that the monodromy of the linearized damped lattice satisfies
``M^T J M = e^{-gamma T} J``.  The consequence the note leans on is the multiplier pairing
``rho rho_hat = e^{-gamma T}``: dissipation does not destroy the symplectic pairing of
Floquet multipliers, it rescales it by a factor that is the same for every pair.  That is
what makes the neutral eigenvalue's partner locatable, and hence what makes ``kappa``
mean something.

(a) ``random_K`` -- the algebra alone.  The relation needs nothing of ``K(t)`` beyond
    symmetry: not a lattice, not a traveling wave, not even continuity in ``t``.  So the
    test redraws ``K`` from a seeded Gaussian at every step and checks the relation anyway.
    The ``sweep`` repeats it over ``n``, ``dt`` and two integrators, so that the result
    cannot be read as an artefact of one setting.  Note what each integrator measures:
    ``expm`` satisfies the relation exactly per step in exact arithmetic, so its residual
    is accumulated round-off; ``rk4`` imposes nothing, so its residual is the integration
    error and must fall with ``dt``.

(b) ``direct_monodromy`` -- THE INDEPENDENT CROSS-CHECK, AND IT IS NOT CLAIMED IN THE
    MANUSCRIPT.  Everything in :mod:`dlkin.spectral` works with the co-traveling pencil,
    which presupposes that Floquet solutions take the form ``e^{nu t} p(n - c t)``.  This
    section does not presuppose it: it builds the linearization of the actual
    Frenkel-Kontorova lattice about a computed kink on a ring of ``n_sites``, integrates it
    over one period ``T = 1/c`` with RK4, composes with the one-site shift, and looks at
    what comes out.  Three things are checked, and a ``status`` is recorded either way:

    * ``cs_relerr``, the conformal-symplectic residual of that monodromy.  RK4 imposes no
      structure, so this is a genuine measurement; it should fall like ``dt^4``.
    * ``rho_translation_err``, ``rho_partner_err`` -- the translation mode must give
      ``rho = 1`` exactly and, by the pairing, its partner ``rho = e^{-gamma/c}``.
    * ``pencil_match`` -- the pencil's own ``exp(nu/c)`` against the monodromy's
      multipliers.  Only the *localized* modes can agree: the ring has ``n_sites``
      sites while the spectral domain is ``2L`` long, so the two discretize the essential
      spectrum on different wavenumber grids and their extended modes are simply different
      objects.  The comparison is reported for the localized ones and the rest is labelled.

    The floor on this cross-check is the ring's seam.  ``V''(phi)`` is ``2 pi``-periodic so
    it is continuous around the ring even though ``phi`` jumps by ``2 pi`` across it, but
    only up to the kink's own tail at ``|xi| = n_sites/2``; ``potential_wrap_mismatch``
    reports that, and it is what limits ``rho_translation_err``, not ``N`` and not ``dt``.

Everything numerical is in ``configs/07_conformal_symplectic.yaml``.

Usage
-----
    python scripts/07_conformal_symplectic.py --config configs/07_conformal_symplectic.yaml
    python scripts/07_conformal_symplectic.py --quick     # reduced settings, ~20 s

Production runtime: about a minute.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, crange, load, march, output_path, path_to, print_table, section

from dlkin import (
    branch_scalars,
    conformal_symplectic_flow_test,
    init_branch,
    lattice_monodromy,
    pencil_eigs,
    real_nontrivial,
    save_result,
)

SCRIPT = "scripts/07_conformal_symplectic.py"


def run_random_K(run) -> Dict[str, Any]:
    """(a) The claimed setting: n = 8, gamma = 0.37, T = 0.9, dt = 1e-4, resampled K."""
    opts = section(run, "random_K")
    record = conformal_symplectic_flow_test(
        n=int(opts["n"]),
        gamma=float(opts["gamma"]),
        T=float(opts["T"]),
        dt=float(opts["dt"]),
        seed=int(opts["seed"]),
        integrator=str(opts["integrator"]),
    )
    print_table(
        "(a) M^T J M = e^{-gamma T} J on a resampled symmetric K(t)",
        ["quantity", "value"],
        [
            ["n", record["n"]],
            ["gamma", record["gamma"]],
            ["T", record["T"]],
            ["dt", record["dt"]],
            ["steps", record["n_steps"]],
            ["integrator", record["integrator"]],
            ["relerr", record["relerr"]],
        ],
        ["", ".6g"],
    )
    return record


def run_sweep(run) -> Dict[str, Any]:
    """(a') The same identity over n, dt and both integrators."""
    opts = section(run, "sweep")
    rows: Dict[str, Any] = {}
    table = []
    for integrator in [str(s) for s in opts["integrators"]]:
        for n in [int(x) for x in opts["n"]]:
            for dt in [float(x) for x in opts["dt"]]:
                rec = conformal_symplectic_flow_test(
                    n=n,
                    gamma=float(opts["gamma"]),
                    T=float(opts["T"]),
                    dt=dt,
                    seed=int(opts["seed"]),
                    integrator=integrator,
                )
                key = f"{integrator}_n{n}_dt{dt:g}"
                rows[key] = rec
                table.append([integrator, n, dt, rec["n_steps"], rec["relerr"]])
                print(f"    {key}: relerr={rec['relerr']:.3e}", flush=True)
    print_table(
        "(a') the same identity across settings",
        ["integrator", "n", "dt", "steps", "relerr"],
        table,
        ["", "d", ".0e", "d", ".3e"],
    )
    return {"rows": rows, "max_relerr": max(r["relerr"] for r in rows.values())}


def run_direct(cfg, run) -> Dict[str, Any]:
    """(b) The monodromy of the actual FK lattice, integrated in time."""
    opts = section(run, "direct_monodromy")
    newton = section(run, "newton")
    approach = dict(run["approach"])
    pencil_cfg = dict(opts["pencil"])
    thresholds = dict(opts["thresholds"])
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    gamma = cfg.model.gamma

    from dlkin import Grid  # noqa: PLC0415 -- the section has its own resolution

    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    solver, psi, drive = init_branch(grid, cfg.model, c0=float(run["c0"]), tol=tol, maxit=maxit)
    psi, drive, _ = march(
        solver,
        psi,
        drive,
        crange(float(approach["start"]), float(approach["stop"]), float(approach["step"])),
        tol,
        maxit,
    )
    current = float(approach["stop"])

    cases: Dict[str, Any] = {}
    table = []
    for name, target in dict(opts["cases"]).items():
        target = float(target)
        psi, drive, _ = march(
            solver,
            psi,
            drive,
            path_to(current, target, float(run["walk_step"]), int(run["decimals"])),
            tol,
            maxit,
        )
        current = target

        lm = lattice_monodromy(
            solver, psi, drive, target, n_sites=int(opts["n_sites"]), n_steps=int(opts["n_steps"])
        )
        rho = lm.multipliers
        order = np.argsort(-np.abs(rho))
        leading = rho[order][: int(opts["n_leading"])]
        partner = float(np.exp(-gamma / target))
        rho_translation_err = float(np.min(np.abs(rho - 1.0)))
        rho_partner_err = float(np.min(np.abs(rho - partner)))

        # the pencil's own prediction, for the modes near nu = 0
        scalars = branch_scalars(solver, psi, drive, target)
        nus = pencil_eigs(
            scalars.M0,
            scalars.M1,
            float(pencil_cfg["shift"]),
            k=int(pencil_cfg["k"]),
            tol=float(pencil_cfg["tol"]),
        )
        localized = [0.0, -gamma] + real_nontrivial(nus, gamma)
        match = []
        for nu in localized:
            predicted = float(np.exp(nu / target))
            err = float(np.min(np.abs(rho - predicted)))
            match.append({"nu": nu, "rho_pencil": predicted, "abs_err": err})

        ok = (
            lm.cs_relerr <= float(thresholds["cs_relerr"])
            and rho_translation_err <= float(thresholds["rho_match"])
            and rho_partner_err <= float(thresholds["rho_match"])
        )
        worst_match = max(m["abs_err"] for m in match)
        status = (
            "ok: conformal relation, both structural multipliers and every localized "
            "pencil mode reproduced by direct time integration"
            if ok and worst_match <= float(thresholds["rho_match"])
            else (
                "inconclusive: "
                + ", ".join(
                    part
                    for part, bad in (
                        (f"cs_relerr={lm.cs_relerr:.2e}", lm.cs_relerr > float(thresholds["cs_relerr"])),
                        (f"|rho-1|={rho_translation_err:.2e}", rho_translation_err > float(thresholds["rho_match"])),
                        (f"|rho-e^-gamma/c|={rho_partner_err:.2e}", rho_partner_err > float(thresholds["rho_match"])),
                        (f"worst localized match={worst_match:.2e}", worst_match > float(thresholds["rho_match"])),
                    )
                    if bad
                )
                + f" (tolerances cs<={thresholds['cs_relerr']:g}, rho<={thresholds['rho_match']:g};"
                f" ring seam {lm.potential_wrap_mismatch:.2e} is the floor)"
            )
        )

        cases[name] = {
            "c": target,
            "L": grid.L,
            "N": grid.N,
            "n_sites": lm.n_sites,
            "T": lm.T,
            "dt": lm.dt,
            "n_steps": lm.n_steps,
            "elapsed_sec": round(lm.elapsed_sec, 2),
            "cs_relerr": lm.cs_relerr,
            "tw_residual_max": lm.tw_residual_max,
            "potential_wrap_mismatch": lm.potential_wrap_mismatch,
            "rho_translation_err": rho_translation_err,
            "rho_partner_target": partner,
            "rho_partner_err": rho_partner_err,
            "max_abs_rho": float(np.max(np.abs(rho))),
            "leading_multipliers": [[float(x.real), float(x.imag)] for x in leading],
            "pencil_match": match,
            "pencil_match_max_abs_err": worst_match,
            "status": status,
        }
        table.append(
            [
                name,
                target,
                lm.cs_relerr,
                lm.tw_residual_max,
                lm.potential_wrap_mismatch,
                rho_translation_err,
                rho_partner_err,
                worst_match,
                cases[name]["max_abs_rho"],
            ]
        )
        print(f"    {name}: {status}", flush=True)

    print_table(
        "(b) monodromy of the linearized FK lattice, by direct RK4 time integration",
        ["case", "c", "cs relerr", "TW resid", "ring seam", "|rho-1|", "|rho-e^-g/c|", "pencil match", "max|rho|"],
        table,
        ["", ".4f", ".2e", ".2e", ".2e", ".2e", ".2e", ".2e", ".6f"],
    )
    bad = [name for name, rec in cases.items() if not rec["status"].startswith("ok")]
    overall = (
        "ok: " + ", ".join(sorted(cases)) + " all reproduced by direct time integration"
        if not bad
        else "inconclusive for " + ", ".join(bad) + " -- see the per-case status"
    )
    print(f"\n    direct_monodromy/status: {overall}")
    return {"status": overall, **cases}


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(
        __doc__ or "", "configs/07_conformal_symplectic.yaml", "data/07_conformal.json"
    )
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    payload["random_K"] = run_random_K(run)
    payload["sweep"] = run_sweep(run)
    payload["direct_monodromy"] = run_direct(cfg, run)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=["random_K", "sweep", "direct_monodromy"],
        elapsed_sec=elapsed,
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
