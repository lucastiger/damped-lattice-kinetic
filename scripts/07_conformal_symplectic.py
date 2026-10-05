#!/usr/bin/env python3
"""Experiment 07 -- Proposition 3.1, ``M^T J M = e^{-gamma/c} J``, twice over.

``random_K``
    The structural statement.  ``A^T J + J A = -gamma J`` for ``A = [[0, I], [K, -gamma I]]``
    uses three facts and no more: Newtonian second-order form, a *symmetric* force Jacobian,
    and *uniform scalar* damping.  Nothing about the Frenkel-Kontorova lattice, the wave or
    the potential enters.  The test therefore resamples a random symmetric ``K(t)`` at every
    integration step: if the identity survives an arbitrary symmetric coupling that changes at
    every instant, it is not a property of the particular problem.  The one-site cyclic shift
    is applied at the end, because the monodromy of a traveling wave carries it and a
    permutation must not disturb a symplectic form.  The sweep over ``(n, dt)`` shows the
    residual is arithmetic rather than a discretization artefact: it does not move with the
    step.

``lattice``
    The same identity for the *actual* monodromy, by leaving the advance-delay formulation
    entirely.  The computed profile is sampled at the integer sites, the nonlinear lattice and
    its linearization are integrated over one period ``T = 1/c``, and the one-site shift is
    applied -- the method of [VCKX20], not ours.  Four things are measured:

    ``drift``            after one period the lattice state should equal the initial one shifted
                         by a site.  This is the number to read first: it says whether the
                         sampled xi-profile really is a lattice traveling wave, and it bounds
                         everything below.
    ``symplectic``       ``(Mx)^T J (My) = e^{-gamma T} x^T J y`` on random pairs, which is
                         ``M^T J M = e^{-gamma T} J`` without ever forming ``M``.
    ``translation``      ``M z = z`` for ``z = (phi'(n), -c phi''(n))``: the neutral multiplier
                         ``rho = 1`` with its predicted eigenvector, in one product.
    ``dominant``         power iteration for the largest-modulus multiplier, compared against
                         ``exp(nu/c)`` from the co-traveling pencil.  At ``c = 0.89`` that is
                         the neutral multiplier; at ``c = 0.8995`` it is the unstable one, and
                         agreement there is a comparison of two completely different methods.

    NONE of this is used by a claim in the manuscript.  It exists so that a drift between the
    xi-formulation and the lattice it is supposed to describe would fail loudly.

Usage
    python scripts/07_conformal_symplectic.py [--config ...] [--quick] [--only random_K,lattice]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, load, march, output_path, path_to, print_table, section

from dlkin import Grid, conformal_symplectic_flow_test, init_branch, lattice_monodromy, save_result
from dlkin.spectral import branch_scalars, pencil_eigs

SCRIPT = "scripts/07_conformal_symplectic.py"
SECTIONS = ("random_K", "lattice")


def run_random_K(run) -> Dict[str, Any]:
    sec = section(run, "random_K")
    ref = dict(sec["reference"])
    result = conformal_symplectic_flow_test(**ref)
    out: Dict[str, Any] = {"relerr": result["relerr"], "reference": result}

    sweep = []
    for n in sec["sweep"]["n"]:
        for dt in sec["sweep"]["dt"]:
            r = conformal_symplectic_flow_test(
                n=int(n), gamma=ref["gamma"], T=ref["T"], dt=float(dt), seed=ref["seed"]
            )
            sweep.append({"n": int(n), "dt": float(dt), "relerr": r["relerr"], "steps": r["steps"]})
    out["sweep"] = sweep
    out["sweep_max_relerr"] = float(max(s["relerr"] for s in sweep))

    print_table(
        "A^T J + J A = -gamma J, tested on a randomly resampled symmetric K(t)",
        ["n", "dt", "steps", "max|M^T J M - e^{-gT} J| / max|J|"],
        [[s["n"], s["dt"], s["steps"], s["relerr"]] for s in sweep],
        ["d", ".0e", "d", ".2e"],
    )
    return out


def run_lattice(cfg, run) -> Dict[str, Any]:
    sec = section(run, "lattice")
    newton = section(run, "newton")
    arn = dict(sec["arnoldi"])
    pen = dict(sec["pencil"])

    grid = Grid(L=cfg.grid.L, N=cfg.grid.N)
    solver, psi, drive = init_branch(
        grid, cfg.model, c0=float(sec["c0"]), tol=newton["tol"], maxit=newton["maxit"]
    )
    out: Dict[str, Any] = {}
    current = float(sec["c0"])
    rows = []
    for c in sec["c_values"]:
        c = float(c)
        psi, drive, _ = march(
            solver, psi, drive, path_to(current, c, float(sec["approach_step"])),
            newton["tol"], newton["maxit"],
        )
        current = c

        mono = lattice_monodromy(solver, psi, drive, c, steps=int(sec["steps"]))
        fine = lattice_monodromy(solver, psi, drive, c, steps=int(sec["steps_refine"]))
        trans = mono.translation_mode_residual()
        trans_fine = fine.translation_mode_residual()
        sym = mono.symplectic_residual(
            pairs=int(sec["symplectic_pairs"]), seed=int(sec["symplectic_seed"])
        )
        dom = mono.dominant_multiplier(iters=int(arn.get("power_iters", 250)), tol=1e-10)

        sc = branch_scalars(
            solver, psi, drive, c,
            normalization=run["phat"]["normalization"], phat_seed=run["phat"]["seed"],
            phat_iters=run["phat"]["iters"], phat_shift=run["phat"]["shift"],
        )
        shift = float(pen["shift"]) if sc.nu2_pred <= 0 or sc.m < 0 else max(sc.nu2_pred, 0.0) + 0.004
        nus = pencil_eigs(sc.M0, sc.M1, shift=shift, k=int(pen["k"]), tol=float(pen["tol"]))
        rho_pencil = np.exp(nus / c)
        j = int(np.argmin(np.abs(np.abs(rho_pencil) - abs(dom["rho"]))))

        entry = {
            "c": c,
            "n_sites": int(mono.sites.size),
            "steps": int(sec["steps"]),
            "drift_position": mono.drift_position,
            "drift_velocity": mono.drift_velocity,
            "drift_position_refined": fine.drift_position,
            "symplectic_residual": sym,
            "translation_residual_rel": trans["residual_rel"],
            "translation_rayleigh_minus_one": trans["rayleigh_minus_one"],
            "translation_residual_refined": trans_fine["residual_rel"],
            "dominant_rho": dom["rho"],
            "dominant_residual_rel": dom["residual_rel"],
            "dominant_iterations": dom["iterations"],
            "dominant_converged": dom["converged"],
            "pencil_rho_nearest": float(abs(rho_pencil[j])),
            "pencil_nu_nearest": float(nus[j].real),
            "dominant_vs_pencil": float(abs(abs(rho_pencil[j]) - abs(dom["rho"]))),
        }
        out[f"c{c:.5f}".replace(".", "p")] = entry
        rows.append(entry)

    out["status"] = "ok"
    out["worst_symplectic_residual"] = float(max(r["symplectic_residual"] for r in rows))
    out["worst_translation_residual"] = float(max(r["translation_residual_rel"] for r in rows))

    print_table(
        "the actual lattice monodromy (method of [VCKX20], independent of the xi-formulation)",
        ["c", "drift", "M^T J M resid", "M z = z resid", "dominant rho", "pencil rho", "diff"],
        [[r["c"], r["drift_position"], r["symplectic_residual"], r["translation_residual_rel"],
          r["dominant_rho"], r["pencil_rho_nearest"], r["dominant_vs_pencil"]] for r in rows],
        [".4f", ".2e", ".2e", ".2e", ".8f", ".8f", ".2e"],
    )
    return out


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/07_conformal_symplectic.yaml", "data/07_conformal.json")
    parser.add_argument("--only", default=None, help="comma-separated subset of " + ", ".join(SECTIONS))
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run
    wanted = SECTIONS if args.only is None else tuple(s.strip() for s in args.only.split(","))
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    if "random_K" in wanted:
        payload["random_K"] = run_random_K(run)
    if "lattice" in wanted:
        try:
            payload["direct_monodromy"] = run_lattice(cfg, run)
        except Exception as exc:  # noqa: BLE001 -- recorded, never silently swallowed
            payload["direct_monodromy"] = {"status": f"failed: {type(exc).__name__}: {exc}"}
            print(f"\n  direct lattice monodromy FAILED and is recorded as such: {exc}")
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out, payload, script=SCRIPT, quick=args.quick, config_path=args.config, config=raw,
        sections=list(wanted), elapsed_sec=elapsed, merge=args.only is not None,
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick}, sections={list(wanted)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
