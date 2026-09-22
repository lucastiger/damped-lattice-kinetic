#!/usr/bin/env python3
"""Experiment 06 -- a snapshot of the pencil spectrum, either side of the threshold.

Shift-invert Arnoldi on ``Q(nu) = nu^2 + nu M1 + M0`` with the shift placed on the
essential-spectrum line ``Re nu = -gamma/2``, at two velocities: ``c = 0.89``, below the
first extremum, and ``c = 0.8995``, above it.

What the window contains splits cleanly into what must be there and what is a result.

**Must be there.**  ``nu = 0``, because ``phi' in ker M0`` by translation invariance; and
``nu = -gamma``, because R2 (``Q(-gamma-nu) = Q(nu)^T``) makes the spectrum invariant under
``nu -> -gamma - nu``, so the partner of ``0`` is an eigenvalue too.  ``nu_zero_abs`` and
``nu_minus_gamma`` locate them, and ``pairing_max_err`` checks the whole computed set for
closure under that involution.  These are checks on the computation, not findings: a
discretization that got them wrong would not be worth reading further.  (The shift is on
the line of symmetry precisely so that the window is nearly self-dual and the check has
something to bite on; a window centred elsewhere would fail it at its own rim, where a
partner can lie outside the ``k`` eigenvalues that converged.)

**The result.**  ``n_real_nontrivial``, the number of real eigenvalues left after those two
are removed.  At ``c = 0.89`` it is ``0``: the kink is spectrally stable, no real
multiplier has left the unit circle, and the rest of the window is the discretized
essential spectrum sitting on ``Re nu = -gamma/2``, i.e. on the multiplier circle of radius
``exp(-gamma/(2c)) = 0.945369``.  At ``c = 0.8995`` it is ``1``, at ``nu ~ +0.0176``,
``rho ~ 1.0198`` -- the mode that crossed at ``c_hat1``.

Both the eigenvalues and the multipliers ``rho = exp(nu/c)`` are written to CSV, so the
figure can show the ``nu``-plane and the ``rho``-plane side by side.

Everything numerical is in ``configs/06_spectrum.yaml``.

Usage
-----
    python scripts/06_spectrum.py --config configs/06_spectrum.yaml
    python scripts/06_spectrum.py --quick        # reduced L/N, ~15 s, not claim-grade

Production runtime: about two minutes.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, crange, load, march, output_path, path_to, print_table, section

from dlkin import branch_scalars, classify_eigs, init_branch, pencil_eigs, save_result

SCRIPT = "scripts/06_spectrum.py"
SPECTRUM_COLUMNS = ("nu_real", "nu_imag", "rho_real", "rho_imag", "abs_rho")


def run_blocks(cfg, run) -> Dict[str, Any]:
    """One spectral snapshot per configured velocity, warm-started along the branch."""
    newton = section(run, "newton")
    phat = section(run, "phat")
    eigs_cfg = section(run, "eigs")
    approach = dict(run["approach"])
    blocks = dict(run["blocks"])
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    gamma = cfg.model.gamma
    grid = cfg.grid

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

    out: Dict[str, Any] = {}
    summary = []
    for name, target in blocks.items():
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
        scalars = branch_scalars(
            solver,
            psi,
            drive,
            target,
            normalization=str(phat["normalization"]),
            phat_seed=int(phat["seed"]),
            phat_iters=int(phat["iters"]),
            phat_shift=float(phat["shift"]),
        )
        nus = pencil_eigs(
            scalars.M0,
            scalars.M1,
            float(eigs_cfg["shift"]),
            k=int(eigs_cfg["k"]),
            tol=float(eigs_cfg["tol"]),
        )
        info = classify_eigs(nus, gamma, imtol=float(eigs_cfg["imtol"]))
        rho = info.rho(target)
        # which eigenvalue is worst-paired, and whether its partner simply fell outside
        # the window: an isolated mode at nu has its dual partner at -gamma-nu, which a
        # window centred on -gamma/2 only reaches while |nu + gamma/2| is small enough
        partner_err = np.abs(nus[:, None] + gamma + nus[None, :]).min(axis=1)
        worst = int(np.argmax(partner_err))
        worst_nu = complex(nus[worst])
        partner_reach = float(np.max(np.abs(nus - float(eigs_cfg["shift"]))))
        partner_outside = bool(
            abs(-gamma - worst_nu - float(eigs_cfg["shift"])) > partner_reach
        )

        csv_path = Path(str(run["csv_template"]).format(block=name))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SPECTRUM_COLUMNS)
            for nu, r in zip(nus, rho):
                writer.writerow([nu.real, nu.imag, r.real, r.imag, abs(r)])

        out[name] = {
            "c": target,
            "L": grid.L,
            "N": grid.N,
            "shift": float(eigs_cfg["shift"]),
            "k": int(eigs_cfg["k"]),
            "n_eigs": int(nus.size),
            "sigma": scalars.sigma,
            "nu_zero_abs": info.nu_zero_abs,
            "nu_minus_gamma": float(info.nu_minus_gamma.real),
            "nu_minus_gamma_imag": float(info.nu_minus_gamma.imag),
            "pairing_max_err": info.pairing_max_err,
            "pairing_worst_nu": [float(worst_nu.real), float(worst_nu.imag)],
            "pairing_worst_partner_outside_window": partner_outside,
            "window_radius": partner_reach,
            "n_real_nontrivial": info.n_real_nontrivial,
            "real_nontrivial": info.real_nontrivial,
            # the essential spectrum lies on Re nu = -gamma/2, i.e. |rho| = exp(-gamma/2c)
            "circle_radius": float(np.exp(-gamma / (2.0 * target))),
            "max_abs_rho": float(np.max(np.abs(rho))),
            "essential_re_max_dev": float(
                np.max(np.abs(np.real(nus[np.abs(np.imag(nus)) >= float(eigs_cfg["imtol"])]) + gamma / 2.0))
                if np.any(np.abs(np.imag(nus)) >= float(eigs_cfg["imtol"]))
                else 0.0
            ),
            "csv": str(csv_path),
            "columns": list(SPECTRUM_COLUMNS),
        }
        summary.append(
            [
                name,
                target,
                out[name]["nu_zero_abs"],
                out[name]["nu_minus_gamma"],
                out[name]["pairing_max_err"],
                out[name]["n_real_nontrivial"],
                out[name]["real_nontrivial"][0] if info.real_nontrivial else None,
                out[name]["circle_radius"],
                out[name]["max_abs_rho"],
            ]
        )
        note = (
            f" [worst-paired nu={worst_nu.real:+.6f}, its partner "
            f"{-gamma - worst_nu.real:+.6f} lies outside the window of radius "
            f"{partner_reach:.4f} -- not a failure of R2]"
            if partner_outside
            else ""
        )
        print(
            f"    {name}: c={target} min|nu|={info.nu_zero_abs:.3e} "
            f"nu(-gamma)={info.nu_minus_gamma.real:.9f} pair={info.pairing_max_err:.2e} "
            f"n_real_nontrivial={info.n_real_nontrivial} {info.real_nontrivial}{note}",
            flush=True,
        )

    print_table(
        f"pencil spectrum  (L = {grid.L:g}, N = {grid.N}, shift = {eigs_cfg['shift']}, "
        f"k = {eigs_cfg['k']})",
        ["block", "c", "min|nu|", "nu near -gamma", "pairing err", "n real", "nu2", "|rho| circle", "max|rho|"],
        summary,
        ["", ".4f", ".3e", ".9f", ".2e", "d", "+.6f", ".6f", ".6f"],
    )
    return out


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/06_spectrum.yaml", "data/06_spectrum.json")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    payload: Dict[str, Any] = dict(run_blocks(cfg, run))
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=list(run["blocks"]),
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
