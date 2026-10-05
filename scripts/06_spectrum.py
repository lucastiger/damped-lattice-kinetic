#!/usr/bin/env python3
"""Experiment 06 -- the spectrum of the co-traveling pencil near the origin.

Three structural facts of ``handoff/SCIENCE_BRIEF.md`` are visible in one picture and are
what this experiment measures:

* ``nu = 0`` is always an eigenvalue (eigenvector ``phi'``, translation invariance), and by
  the duality ``Q(-gamma-nu) = Q(nu)^T`` of R2 so is ``nu = -gamma``, i.e.
  ``rho = e^{-gamma/c}``.  Both are structural: they are there at every velocity, stable or
  not, and finding them to ``1e-7`` is a test of the whole discretization.
* The same duality pairs the rest of the spectrum, ``nu + nu_hat = -gamma``.  ``pairing_max_err``
  is the largest distance from ``-gamma-nu`` to the nearest other computed eigenvalue; it has
  no reason to be small unless the duality holds.
* The essential spectrum sits on ``Re nu = -gamma/2`` (R3), i.e. on the circle
  ``|rho| = e^{-gamma/(2c)}``, provided ``mu sqrt(1-sigma^2) > gamma^2/4`` -- which at these
  parameters it is, by a factor of 300.

Two velocities: ``c = 0.89``, below the threshold, where the only real eigenvalues are the two
structural ones; and ``c = 0.8995``, past ``c_hat1 = 0.8989297``, where exactly one real
multiplier has left the unit circle.  The eigenvalues of both are written to CSV for Figure 5.

A note on counting real eigenvalues near the essential spectrum: on a finite domain the
continuum becomes ``O(N)`` discrete eigenvalues scattered slightly off the line, and some are
returned with ``|Im nu|`` below the ``imtol`` used to call an eigenvalue real.  They are not
isolated modes, and ``n_real_nontrivial`` would be meaningless if they were counted.  The
duality identifies them: a pair of real candidates summing to ``-gamma`` is symmetric about the
line and belongs to the discretized continuum, whereas an isolated mode has its dual partner
far away.  That is the rule applied here, the same one experiment 03 uses.

Usage
    python scripts/06_spectrum.py [--config ...] [--quick]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, build_parser, load, march, output_path, path_to, print_table, section

from dlkin import Grid, init_branch, save_result
from dlkin.spectral import branch_scalars, classify_real_eigenvalues, pencil_eigs


SCRIPT = "scripts/06_spectrum.py"


def _structural_split(nus: np.ndarray, gamma: float, eigs_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The two structural eigenvalues, the isolated real ones, and the duality residual.

    The isolated/essential split is :func:`dlkin.spectral.classify_real_eigenvalues`, the same
    routine experiment 03 uses -- a candidate counts as discretized continuum only when it is
    both half of a dual pair and within ``line_factor`` spreads of ``Re nu = -gamma/2``.  The
    pairing test is restricted to the inner part of the Arnoldi window: the dual partner of an
    eigenvalue at the edge of the window need not have converged, so including it would
    measure the window size rather than the duality.
    """
    imtol = float(eigs_cfg["imtol"])
    zero = float(min(abs(x) for x in nus))
    # The two structural eigenvalues are resolved only as well as the discretization allows
    # (1e-8 at N=2048, 3e-6 at the quick resolution), so the cut that removes them has to
    # scale with what was actually achieved rather than being a fixed constant.
    zero_tol = max(float(eigs_cfg["zero_tol"]), 10.0 * zero)
    split = classify_real_eigenvalues(
        nus,
        gamma,
        imtol=imtol,
        duality_tol=float(eigs_cfg.get("duality_tol", 1e-6)),
        line_factor=float(eigs_cfg.get("line_factor", 2.0)),
        zero_tol=zero_tol,
    )

    near_gamma = min(nus, key=lambda x: abs(x + gamma))

    shift = complex(eigs_cfg["shift"])
    order = sorted(nus, key=lambda x: abs(x - shift))
    inner = order[: max(2, int(round(float(eigs_cfg.get("pairing_fraction", 0.6)) * len(order))))]
    pairing = max(float(min(abs(-gamma - x - y) for y in nus)) for x in inner)

    return {
        "nu_zero_abs": zero,
        "nu_minus_gamma": float(near_gamma.real),
        "nu_minus_gamma_imag": float(near_gamma.imag),
        "pairing_max_err": pairing,
        "pairing_tested": len(inner),
        "n_real_nontrivial": len(split["isolated"]),
        "n_unstable": int(sum(1 for x in split["isolated"] if x > 0.0)),
        "zero_tol_used": zero_tol,
        "real_nontrivial": split["isolated"],
        "essential_pairs": split["essential_pairs"],
        "delta_essential": split["delta_essential"],
        "n_computed": int(len(nus)),
    }


def run_point(cfg, run, spec, solver, psi, drive) -> Dict[str, Any]:
    eigs_cfg = section(run, "eigs")
    gamma = cfg.model.gamma
    c = float(spec["c"])
    sc = branch_scalars(
        solver, psi, drive, c,
        normalization=run["phat"]["normalization"], phat_seed=run["phat"]["seed"],
        phat_iters=run["phat"]["iters"], phat_shift=run["phat"]["shift"],
    )
    nus = pencil_eigs(sc.M0, sc.M1, shift=eigs_cfg["shift"], k=eigs_cfg["k"], tol=eigs_cfg["tol"])
    out = _structural_split(nus, gamma, eigs_cfg)
    out.update(
        {
            "c": c,
            "sigma": sc.sigma,
            "sigma_prime": sc.sigma_prime,
            "circle_radius": float(np.exp(-gamma / (2.0 * c))),
            "dual_line_re_nu": -gamma / 2.0,
            "essential_condition": float(cfg.model.mu * np.sqrt(1.0 - sc.sigma ** 2) - gamma ** 2 / 4.0),
            "shift": eigs_cfg["shift"],
            "k": eigs_cfg["k"],
        }
    )

    csv_path = REPO_ROOT / spec["csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["re_nu", "im_nu", "abs_rho", "arg_rho"])
        for x in sorted(nus, key=lambda z: -z.real):
            rho = np.exp(x / c)
            writer.writerow([f"{x.real:.12e}", f"{x.imag:.12e}", f"{abs(rho):.12e}", f"{np.angle(rho):.12e}"])
    out["csv"] = spec["csv"]

    print_table(
        f"pencil spectrum near the origin at c = {c}",
        ["|nu| nearest 0", "nu nearest -gamma", "pairing err", "isolated real", "circle |rho|"],
        [[out["nu_zero_abs"], out["nu_minus_gamma"], out["pairing_max_err"],
          out["n_real_nontrivial"], out["circle_radius"]]],
        [".2e", ".9f", ".2e", "d", ".6f"],
    )
    if out["real_nontrivial"]:
        print("    isolated real eigenvalues:",
              ", ".join(f"{r:+.6f} (rho={np.exp(r / c):.5f})" for r in out["real_nontrivial"]))
    if out["essential_pairs"]:
        print("    discarded as discretized continuum (dual pairs on Re nu = -gamma/2, "
              f"spread {out['delta_essential']:.2e}):",
              ", ".join(f"{x:+.6f}" for x in out["essential_pairs"]))
    return out


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/06_spectrum.yaml", "data/06_spectrum.json")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run
    newton = section(run, "newton")
    points = section(run, "points")

    started = time.perf_counter()
    grid = Grid(L=cfg.grid.L, N=cfg.grid.N)
    solver, psi, drive = init_branch(
        grid, cfg.model, c0=float(run["c0"]), tol=newton["tol"], maxit=newton["maxit"]
    )
    payload: Dict[str, Any] = {}
    current = float(run["c0"])
    for spec in sorted(points.values(), key=lambda s: float(s["c"])):
        values = path_to(current, float(spec["c"]), float(run["approach_step"]))
        psi, drive, _ = march(solver, psi, drive, values, newton["tol"], newton["maxit"])
        current = float(spec["c"])
        payload[spec["key"]] = run_point(cfg, run, spec, solver, psi, drive)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out, payload, script=SCRIPT, quick=args.quick, config_path=args.config, config=raw,
        sections=list(payload), elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
