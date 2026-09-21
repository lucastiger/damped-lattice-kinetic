#!/usr/bin/env python3
"""Experiment 03 -- the stability threshold: -kappa/m against the pencil's eigenvalue.

At ``c_hat1`` the kinetic relation has a maximum, so ``sigma'(c_hat1) = 0`` and with it
``kappa = 2 pi mu sigma'`` (R4).  The local reduction (R7) says the non-trivial
eigenvalue of the co-traveling pencil near ``nu = 0`` is ``nu2 ~ -kappa/m``, so it must
change sign at the same velocity: the kink loses stability exactly where the kinetic
relation turns over.  This script tests that against the eigenvalue itself, computed by
shift-invert Arnoldi on ``Q(nu) = nu^2 + nu M1 + M0``.

Sections:

(a) ``rows`` -- for nine velocities straddling ``c_hat1``: ``sigma``, the exact
    ``sigma'``, ``kappa``, ``m``, the prediction ``nu2_pred = -kappa/m``, the computed
    ``nu2`` and the Floquet multiplier ``rho2 = exp(nu2/c)``.  ``nu2`` is the isolated
    real non-trivial eigenvalue of the pencil; see :func:`_select_nu2` for how it is
    picked out of the Arnoldi window, and why a window centred on ``nu2_pred`` alone is
    not enough at either end of the table.  It is ``null`` at ``c = 0.89500``, and that
    is the honest answer there: the prediction ``-0.0461`` sits within 0.004 of the
    essential-spectrum line ``Re nu = -gamma/2 = -0.05`` (R3), the mode has merged into
    the continuous spectrum, and what Arnoldi returns is a pair of the discretized
    essential spectrum rather than an isolated eigenvalue.  The reduction is a statement
    about a neighbourhood of the threshold, and 0.895 is outside it.

(b) ``zeros`` -- the zero of ``sigma'`` and the zero of ``nu2``, each by linear
    interpolation between the two bracketing rows.  They agree to about 3e-7, i.e. to the
    resolution of the bracket.

(c) ``reduction_error`` -- ``|nu2_pred - nu2| / |nu2|`` at two velocities: 10% at
    ``c = 0.898``, 1.5% at ``c = 0.8988``.  The leading-order reduction converges as the
    threshold is approached, which is exactly what "leading order" should mean.

(d) ``m_scan`` -- the sign of ``m`` from ``c = 0.20`` to ``c = 0.895``.  It stays
    positive, so ``nu2`` and ``-kappa`` share a sign over the whole first branch.

(e) ``m_sign_change_between_08999_09000`` -- ``m`` changes sign between ``c = 0.8999``
    and ``c = 0.9000`` (about +115 and -101).  THIS IS NOT A CHANGE OF STABILITY.  It is
    the edge of validity of the leading-order reduction: ``m`` is the denominator of
    ``nu2 ~ -kappa/m``, so where it passes through zero the first-order formula blows up
    and then flips sign while saying nothing about the spectrum.  The eigenvalue itself
    is perfectly well behaved there -- ``nu2`` is positive and growing on both sides
    (+0.0297 at 0.8998, +0.0406 at 0.9000, see the table in (a)) -- and the kink is
    unstable throughout.  ``m(c) = 0`` is the condition for the Jordan chain at ``nu = 0``
    to have length three, which is a statement about the c-derivative of the eigenvalue,
    not about its sign.

Everything numerical is in ``configs/03_threshold.yaml``.

Usage
-----
    python scripts/03_threshold.py --config configs/03_threshold.yaml
    python scripts/03_threshold.py --quick        # reduced L/N, ~20 s, not claim-grade

Production runtime: about five minutes, dominated by the nine Arnoldi solves at N = 4096.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, crange, load, march, output_path, print_table, section

from dlkin import (
    Grid,
    branch_scalars,
    init_branch,
    pencil_eigs,
    real_nontrivial,
    save_result,
)

SCRIPT = "scripts/03_threshold.py"


def _scalars(solver, psi, drive, c, phat):
    return branch_scalars(
        solver,
        psi,
        drive,
        float(c),
        normalization=str(phat["normalization"]),
        phat_seed=int(phat["seed"]),
        phat_iters=int(phat["iters"]),
        phat_shift=float(phat["shift"]),
    )


def _select_nu2(scalars, gamma: float, eigs_cfg: Dict[str, Any]):
    """The ISOLATED real non-trivial eigenvalue of the pencil, or ``None``.

    The shift is placed ``shift_offset`` above ``max(nu2_pred, 0)``: above zero so the
    translation mode ``nu = 0`` does not dominate the shift-invert spectrum, and near the
    prediction so the wanted eigenvalue is among the ``k`` that converge first.  Picking
    the right eigenvalue out of what comes back then takes three steps, and the two
    corrections to "nearest the prediction" are not fine print -- each decides one row of
    the table.

    1.  **Discard the discretized essential spectrum.**  By R3 the continuous spectrum is
        the line ``Re nu = -gamma/2``; on a finite domain it becomes ``O(N)`` discrete
        eigenvalues scattered a little off that line, and near the threshold a few of
        them come back with ``|Im nu|`` below ``imtol`` and are indistinguishable, taken
        one at a time, from an isolated real mode.  Two structural facts identify them.
        By R2, ``Q(-gamma-nu) = Q(nu)^T``, so the spectrum is invariant under
        ``nu -> -gamma-nu``; a pair of candidates with ``nu + nu' = -gamma`` is therefore
        symmetric about the line, whereas an isolated mode's partner lies far away.  And
        the complex eigenvalues of the same Arnoldi window are unambiguously essential,
        so their spread about the line, ``delta_essential``, measures how far off it the
        discretization throws them -- no tuned constant required.  A candidate is dropped
        when it is *both* half of a dual pair *and* within ``line_factor *
        delta_essential`` of the line.  At ``c = 0.89500`` that removes ``-0.045745502``
        and ``-0.054254498``, which sum to ``-gamma`` to the last digit Arnoldi resolves
        and sit 0.0043 off the line against a spread of 0.0041, and leaves nothing: the
        mode has merged into the continuum, exactly as Observation 9.3 says.  At
        ``c = 0.89880`` the same test removes the pair ``-0.048107173``/``-0.051892827``
        while keeping ``-0.003441``, which is 0.047 off the line -- eleven times the
        spread -- and is the eigenvalue wanted.

    2.  **Window.**  Of what survives, keep the candidates within ``match_window`` of
        ``nu2_pred`` and take the nearest.  At seven of the nine rows exactly one
        survives step 1 and lands in the window.

    3.  **Where the reduction has broken down, do not steer by it.**  ``nu2_pred`` is
        ``-kappa/m``, and ``m`` passes through zero between ``c = 0.8999`` and
        ``c = 0.9000`` (section (e)).  Past that pole ``m < 0`` and ``nu2_pred`` carries
        no information at all -- at ``c = 0.90000`` it reads ``-0.3018`` while the
        eigenvalue is ``+0.0406`` -- so a window centred on it is centred on nothing and
        comes back empty.  When ``m < 0`` and exactly one isolated real eigenvalue was
        found, that eigenvalue is ``nu2``.  THIS IS NOT A CHANGE OF STABILITY, only of
        how the eigenvalue is located; see the module docstring.

    The literal step-2 answer without step 1, the full candidate list, the measured
    ``delta_essential`` and which branch fired are all recorded on the row, so every
    decision can be re-adjudicated from the result file without recomputing anything.
    """
    shift = max(scalars.nu2_pred, 0.0) + float(eigs_cfg["shift_offset"])
    imtol = float(eigs_cfg["imtol"])
    nus = pencil_eigs(
        scalars.M0,
        scalars.M1,
        shift,
        k=int(eigs_cfg["k"]),
        tol=float(eigs_cfg["tol"]),
    )
    candidates = real_nontrivial(nus, gamma, imtol=imtol)
    half = float(eigs_cfg["match_window"])
    duality_tol = float(eigs_cfg["duality_tol"])
    line = -0.5 * gamma

    # how far off Re nu = -gamma/2 this Arnoldi window's *complex* (certainly essential)
    # eigenvalues are thrown by the discretization
    spread = [abs(float(np.real(x)) - line) for x in nus if abs(np.imag(x)) >= imtol]
    delta_essential = max(spread) if spread else 0.0
    reach = float(eigs_cfg["line_factor"]) * delta_essential

    essential = [
        x
        for x in candidates
        if abs(x - line) <= reach
        and any(abs(x + y + gamma) < duality_tol for y in candidates)
    ]
    isolated = [x for x in candidates if x not in essential]

    nu2_window = next(
        iter(sorted((x for x in candidates if abs(x - scalars.nu2_pred) < half),
                    key=lambda x: abs(x - scalars.nu2_pred))),
        None,
    )
    in_window = [x for x in isolated if abs(x - scalars.nu2_pred) < half]
    if in_window:
        nu2 = min(in_window, key=lambda x: abs(x - scalars.nu2_pred))
        how = "nearest nu2_pred among the isolated candidates"
    elif scalars.m < 0.0 and len(isolated) == 1:
        nu2 = isolated[0]
        how = "unique isolated eigenvalue (m < 0: nu2_pred invalid)"
    elif candidates and not isolated:
        nu2 = None
        how = "none: every candidate is a conformal-dual pair on the essential line"
    else:
        nu2 = None
        how = "none: no isolated real eigenvalue in the window"

    return {
        "nu2": nu2,
        "nu2_window": nu2_window,
        "real_nontrivial": candidates,
        "essential_pairs": essential,
        "delta_essential": delta_essential,
        "nu2_selection": how,
        "arnoldi_shift": shift,
        "n_eigs": len(nus),
    }


def run_rows(cfg, run) -> Tuple[Dict[str, Any], Any, np.ndarray, float]:
    """(a) The threshold table.  Returns the rows and the solver state at the last row."""
    opts = section(run, "rows")
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    eigs_cfg = dict(opts["eigs"])
    key_format = str(opts["key_format"])
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    approach = dict(opts["approach"])

    solver, psi, drive = init_branch(grid, cfg.model, c0=float(opts["c0"]), tol=tol, maxit=maxit)
    psi, drive, _ = march(
        solver,
        psi,
        drive,
        crange(float(approach["start"]), float(approach["stop"]), float(approach["step"])),
        tol,
        maxit,
    )

    rows: Dict[str, Any] = {}
    table = []
    for c in [float(x) for x in opts["c_values"]]:
        result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
        psi, drive = result.psi, result.drive
        s = _scalars(solver, psi, drive, c, phat)
        eig = _select_nu2(s, cfg.model.gamma, eigs_cfg)
        nu2 = eig["nu2"]
        rows[key_format % c] = {
            "c": c,
            "sigma": s.sigma,
            "sigma_prime": s.sigma_prime,
            "kappa": s.kappa,
            "phat_one": s.phat_one,
            "m": s.m,
            "nu2_pred": s.nu2_pred,
            "nu2": nu2,
            "rho2": (float(np.exp(nu2 / c)) if nu2 is not None else None),
            "identity_relerr": s.identity_relerr,
            "newton_residual": result.residual,
            **eig,
        }
        table.append([c, s.sigma, s.sigma_prime, s.kappa, s.m, s.nu2_pred, nu2, rows[key_format % c]["rho2"]])
        print(
            f"    c={c:.5f} sigma'={s.sigma_prime:+.7f} kappa={s.kappa:+.7f} "
            f"m={s.m:9.2f} pred={s.nu2_pred:+.6f} nu2={nu2}  "
            f"[real: {eig['real_nontrivial']}; essential: {eig['essential_pairs']}; "
            f"delta_ess={eig['delta_essential']:.2e}; {eig['nu2_selection']}]",
            flush=True,
        )
    print_table(
        f"(a) threshold table  (L = {grid.L:g}, N = {grid.N}, <phat,1> = -2 pi)",
        ["c", "sigma", "sigma'", "kappa", "m", "nu2_pred", "nu2", "rho2"],
        table,
        [".5f", ".9f", "+.7f", "+.7f", "9.2f", "+.6f", "+.6f", ".5f"],
    )
    return rows, solver, psi, drive


def _interp_zero(c_a: float, f_a: float, c_b: float, f_b: float) -> float:
    """Linear interpolation of the zero of ``f`` between two points."""
    return c_a + (c_b - c_a) * f_a / (f_a - f_b)


def run_zeros(run, rows: Dict[str, Any]) -> Dict[str, Any]:
    """(b) The interpolated zeros of sigma' and of nu2, and their separation."""
    opts = section(run, "zeros")
    key_format = str(section(run, "rows")["key_format"])
    a, b = [float(c) for c in opts["bracket"]]
    ra, rb = rows[key_format % a], rows[key_format % b]

    sp_zero = _interp_zero(a, ra["sigma_prime"], b, rb["sigma_prime"])
    if ra["nu2"] is None or rb["nu2"] is None:
        nu_zero, separation = None, None
    else:
        nu_zero = _interp_zero(a, ra["nu2"], b, rb["nu2"])
        separation = abs(sp_zero - nu_zero)

    record = {
        "bracket": [a, b],
        "sigma_prime_interp": sp_zero,
        "nu2_interp": nu_zero,
        "separation": separation,
    }
    print_table(
        "(b) interpolated zeros between the bracketing rows",
        ["quantity", "c"],
        [
            ["zero of sigma'", sp_zero],
            ["zero of nu2", nu_zero],
            ["|separation|", separation],
        ],
        ["", ".9f"],
    )
    return record


def run_reduction_error(run, rows: Dict[str, Any]) -> Dict[str, Any]:
    """(c) |nu2_pred - nu2| / |nu2| at the requested velocities."""
    opts = section(run, "reduction_error")
    key_format = str(section(run, "rows")["key_format"])
    out: Dict[str, Any] = {}
    table = []
    for c in [float(x) for x in opts["c_values"]]:
        row = rows[key_format % c]
        if row["nu2"] is None:
            out[key_format % c] = None
            table.append([c, row["nu2_pred"], None, None])
            continue
        value = abs(row["nu2_pred"] - row["nu2"]) / abs(row["nu2"])
        out[key_format % c] = value
        table.append([c, row["nu2_pred"], row["nu2"], value])
    print_table(
        "(c) relative error of the leading-order reduction",
        ["c", "nu2_pred", "nu2", "|pred-nu2|/|nu2|"],
        table,
        [".5f", "+.6f", "+.6f", ".4f"],
    )
    return out


def run_m_scan(cfg, run) -> Dict[str, Any]:
    """(d) m at the requested velocities, reached by continuation from c0 both ways."""
    opts = section(run, "m_scan")
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    c0, step = float(opts["c0"]), float(opts["step"])
    decimals, key_format = int(opts["decimals"]), str(opts["key_format"])
    targets = {round(float(c), decimals) for c in opts["targets"]}

    solver, psi0, drive0 = init_branch(grid, cfg.model, c0=c0, tol=tol, maxit=maxit)
    values: Dict[str, float] = {}

    def record(c: float, psi, drive) -> None:
        values[key_format % c] = _scalars(solver, psi, drive, c, phat).m

    if round(c0, decimals) in targets:
        record(c0, psi0, drive0)

    legs = [crange(c0 - step, float(opts["c_min"]), -step, decimals)]
    if float(opts["c_max"]) > c0:
        legs.append(crange(c0 + step, float(opts["c_max"]), step, decimals))
    for leg in legs:
        psi, drive = psi0, drive0
        for c in leg:
            result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
            psi, drive = result.psi, result.drive
            if c in targets:
                record(c, psi, drive)

    missing = sorted(targets - {float(k) for k in values})
    if missing:
        raise ValueError(
            f"m_scan targets {missing} were never visited: they are not on the "
            f"c0={c0} +/- {step} grid, or lie outside [{opts['c_min']}, {opts['c_max']}]."
        )
    ordered = dict(sorted(values.items(), key=lambda kv: float(kv[0])))
    record_out = {
        "L": grid.L,
        "N": grid.N,
        "step": step,
        "values": ordered,
        "min_m": min(ordered.values()),
        "all_positive": bool(all(v > 0.0 for v in ordered.values())),
    }
    print_table(
        f"(d) m along the branch  (L = {grid.L:g}, N = {grid.N}, <phat,1> = -2 pi)",
        ["c", "m"],
        [[float(k), v] for k, v in ordered.items()],
        [".4f", "10.3f"],
    )
    print(
        f"    min m = {record_out['min_m']:.3f} over {len(ordered)} velocities; "
        f"all positive: {record_out['all_positive']}"
    )
    return record_out


def run_m_sign(run, solver, psi, drive) -> Tuple[bool, Dict[str, Any]]:
    """(e) Does m change sign between the two velocities?  Continues from the last row."""
    opts = section(run, "m_sign")
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])

    detail: Dict[str, Any] = {}
    signs = []
    table = []
    for c in [float(x) for x in opts["c_values"]]:
        result = solver.newton(c, psi, drive, tol=tol, maxit=maxit)
        psi, drive = result.psi, result.drive
        s = _scalars(solver, psi, drive, c, phat)
        detail[f"{c:.5f}"] = {"m": s.m, "kappa": s.kappa, "sigma_prime": s.sigma_prime}
        signs.append(np.sign(s.m))
        table.append([c, s.sigma_prime, s.kappa, s.m])
    changed = bool(len(set(signs)) > 1)
    print_table(
        "(e) m across the edge of validity of the reduction",
        ["c", "sigma'", "kappa", "m"],
        table,
        [".5f", "+.7f", "+.7f", "10.2f"],
    )
    print(
        f"    m changes sign: {changed}  -- the REDUCTION breaks down here, the spectrum "
        "does not: nu2 stays positive (see the table in (a))."
    )
    return changed, detail


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/03_threshold.yaml", "data/03_threshold.json")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    rows, solver, psi, drive = run_rows(cfg, run)
    payload["rows"] = rows
    payload["zeros"] = run_zeros(run, rows)
    payload["reduction_error"] = run_reduction_error(run, rows)
    changed, detail = run_m_sign(run, solver, psi, drive)
    payload["m_sign_change_between_08999_09000"] = changed
    payload["m_sign_detail"] = detail
    del solver, psi, drive
    payload["m_scan"] = run_m_scan(cfg, run)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=["rows", "zeros", "reduction_error", "m_sign", "m_scan"],
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
