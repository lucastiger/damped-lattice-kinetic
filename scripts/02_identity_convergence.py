#!/usr/bin/env python3
"""Experiment 02 -- the Jordan-chain identity, its convergence, and the exact sigma'(c).

The identity (R4 of ``handoff/SCIENCE_BRIEF.md``) is

    kappa(c) = <phat, M1 phi'>_h = -mu sigma'(c) <phat,1>_h ,

which with the ``<phat,1>_h = -2 pi`` normalization used throughout this script reads
``kappa = 2 pi mu sigma'(c)``.  Nothing in the computation imposes it: ``kappa`` is a
pairing of the left null vector against ``M1 phi'``, ``sigma'`` comes from a bordered
linear solve, and ``<phat,1>`` from the normalization.  The relative error between the
two sides is therefore a genuine test of the discretization, and is what (b) tabulates.

Sections (select with ``--only``; several may be given, comma-separated):

``c089``
    All the scalars of Observation 9.2 at ``c = 0.89``, ``L = 200``, ``N = 4096``: the
    identity's two sides, the free orthogonality ``<phat, V''(phi)> = 0`` (Lemma 4.2), the
    power balance (R6), and the far-field branch derivative ``d_c phi`` against its
    closed form ``sigma'/sqrt(1-sigma^2)``.

``convergence``
    The same relative error at six resolutions.  It is a *spectral* convergence: 3e-4 at
    ``h ~ 0.39``, 1.3e-8 at ``h ~ 0.195``, below 1e-12 at ``h ~ 0.098``.  ``res_M0p0``,
    the residual ``|M0 phi'|_inf / |phi'|_inf`` of the kernel relation, tracks it.
    ``L300_N6144`` is the heaviest computation in the repository: 6145 x 6145 dense
    matrices, about 300 MB each, several live at once, ~2 GB resident and about two
    minutes of wall-clock on four cores.  Run it on its own with

        python scripts/02_identity_convergence.py --only convergence --cases L300_N6144

    which merges the one case into an existing ``data/02_identity.json`` rather than
    replacing the file.

``sweep``
    The maximum relative error along the whole first branch, from ``c = 0.5`` up to
    ``c = 0.90015`` -- past ``c_hat1``, where ``sigma'`` changes sign, and up to just
    short of the fold at ``c_max = 0.900196``.  The identity is an algebraic consequence
    of ``M0 [d_c phi] = M1 phi' + mu sigma' 1``, so it must hold at every velocity, not
    only at the convenient ones.

    READ THE RELATIVE ERROR WITH THE ABSOLUTE ONE BESIDE IT, which is why both are
    recorded.  The relative error peaks at ``c = 0.89892`` -- 1.8e-10, four orders worse
    than anywhere else -- and that peak is a property of the *denominator*, not of the
    identity.  ``c = 0.89892`` is 1e-5 from ``c_hat1``, so both sides of
    ``kappa = -mu sigma' <phat,1>`` have collapsed to 0.0747, 0.6% of their value at
    ``c = 0.89``; the absolute disagreement there, 1.3e-11, is in line with the 1.4e-12
    at ``c = 0.89`` and smaller than the 1.6e-11 at ``c = 0.8995``.  What sets that
    absolute floor is the accuracy of the computed left null vector: inverse iteration
    leaves ``|M0^T phat| / |phat| ~ 1.2e-13``, and dividing by the second singular value
    of ``M0`` (9.89e-2) puts the error in the direction of ``phat`` at ~1.3e-12, which
    paired against ``M1 phi'`` is ~1e-11.  Raising the iteration count to 400 or changing
    the seed moves it by a few percent; tightening Newton (residual 5.6e-13) and taking a
    different continuation path to the same velocity do not move it at all.

``fd``
    WHY THE EXACT DERIVATIVE IS USED.  A centred finite difference of ``sigma(c)`` with
    step ``1e-3`` -- a perfectly reasonable-looking choice, and the step the continuation
    itself walks with -- disagrees with the bordered-solve value by about 1.6e-4 in
    relative terms at ``c = 0.89``.  That is a truncation error of the difference
    formula, set by ``sigma'''(c)``, and it does NOT shrink with ``N`` or ``L``: it would
    put a floor of 1.6e-4 under every identity error in (b), destroying the 1e-12 result
    and with it any claim that the identity is verified rather than assumed.  ``step`` is
    the spacing between the two evaluation points, ``sigma(c +/- step/2)``; the record
    also carries the ``sigma(c +/- step)`` variant, four times worse because the
    truncation error is quadratic in the spacing.

Everything numerical is in ``configs/02_identity_convergence.yaml``.

Usage
-----
    python scripts/02_identity_convergence.py --config configs/02_identity_convergence.yaml
    python scripts/02_identity_convergence.py --only convergence --cases L300_N6144
    python scripts/02_identity_convergence.py --quick        # reduced L/N, ~25 s

Production runtime: about 11 minutes for everything, of which ``sweep`` is ~7 (166
Newton solves at N = 4096 to walk from c = 0.5 to the fold) and ``L300_N6144`` ~2.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, crange, load, march, output_path, path_to, print_table, section

from dlkin import Grid, branch_scalars, init_branch, save_result

SCRIPT = "scripts/02_identity_convergence.py"
SECTIONS = ("c089", "convergence", "sweep", "fd")


def _scalars_at(cfg, run, L: float, N: int, c0: float, approach, c: float):
    """Land on the branch at ``(L, N)``, march to ``c``, and return the scalars there."""
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(L), N=int(N))
    solver, psi, drive = init_branch(grid, cfg.model, c0=float(c0), tol=tol, maxit=maxit)
    values = crange(float(approach["start"]), float(approach["stop"]), float(approach["step"]))
    psi, drive, residual = march(solver, psi, drive, values, tol, maxit)
    scalars = branch_scalars(
        solver,
        psi,
        drive,
        float(c),
        normalization=str(phat["normalization"]),
        phat_seed=int(phat["seed"]),
        phat_iters=int(phat["iters"]),
        phat_shift=float(phat["shift"]),
    )
    return solver, psi, drive, scalars, residual


def run_c089(cfg, run) -> Dict[str, Any]:
    """(a) The full scalar record at c = 0.89."""
    opts = section(run, "c089")
    _, _, _, s, residual = _scalars_at(
        cfg, run, opts["L"], opts["N"], opts["c0"], dict(opts["approach"]), opts["c"]
    )
    record = {
        "L": float(opts["L"]),
        "N": int(opts["N"]),
        "c": float(opts["c"]),
        "normalization": s.normalization,
        "sigma": s.sigma,
        "sigma_prime": s.sigma_prime,
        "kappa": s.kappa,
        "phat_one": s.phat_one,
        # Lemma 4.2: M0 1 = mu cos phi, so <phat, V''(phi)> = <M0^T phat, 1> = 0 for free
        "phat_dot_cos": abs(s.phat_dot_Vpp),
        "m": s.m,
        "nu2_pred": s.nu2_pred,
        "identity_relerr": s.identity_relerr,
        "kappa_predicted": -s.drive_prime * s.phat_one,
        "power_balance_relerr": s.power_balance_relerr,
        "sigma_power_balance": s.sigma_power_balance,
        "dcphi_edge": s.dcphi_edge,
        "dcphi_edge_pred": s.dcphi_edge_pred,
        "res_M0p0": s.res_M0_phi_prime,
        "res_phat_rel": s.res_phat_rel,
        "newton_residual": residual,
    }
    print_table(
        f"(a) scalars at c = {record['c']}  (L = {record['L']:g}, N = {record['N']}, "
        f"<phat,1> = -2 pi)",
        ["quantity", "value"],
        [
            ["sigma", record["sigma"]],
            ["sigma' (exact)", record["sigma_prime"]],
            ["kappa = <phat, M1 phi'>", record["kappa"]],
            ["-mu sigma' <phat,1>", record["kappa_predicted"]],
            ["<phat,1>", record["phat_one"]],
            ["|<phat, cos phi>|", record["phat_dot_cos"]],
            ["m", record["m"]],
            ["identity relerr", record["identity_relerr"]],
            ["power balance relerr", record["power_balance_relerr"]],
            ["d_c phi at the edge", record["dcphi_edge"]],
            ["sigma'/sqrt(1-sigma^2)", record["dcphi_edge_pred"]],
            ["|M0 phi'|/|phi'|", record["res_M0p0"]],
            ["|M0^T phat|/|phat|", record["res_phat_rel"]],
        ],
        ["", ".12e"],
    )
    return record


def run_convergence(cfg, run, cases: List[str] | None) -> Dict[str, Any]:
    """(b) The identity error at each (L, N)."""
    opts = section(run, "convergence")
    all_cases: Dict[str, Any] = dict(opts["cases"])
    names = list(all_cases) if not cases else list(cases)
    unknown = [n for n in names if n not in all_cases]
    if unknown:
        raise ValueError(f"unknown convergence case(s) {unknown}; config has {list(all_cases)}.")

    out: Dict[str, Any] = {}
    rows = []
    for name in names:
        case = dict(all_cases[name])
        started = time.perf_counter()
        _, _, _, s, residual = _scalars_at(
            cfg, run, case["L"], case["N"], opts["c0"], dict(opts["approach"]), opts["c"]
        )
        elapsed = time.perf_counter() - started
        grid_h = 2.0 * float(case["L"]) / int(case["N"])
        out[name] = {
            "L": float(case["L"]),
            "N": int(case["N"]),
            "h": grid_h,
            "c": float(opts["c"]),
            "relerr": s.identity_relerr,
            "res_M0p0": s.res_M0_phi_prime,
            "kappa": s.kappa,
            "sigma_prime": s.sigma_prime,
            "newton_residual": residual,
            "elapsed_sec": round(elapsed, 2),
        }
        rows.append([name, case["L"], case["N"], grid_h, s.identity_relerr, s.res_M0_phi_prime, elapsed])
        print(f"    {name}: relerr={s.identity_relerr:.3e}  res={s.res_M0_phi_prime:.2e}  ({elapsed:.1f} s)", flush=True)
    print_table(
        "(b) convergence of the identity kappa = -mu sigma' <phat,1>",
        ["case", "L", "N", "h", "relerr", "res_M0p0", "sec"],
        rows,
        ["", "g", "d", ".4f", ".3e", ".2e", ".1f"],
    )
    return out


def run_sweep(cfg, run) -> Dict[str, Any]:
    """(c) The identity error along the whole branch, up to just short of the fold."""
    opts = section(run, "sweep")
    newton = section(run, "newton")
    phat = section(run, "phat")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(opts["L"]), N=int(opts["N"]))
    targets = sorted(float(c) for c in opts["targets"])
    walk, fine = float(opts["walk_step"]), float(opts["fine_step"])
    fine_from, decimals = float(opts["fine_from"]), int(opts["decimals"])

    solver, psi, drive = init_branch(grid, cfg.model, c0=float(opts["c0"]), tol=tol, maxit=maxit)
    current = float(opts["c0"])
    values: Dict[str, float] = {}
    abs_err: Dict[str, float] = {}
    rows = []

    def visit(target: float, step: float) -> None:
        nonlocal psi, drive, current
        psi, drive, _ = march(solver, psi, drive, path_to(current, target, step, decimals), tol, maxit)
        current = target
        s = branch_scalars(
            solver,
            psi,
            drive,
            target,
            normalization=str(phat["normalization"]),
            phat_seed=int(phat["seed"]),
            phat_iters=int(phat["iters"]),
            phat_shift=float(phat["shift"]),
        )
        values[f"{target:.5f}"] = s.identity_relerr
        # the absolute disagreement, which stays flat while the two sides pass through
        # zero at c_hat1 -- that is what makes the RELATIVE error peak there
        abs_err[f"{target:.5f}"] = abs(s.kappa + s.drive_prime * s.phat_one)
        rows.append(
            [target, s.sigma, s.sigma_prime, s.kappa, s.identity_relerr, abs_err[f"{target:.5f}"]]
        )
        print(
            f"    sweep c={target:.5f}: relerr={s.identity_relerr:.3e} "
            f"abs={abs_err[f'{target:.5f}']:.3e}",
            flush=True,
        )

    # walk down to the lowest target first, then up through all of them in order
    psi, drive, _ = march(solver, psi, drive, path_to(current, targets[0], walk, decimals), tol, maxit)
    current = targets[0]
    visit(targets[0], walk)
    for target in targets[1:]:
        visit(target, fine if max(current, target) > fine_from else walk)

    print_table(
        f"(c) identity error along the branch  (L = {grid.L:g}, N = {grid.N})",
        ["c", "sigma", "sigma'", "kappa", "relerr", "|kappa + f' <phat,1>|"],
        rows,
        [".5f", ".9f", "+.7f", "+.7f", ".3e", ".3e"],
    )
    worst = max(values, key=lambda k: values[k])
    worst_abs = max(abs_err, key=lambda k: abs_err[k])
    return {
        "L": grid.L,
        "N": grid.N,
        "values": values,
        "abs_err": abs_err,
        "max_relerr": values[worst],
        "argmax_c": float(worst),
        "max_abs_err": abs_err[worst_abs],
        "argmax_abs_c": float(worst_abs),
    }


def run_fd(cfg, run) -> Tuple[float, Dict[str, Any]]:
    """(d) Exact sigma'(c) against a centred finite difference of spacing ``fd.step``.

    ``step`` is the distance between the two points the difference is formed from --
    ``(sigma(c + step/2) - sigma(c - step/2)) / step`` -- which is the difference a
    continuation walking in steps of ``step`` can actually form from its own records.
    The wider ``(sigma(c + step) - sigma(c - step)) / (2 step)`` is reported alongside it:
    its points are twice as far apart, and the truncation error of a centred difference
    is quadratic in that distance, so it is four times worse (6.5e-4 against 1.6e-4).
    Neither is anywhere near the 1e-14 the bordered solve delivers.
    """
    opts = section(run, "fd")
    newton = section(run, "newton")
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    c, step = float(opts["c"]), float(opts["step"])
    solver, psi, drive, s, _ = _scalars_at(
        cfg, run, opts["L"], opts["N"], opts["c0"], dict(opts["approach"]), c
    )
    exact = s.sigma_prime

    def centred(spacing: float) -> Tuple[float, float, float]:
        half = 0.5 * spacing
        lo = solver.newton(c - half, psi, drive, tol=tol, maxit=maxit).sigma
        hi = solver.newton(c + half, psi, drive, tol=tol, maxit=maxit).sigma
        return lo, hi, (hi - lo) / spacing

    sigma_minus, sigma_plus, fd = centred(step)
    discrepancy = abs(exact - fd) / abs(exact)
    _, _, fd_wide = centred(2.0 * step)
    discrepancy_wide = abs(exact - fd_wide) / abs(exact)

    print_table(
        f"(d) exact sigma'(c) vs a centred difference  "
        f"(L = {opts['L']:g}, N = {opts['N']}, c = {c})",
        ["quantity", "value"],
        [
            [f"sigma({c - 0.5 * step:.6g})", sigma_minus],
            [f"sigma({c + 0.5 * step:.6g})", sigma_plus],
            [f"centred difference, points {step:g} apart", fd],
            [f"centred difference, points {2.0 * step:g} apart", fd_wide],
            ["exact (bordered solve)", exact],
            [f"relative discrepancy, points {step:g} apart", discrepancy],
            [f"relative discrepancy, points {2.0 * step:g} apart", discrepancy_wide],
        ],
        ["", ".12e"],
    )
    detail = {
        "L": float(opts["L"]),
        "N": int(opts["N"]),
        "c": c,
        "step": step,
        "convention": "(sigma(c + step/2) - sigma(c - step/2)) / step",
        "sigma_minus": sigma_minus,
        "sigma_plus": sigma_plus,
        "sigma_prime_exact": exact,
        "sigma_prime_fd": fd,
        "relative_discrepancy": discrepancy,
        "sigma_prime_fd_double_spacing": fd_wide,
        "relative_discrepancy_double_spacing": discrepancy_wide,
    }
    return discrepancy, detail


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(
        __doc__ or "", "configs/02_identity_convergence.yaml", "data/02_identity.json"
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of " + ", ".join(SECTIONS) + "; the result is merged "
        "into an existing --out file instead of replacing it",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="comma-separated subset of the convergence cases (e.g. L300_N6144)",
    )
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    wanted = SECTIONS if args.only is None else tuple(s.strip() for s in args.only.split(","))
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    cases = [c.strip() for c in args.cases.split(",")] if args.cases else None

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    if "c089" in wanted:
        payload["c089"] = run_c089(cfg, run)
    if "convergence" in wanted:
        payload["convergence"] = run_convergence(cfg, run, cases)
    if "sweep" in wanted:
        payload["sweep"] = run_sweep(cfg, run)
    if "fd" in wanted:
        payload["fd_discrepancy"], payload["fd_detail"] = run_fd(cfg, run)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=list(wanted),
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"], "cases": cases},
        merge=args.only is not None,
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick}, sections={list(wanted)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
