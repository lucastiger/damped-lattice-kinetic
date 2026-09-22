#!/usr/bin/env python3
"""Experiment 05 -- the two null vectors: decay, localization, and the simplicity of ker M0.

``phi'`` spans ``ker M0`` and ``phat`` spans ``ker M0^T``.  Hypotheses H2-H3 of the note
ask for two things about them that the analysis cannot yet supply: that the kernel is
one-dimensional, and that ``phat`` decays exponentially (hence lies in ``L^1``, which is
what makes the pairings ``<phat, .>`` defined on the infinite lattice).  This experiment
measures both, and rules out the shortcut that would make the second one free.

(a) ``decay`` -- the two envelopes at ``|xi| in {0, 10, 20, 40, 80, 120, 199}``, each
    function first divided by its own max-norm and then read at the grid point nearest
    ``+/-|xi|``, taking the larger of the two sides.  **These numbers are not monotone in
    ``|xi|`` and are not meant to be**: the tails oscillate (the decay rate is the
    imaginary part of a root of ``L(k) = 0``, whose real part sets the oscillation), so a
    sample at one point can land near a zero of the envelope.  ``5.3e-3`` at ``|xi| = 20``
    against ``1.2e-2`` at ``|xi| = 40`` is that, not an error.  What matters is that both
    columns fall by nine orders over the domain and fall *together*.

(b) ``localization`` -- the fraction of discrete ``L^2`` mass inside ``|xi| < 20``.  Both
    null vectors put 99.0% of themselves there.

(c) ``sv`` -- the three smallest singular values of ``M0``, by full SVD.  They come out at
    ``2.7e-9``, ``9.9e-2``, ``1.9e-1``: one at the level of the discretization error and the
    next seven orders of magnitude above it.  THAT IS THE EVIDENCE THAT ``ker M0`` IS
    ONE-DIMENSIONAL (Hypothesis H2) at this velocity -- a second null direction would have
    to hide in a gap that is not there.  It is evidence, not a proof: it is a statement
    about the discretized operator on a finite domain, and H2 concerns the infinite lattice.

(d) ``res_phat_rel`` and the profiles, written to CSV for the figure.

(e) ``weight_test`` -- the shortcut that does not work.  ``phat`` solves the *anti-damped*
    equation, so one might hope it is just ``phi'`` in an exponential weight,
    ``phat = C phi' e^{-gamma xi / c}``, which would make its decay follow from ``phi'``'s
    and hand us Hypothesis H3 for free.  Two diagnostics, and they are not equally useful.

    The **pointwise** ratio ``|phat| / (|phi'| e^{-gamma xi/c})`` is the literal test: under
    the weight it would be constant, span zero.  Measured over ``|xi| < 150`` it spans some
    29 decades, so the weight is refuted -- but the size of that span means nothing.  Both
    functions oscillate and vanish at isolated points, so the ratio blows up near every zero
    of ``phi'`` and collapses near every zero of ``phat``; the ``p0_floor`` cut removes only
    the worst of it, and the 5th-95th percentile span is still 27 decades.

    The **envelope** ratio is the interpretable one, and it is the decay table of (a) made
    continuous.  Over each window in ``|xi|`` it takes the largest ``|phat|`` on *either*
    side over the largest ``|phi'|`` on either side.  That symmetrization is the whole
    point: the two null vectors do not decay at the same rate on the same side -- ``phi'``
    solves the damped equation and ``phat`` the anti-damped one, so each is the other's
    mirror -- and a one-sided comparison measures that asymmetry rather than the weight.
    Symmetrized, the two envelopes track each other, staying within about half a decade of
    1 across the domain; that is the ``0.8``-``2.2`` of the decay table's last column.
    Under the weight hypothesis the same quantity would have to *grow* like
    ``e^{gamma |xi| / c}``, by ``7.3`` decades out to ``|xi| = 150`` (and by ``e^{22}`` out
    to the edge of the domain, which is the figure the note quotes).  It does not.  So the
    weight is absent, and H3 does not follow from the decay of ``phi'``.

Everything numerical is in ``configs/05_null_vectors.yaml``.

Usage
-----
    python scripts/05_null_vectors.py --config configs/05_null_vectors.yaml
    python scripts/05_null_vectors.py --quick        # reduced L/N, ~10 s, not claim-grade

Production runtime: about two minutes, most of it the 2048 x 2048 SVD.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_parser, crange, load, march, output_path, print_table, section

from dlkin import Grid, branch_scalars, init_branch, save_result

SCRIPT = "scripts/05_null_vectors.py"
PROFILE_COLUMNS = ("xi", "phi_prime", "phat")


def _land(cfg, run, L: float, N: int):
    """Land on the branch at ``(L, N)`` and march to the working velocity."""
    newton = section(run, "newton")
    phat = section(run, "phat")
    approach = dict(run["approach"])
    tol, maxit = float(newton["tol"]), int(newton["maxit"])
    grid = Grid(L=float(L), N=int(N))
    solver, psi, drive = init_branch(grid, cfg.model, c0=float(run["c0"]), tol=tol, maxit=maxit)
    psi, drive, _ = march(
        solver,
        psi,
        drive,
        crange(float(approach["start"]), float(approach["stop"]), float(approach["step"])),
        tol,
        maxit,
    )
    scalars = branch_scalars(
        solver,
        psi,
        drive,
        float(run["c"]),
        normalization=str(phat["normalization"]),
        phat_seed=int(phat["seed"]),
        phat_iters=int(phat["iters"]),
        phat_shift=float(phat["shift"]),
    )
    return grid, solver, psi, drive, scalars


def _unit_envelopes(scalars) -> Tuple[np.ndarray, np.ndarray]:
    """``phi'`` and ``phat``, each divided by its own max-norm."""
    p0 = scalars.phi_prime / np.max(np.abs(scalars.phi_prime))
    ph = scalars.phat / np.max(np.abs(scalars.phat))
    return p0, ph


def run_decay(cfg, run) -> Tuple[Dict[str, Any], Dict[str, Any], Any, Grid]:
    """(a)+(d) the decay table, res_phat_rel, and the profile CSV."""
    opts = section(run, "decay")
    grid, _, _, _, scalars = _land(cfg, run, opts["L"], opts["N"])
    p0, ph = _unit_envelopes(scalars)

    decay: Dict[str, Any] = {}
    table = []
    for offset in [float(x) for x in opts["offsets"]]:
        j_plus = int(np.argmin(np.abs(grid.xi - offset)))
        j_minus = int(np.argmin(np.abs(grid.xi + offset)))
        row = {
            "p0": float(max(abs(p0[j_plus]), abs(p0[j_minus]))),
            "phat": float(max(abs(ph[j_plus]), abs(ph[j_minus]))),
            "xi_plus": float(grid.xi[j_plus]),
            "xi_minus": float(grid.xi[j_minus]),
        }
        decay[f"{offset:g}"] = row
        table.append([offset, row["xi_plus"], row["p0"], row["phat"], row["p0"] / row["phat"]])

    csv_path = Path(opts["csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PROFILE_COLUMNS)
        for xi, a, b in zip(grid.xi, p0, ph):
            writer.writerow([xi, a, b])

    print_table(
        f"(a) decay envelopes  (c = {run['c']}, L = {grid.L:g}, N = {grid.N}; each function "
        "normalized by its own max -- NOT monotone, the tails oscillate)",
        ["|xi|", "nearest grid xi", "phi'", "phat", "ratio"],
        table,
        ["g", ".4f", ".4e", ".4e", ".3f"],
    )
    detail = {
        "L": grid.L,
        "N": grid.N,
        "c": float(run["c"]),
        "normalization": scalars.normalization,
        "csv": str(csv_path),
        "columns": list(PROFILE_COLUMNS),
        "res_phat_rel": scalars.res_phat_rel,
        "res_M0p0": scalars.res_M0_phi_prime,
    }
    return decay, detail, scalars, grid


def run_localization(cfg, run) -> Dict[str, Any]:
    """(b) fraction of discrete L2 mass inside |xi| < radius."""
    opts = section(run, "localization")
    grid, _, _, _, scalars = _land(cfg, run, opts["L"], opts["N"])
    radius = float(opts["radius"])
    inside = np.abs(grid.xi) < radius

    record: Dict[str, Any] = {"L": grid.L, "N": grid.N, "radius": radius}
    for name, f in (("p0", scalars.phi_prime), ("phat", scalars.phat)):
        mass = f**2
        record[name] = float(np.sum(mass[inside]) / np.sum(mass))
    print_table(
        f"(b) localization: share of discrete L2 mass with |xi| < {radius:g}  "
        f"(L = {grid.L:g}, N = {grid.N})",
        ["vector", "fraction"],
        [["phi'", record["p0"]], ["phat", record["phat"]]],
        ["", ".6f"],
    )
    return record


def run_sv(cfg, run) -> Dict[str, Any]:
    """(c) the smallest singular values of M0: the evidence that ker M0 is one-dimensional."""
    opts = section(run, "sv")
    grid, _, _, _, scalars = _land(cfg, run, opts["L"], opts["N"])
    started = time.perf_counter()
    values = np.sort(np.linalg.svd(scalars.M0, compute_uv=False))
    elapsed = time.perf_counter() - started
    n = int(opts["n_smallest"])

    record: Dict[str, Any] = {
        "L": grid.L,
        "N": grid.N,
        "c": float(run["c"]),
        "elapsed_sec": round(elapsed, 2),
        "gap_ratio": float(values[1] / values[0]),
    }
    for i in range(n):
        record[f"s{i + 1}"] = float(values[i])
    print_table(
        f"(c) smallest singular values of M0  (L = {grid.L:g}, N = {grid.N}, "
        f"c = {run['c']}; {elapsed:.1f} s)",
        ["index", "singular value"],
        [[f"s{i + 1}", record[f"s{i + 1}"]] for i in range(n)],
        ["", ".6e"],
    )
    print(
        f"    s2/s1 = {record['gap_ratio']:.3e}: one singular value at the discretization "
        "level and the next seven orders above it -- ker M0 is one-dimensional here."
    )
    return record


def run_weight_test(cfg, run, scalars, grid) -> Dict[str, Any]:
    """(e) Is phat an exponentially weighted copy of phi'?  No -- see the module docstring."""
    opts = section(run, "weight_test")
    c, gamma = float(run["c"]), cfg.model.gamma
    xi_max, floor = float(opts["xi_max"]), float(opts["p0_floor"])
    window = float(opts["envelope_window"])
    pct = float(opts["percentile"])
    p0, ph = _unit_envelopes(scalars)

    # -- the literal pointwise test -------------------------------------------------
    keep = (np.abs(grid.xi) < xi_max) & (np.abs(p0) > floor)
    xi = grid.xi[keep]
    ratio = np.abs(ph[keep]) / (np.abs(p0[keep]) * np.exp(-gamma * xi / c))
    lo, hi = np.percentile(ratio, [pct, 100.0 - pct])

    # -- the envelope test, symmetrized in |xi| exactly as the decay table in (a) is:
    #    the largest |phat| on either side over the largest |phi'| on either side.  phi'
    #    and phat are mirror images in their decay rates, so a one-sided ratio measures
    #    that asymmetry rather than the weight; only the symmetrized one is comparable.
    abs_xi = np.abs(grid.xi)
    edges = np.arange(0.0, xi_max + window, window)
    centres, env = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (abs_xi >= a) & (abs_xi < b)
        if not np.any(sel):
            continue
        top_p0 = float(np.max(np.abs(p0[sel])))
        if top_p0 <= floor:
            continue
        centres.append(0.5 * (a + b))
        env.append(float(np.max(np.abs(ph[sel]))) / top_p0)
    env = np.asarray(env)

    # what each hypothesis predicts, in decades, across the retained range
    pointwise_span_if_no_weight = gamma * (float(xi.max()) - float(xi.min())) / c / np.log(10.0)
    envelope_span_if_weight = gamma * xi_max / c / np.log(10.0)

    record = {
        "c": c,
        "xi_max": xi_max,
        "p0_floor": floor,
        "n_points": int(keep.sum()),
        "ratio_max": float(ratio.max()),
        "ratio_min": float(ratio.min()),
        "ratio_max_over_min": float(ratio.max() / ratio.min()),
        "log10_span": float(np.log10(ratio.max() / ratio.min())),
        "percentile": pct,
        f"ratio_p{int(100 - pct)}": float(hi),
        f"ratio_p{int(pct)}": float(lo),
        "log10_span_percentile": float(np.log10(hi / lo)),
        "log10_span_if_no_weight": float(pointwise_span_if_no_weight),
        "log10_span_if_weight_exact": 0.0,
        "envelope_symmetrized": True,
        "envelope_window": window,
        "envelope_n_windows": int(env.size),
        "envelope_ratio_max": float(env.max()),
        "envelope_ratio_min": float(env.min()),
        "envelope_log10_span": float(np.log10(env.max() / env.min())),
        "envelope_log10_span_if_weight": float(envelope_span_if_weight),
        "envelope_centres": [float(x) for x in centres],
        "envelope_ratio": [float(x) for x in env],
    }
    print_table(
        f"(e) is phat = phi' x exp(-gamma xi/c)?   |xi| < {xi_max:g}, c = {c}",
        ["quantity", "value"],
        [
            ["pointwise ratio: max", record["ratio_max"]],
            ["pointwise ratio: min", record["ratio_min"]],
            ["pointwise log10 span", record["log10_span"]],
            [f"pointwise log10 span, p{int(pct)}-p{int(100 - pct)}", record["log10_span_percentile"]],
            ["  -- span if the weight were exact", 0.0],
            [f"ENVELOPE ratio (symmetrized in |xi|), {window:g}-wide windows: max", record["envelope_ratio_max"]],
            ["ENVELOPE ratio: min", record["envelope_ratio_min"]],
            ["ENVELOPE log10 span", record["envelope_log10_span"]],
            ["  -- growth the weight would force", record["envelope_log10_span_if_weight"]],
        ],
        ["", ".4g"],
    )
    print(
        f"    Pointwise: span {record['log10_span']:.1f} decades against 0 if the weight "
        "were exact, so the weight is\n    refuted -- but that span is dominated by the two "
        "functions' isolated zeros and its size is\n    not interpretable.  ENVELOPE "
        f"(symmetrized): span {record['envelope_log10_span']:.2f} decades, against "
        f"{record['envelope_log10_span_if_weight']:.1f} decades of\n    GROWTH the weight "
        f"would force out to |xi| = {xi_max:g}.  The two envelopes track each other "
        "instead:\n    phat is NOT phi' in the weight exp(-gamma xi/c), so H3 does not "
        "follow from the decay of phi'."
    )
    return record


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/05_null_vectors.yaml", "data/05_nullvec.json")
    args = parser.parse_args(argv)
    raw, cfg = load(args)
    run = cfg.run

    started = time.perf_counter()
    payload: Dict[str, Any] = {}
    decay, detail, scalars, grid = run_decay(cfg, run)
    payload["decay"] = decay
    payload["decay_detail"] = detail
    payload["res_phat_rel"] = detail["res_phat_rel"]
    payload["localization"] = run_localization(cfg, run)
    payload["sv"] = run_sv(cfg, run)
    payload["weight_test"] = run_weight_test(cfg, run, scalars, grid)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out,
        payload,
        script=SCRIPT,
        quick=args.quick,
        config_path=args.config,
        config=raw,
        sections=["decay", "localization", "sv", "weight_test"],
        elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]},
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
