#!/usr/bin/env python3
"""Experiment 05 -- the null vectors ``phi'`` and ``phat``, and the evidence for H2 and H3.

Everything the identity rests on is a pairing against ``phat``, the null vector of the
*anti-damped* operator ``M0^T`` (R2 of ``handoff/SCIENCE_BRIEF.md``).  Lemma 4.1 pairs it
against functions that are merely bounded -- ``d_c phi``, which tends to
``sigma'/sqrt(1-sigma^2)`` at both ends, and the constant ``1`` -- so the pairing is ``L^1``
against ``L^infinity`` and needs ``phat`` to be integrable.  That is Hypothesis H3, and it is
assumed in the note, not proved.  Hypothesis H2, that ``ker M0`` is one-dimensional, is also
assumed.  This experiment is the numerical support for both.

Sections (``--only``, comma-separated):

``decay``
    The envelope of ``phi'`` and ``phat`` at ``c = 0.89``.  Each vector is divided by its own
    max-norm and read at the grid point nearest ``+/-|xi|``, taking the larger of the two
    sides.  The tails oscillate -- the characteristic roots of ``L(k) = 0`` are complex -- so
    these numbers are NOT monotone in ``|xi|``, and the dip at ``|xi| = 20`` sitting below the
    value at ``|xi| = 40`` is a zero of the oscillation, not a convergence failure.  Both
    vectors fall by nine orders of magnitude across the domain.

``localization``
    Fraction of the discrete ``L2`` mass inside ``|xi| < 20``: about 0.99 for both.

``singular_values``
    The three smallest singular values of ``M0``.  A gap of seven orders of magnitude between
    the first and the second is what "``ker M0`` is one-dimensional" means numerically.

``weight_test``
    Remark 3.5(2) of the note.  The substitution ``chi = e^{-gamma xi / c} psi`` intertwines
    the *differential* parts of ``M0`` and ``M0^T`` exactly but maps the shifts to
    ``e^{-/+ gamma/c} psi(xi +/- 1)``, so it does not intertwine the two operators and the
    weighted-space route of [VCKX20] does not close that way.  If it did -- if ``phat`` were
    ``e^{-gamma xi/c}`` times ``phi'`` -- then ``|phat| / (|phi'| e^{-gamma xi/c})`` would be
    constant across the domain.  Over ``|xi| < 150`` at ``gamma/c = 0.112`` a genuine
    exponential tilt would span ``e^{2 * 0.112 * 150} ~ e^{34}``.  What is recorded is the
    span actually observed.  Both functions oscillate and vanish at isolated points, so this
    is an envelope diagnostic: the ratio is formed only where ``|phi'|`` exceeds ``floor``.

Usage
    python scripts/05_null_vectors.py [--config ...] [--quick] [--only decay,...]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, build_parser, crange, load, march, output_path, print_table, section

from dlkin import Grid, init_branch, save_result
from dlkin.spectral import branch_scalars, left_null, normalize_phat

SCRIPT = "scripts/05_null_vectors.py"
SECTIONS = ("decay", "localization", "singular_values", "weight_test")


def _reach(cfg, run, sec) -> tuple:
    """Converge the branch at the section's (L, N) and walk to its velocity."""
    grid = Grid(L=float(sec.get("L", cfg.grid.L)), N=int(sec.get("N", cfg.grid.N)))
    newton = section(run, "newton")
    solver, psi, drive = init_branch(
        grid, cfg.model, c0=float(sec.get("c0", 0.88)), tol=newton["tol"], maxit=newton["maxit"]
    )
    approach = sec.get("approach")
    target = float(sec["c"])
    values = (
        crange(approach["start"], approach["stop"], approach["step"])
        if approach
        else [round(x, 7) for x in np.arange(0.881, target + 1e-12, 0.001)]
    )
    psi, drive, _ = march(solver, psi, drive, values, newton["tol"], newton["maxit"])
    if abs(values[-1] - target) > 1e-12:
        psi, drive, _ = march(solver, psi, drive, [target], newton["tol"], newton["maxit"])
    return solver, psi, drive, grid


def run_decay(cfg, run) -> Dict[str, Any]:
    sec = section(run, "decay")
    phat_cfg = section(run, "phat")
    solver, psi, drive, grid = _reach(cfg, run, sec)
    c = float(sec["c"])
    sc = branch_scalars(
        solver, psi, drive, c,
        normalization=phat_cfg["normalization"], phat_seed=phat_cfg["seed"],
        phat_iters=phat_cfg["iters"], phat_shift=phat_cfg["shift"],
    )
    p0 = sc.phi_prime / np.max(np.abs(sc.phi_prime))
    ph = sc.phat / np.max(np.abs(sc.phat))

    decay: Dict[str, Dict[str, float]] = {}
    for station in sec["stations"]:
        x = float(station)
        j = int(np.argmin(np.abs(grid.xi - x)))
        jm = int(np.argmin(np.abs(grid.xi + x)))
        key = f"{int(station)}" if float(station).is_integer() else f"{station}"
        decay[key] = {
            "p0": float(max(abs(p0[j]), abs(p0[jm]))),
            "phat": float(max(abs(ph[j]), abs(ph[jm]))),
        }

    csv_path = REPO_ROOT / sec["profile_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["xi", "phi_prime", "phat"])
        for x, a, b in zip(grid.xi, p0, ph):
            writer.writerow([f"{x:.10g}", f"{a:.10e}", f"{b:.10e}"])

    print_table(
        f"decay envelope at c = {c} (each vector max-normalized)",
        ["|xi|", "phi'", "phat"],
        [[k, v["p0"], v["phat"]] for k, v in decay.items()],
        ["", ".3e", ".3e"],
    )
    return {
        **decay,
        "c": c,
        "L": grid.L,
        "N": grid.N,
        "res_phat_rel": sc.res_phat_rel,
        "res_M0_phi_prime": sc.res_M0_phi_prime,
        "profile_csv": sec["profile_csv"],
    }


def run_localization(cfg, run) -> Dict[str, Any]:
    sec = section(run, "localization")
    phat_cfg = section(run, "phat")
    solver, psi, drive, grid = _reach(cfg, run, {**sec, "approach": None})
    sc = branch_scalars(
        solver, psi, drive, float(sec["c"]),
        normalization=phat_cfg["normalization"], phat_seed=phat_cfg["seed"],
        phat_iters=phat_cfg["iters"], phat_shift=phat_cfg["shift"],
    )
    radius = float(sec["radius"])
    inside = np.abs(grid.xi) < radius
    frac = lambda v: float(np.sum(v[inside] ** 2) / np.sum(v ** 2))  # noqa: E731
    out = {
        "p0": frac(sc.phi_prime),
        "phat": frac(sc.phat),
        "radius": radius,
        "L": grid.L,
        "N": grid.N,
        "c": float(sec["c"]),
    }
    print_table(
        f"L2 mass inside |xi| < {radius} (L={grid.L}, N={grid.N})",
        ["phi'", "phat"], [[out["p0"], out["phat"]]], [".6f", ".6f"],
    )
    return out


def run_singular_values(cfg, run) -> Dict[str, Any]:
    sec = section(run, "singular_values")
    solver, psi, drive, grid = _reach(cfg, run, {**sec, "approach": None})
    solver.build(float(sec["c"]))
    sv = np.linalg.svd(solver.M0(psi), compute_uv=False)
    sv = np.sort(sv)[: int(sec["count"])]
    out = {f"s{i + 1}": float(v) for i, v in enumerate(sv)}
    out.update({"L": grid.L, "N": grid.N, "c": float(sec["c"]), "gap_s2_over_s1": float(sv[1] / sv[0])})
    print_table(
        f"smallest singular values of M0 (L={grid.L}, N={grid.N}, c={sec['c']})",
        [f"s{i + 1}" for i in range(len(sv))] + ["s2/s1"],
        [[*[float(v) for v in sv], out["gap_s2_over_s1"]]],
        [".4e"] * len(sv) + [".2e"],
    )
    return out


def run_weight_test(cfg, run, decay_cache) -> Dict[str, Any]:
    """Does ``phat`` carry the exponential tilt ``e^{-gamma xi/c}`` relative to ``phi'``?

    Pointwise ratios are useless here: both vectors oscillate and vanish at isolated points,
    and their zeros are at different places, so ``|phat| / |phi'|`` spikes.  The comparison is
    therefore between *envelopes* -- a running maximum over one tail wavelength -- and the
    quantity reported is the least-squares slope of ``log(env(phat) / env(phi'))`` against
    ``xi``.  If ``phat = e^{-gamma xi/c} phi'`` that slope is exactly ``-gamma/c``; if the
    weight plays no role it is zero.
    """
    sec = section(run, "weight_test")
    solver, psi, drive, grid, sc, c = decay_cache
    window = float(sec["window"])
    floor = float(sec["floor"])
    gamma = cfg.model.gamma

    p0 = np.abs(sc.phi_prime / np.max(np.abs(sc.phi_prime)))
    ph = np.abs(sc.phat / np.max(np.abs(sc.phat)))
    # running maximum over one tail wavelength (2 pi / Re k ~ 2.7 in xi units)
    half = max(1, int(round(float(sec.get("envelope_width", 3.0)) / (2.0 * grid.L / grid.N) / 2)))
    kernel = 2 * half + 1
    def envelope(v: np.ndarray) -> np.ndarray:
        padded = np.concatenate([v[-half:], v, v[:half]])
        return np.max(np.lib.stride_tricks.sliding_window_view(padded, kernel), axis=-1)

    e0, eh = envelope(p0), envelope(ph)
    mask = (np.abs(grid.xi) < window) & (e0 > floor) & (eh > floor)
    x = grid.xi[mask]
    ratio = eh[mask] / e0[mask]

    # What the weight would require: log(env phat / env phi') linear in xi with slope -gamma/c.
    slope, intercept = np.polyfit(x, np.log(ratio), 1)
    residual = float(np.std(np.log(ratio) - (slope * x + intercept)))

    # What is actually there: phat's envelope at +xi matches phi''s at -xi.  Reversing the sign
    # of gamma reverses which side of the kink radiates, so the adjoint's slowly-decaying tail
    # is on the opposite side -- a reflection, not a tilt.  Compared at matched stations
    # because the two oscillate with different phase, so only the envelopes can be compared.
    reflection = {}
    for station in sec.get("reflection_stations", [50, 100, 150, 190]):
        xs = float(station)
        if xs >= grid.L:
            continue
        j = int(np.argmin(np.abs(grid.xi - xs)))
        jm = int(np.argmin(np.abs(grid.xi + xs)))
        half_w = max(1, int(round(1.5 / (2.0 * grid.L / grid.N))))
        a = float(np.max(p0[max(0, jm - half_w): jm + half_w]))
        b = float(np.max(ph[max(0, j - half_w): j + half_w]))
        reflection[str(int(station))] = {
            "phi_prime_at_minus": a, "phat_at_plus": b, "ratio": float(b / a) if a > 0 else None,
        }
    ratios = [v["ratio"] for v in reflection.values() if v["ratio"] is not None]

    out = {
        "observed_slope": float(slope),
        "slope_if_weight_held": float(-gamma / c),
        "slope_sign_matches_weight": bool(slope * (-gamma / c) > 0),
        "slope_fit_residual_std": residual,
        "reflection": reflection,
        "reflection_ratio_min": float(min(ratios)) if ratios else None,
        "reflection_ratio_max": float(max(ratios)) if ratios else None,
        "window": window,
        "envelope_width_xi": float(sec.get("envelope_width", 3.0)),
        "gamma_over_c": float(gamma / c),
        "points_used": int(mask.sum()),
    }
    print_table(
        "the weighted-space hypothesis: slope of log(env phat / env phi') against xi",
        ["observed slope", "slope if the weight held", "same sign?", "fit residual"],
        [[out["observed_slope"], out["slope_if_weight_held"],
          out["slope_sign_matches_weight"], residual]],
        ["+.5f", "+.5f", "", ".2f"],
    )
    print_table(
        "what is there instead: env|phat|(+xi) against env|phi'|(-xi)",
        ["|xi|", "env |phi'|(-xi)", "env |phat|(+xi)", "ratio"],
        [[k, v["phi_prime_at_minus"], v["phat_at_plus"], v["ratio"]] for k, v in reflection.items()],
        ["", ".4e", ".4e", ".5f"],
    )
    return out


def main(argv: List[str] | None = None) -> int:
    parser = build_parser(__doc__ or "", "configs/05_null_vectors.yaml", "data/05_nullvec.json")
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
    decay_cache = None
    if "decay" in wanted or "weight_test" in wanted:
        sec = section(run, "decay")
        phat_cfg = section(run, "phat")
        solver, psi, drive, grid = _reach(cfg, run, sec)
        sc = branch_scalars(
            solver, psi, drive, float(sec["c"]),
            normalization=phat_cfg["normalization"], phat_seed=phat_cfg["seed"],
            phat_iters=phat_cfg["iters"], phat_shift=phat_cfg["shift"],
        )
        decay_cache = (solver, psi, drive, grid, sc, float(sec["c"]))
    if "decay" in wanted:
        payload["decay"] = run_decay(cfg, run)
    if "localization" in wanted:
        payload["localization"] = run_localization(cfg, run)
    if "singular_values" in wanted:
        payload["sv"] = run_singular_values(cfg, run)
    if "weight_test" in wanted:
        payload["weight_test"] = run_weight_test(cfg, run, decay_cache)
    elapsed = time.perf_counter() - started

    out = output_path(args)
    save_result(
        out, payload, script=SCRIPT, quick=args.quick, config_path=args.config, config=raw,
        sections=list(wanted), elapsed_sec=elapsed,
        extra={"normalization": run["phat"]["normalization"]}, merge=args.only is not None,
    )
    print(f"\nwrote {out}  ({elapsed:.1f} s, quick={args.quick}, sections={list(wanted)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
