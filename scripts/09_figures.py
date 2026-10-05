#!/usr/bin/env python3
"""Generate the six figures of the note from ``data/``, and nothing else.

Every figure is built from the result files the experiment scripts wrote.  Nothing here
solves, continues or diagonalizes anything: if a figure needs a quantity, that quantity must
first be recorded by the experiment that owns it.  That keeps the figures downstream of the
validated data rather than beside it, so a number in a figure cannot disagree with the same
number in a table.

Filenames are contractual -- ``note.tex`` calls ``\\figinclude{fig1_kinetic_relation.pdf}`` and
friends, and falls back to a placeholder box when a file is absent.

Usage
    python scripts/09_figures.py [--only fig1,fig3] [--outdir figures] [--quick-data]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

# colour-blind-safe; every series is distinguished by marker/linestyle as well as colour
BLUE, ORANGE, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"

RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,
}
WIDTH = 6.5


def load(name: str) -> Dict[str, Any]:
    path = DATA / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run the experiment that produces it (see README) before "
            "generating figures; this script never computes physics itself."
        )
    return json.loads(path.read_text())


def load_csv(name: str) -> Dict[str, np.ndarray]:
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the experiment that writes it first.")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "preview").mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    # fixed metadata: byte-identical across runs in one environment (not across matplotlib versions)
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(outdir / "preview" / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(REPO_ROOT)}")
    return pdf


# --------------------------------------------------------------------- 1 ----
def fig1(outdir: Path) -> None:
    """The kinetic relation, with the two kinds of critical point marked."""
    curve = load_csv("01_kinetic_curve.csv")
    kin = load("01_kinetic.json")
    fold = load("04_fold.json")
    c_hat1 = kin["critical"]["c_hat1"]
    s_hat1 = kin["critical"]["sigma_hat1"]
    c_max = fold["c_max"]

    # in STEP order: the branch is multivalued in c, so sorting by c would interleave the two
    # branches and draw a zigzag across them
    steps = sorted(fold["rows"], key=lambda k: int(k))
    rows = [fold["rows"][k] for k in steps]
    fc = np.array([r["c"] for r in rows])
    fs = np.array([r["sigma"] for r in rows])
    unstable = np.array([r.get("n_unstable") for r in rows], dtype=object)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), constrained_layout=True)
    ax1.plot(curve["c"], curve["sigma"], color=BLUE, lw=1.2)
    ax1.plot([c_hat1], [s_hat1], "o", color=ORANGE, ms=5, zorder=5)
    ax1.set_xlabel(r"$c$")
    ax1.set_ylabel(r"$\sigma$")
    ax1.set_title(r"first branch", fontsize=9)
    ax1.grid(alpha=0.25, lw=0.5)

    zoom = (curve["c"] > 0.8975)
    ax2.plot(curve["c"][zoom], curve["sigma"][zoom], color=BLUE, lw=1.2, label="natural continuation")
    ax2.plot(fc, fs, "-", color=GREEN, lw=1.0, label="pseudo-arclength")
    ax2.plot([c_hat1], [s_hat1], "o", color=ORANGE, ms=5, zorder=5,
             label=r"$\hat c_1$: $\sigma'=0$, $\kappa=0$")
    j = int(np.argmin(np.abs(fc - c_max)))
    ax2.plot([c_max], [fs[j]], "s", color=PURPLE, ms=5, zorder=5,
             label=r"$c_{\max}$: $\dot c=0$, $\langle\hat p,1\rangle=0$")
    ax2.set_xlabel(r"$c$")
    ax2.set_ylabel(r"$\sigma$")
    ax2.set_title("the two kinds of critical point", fontsize=9)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(loc="lower left", frameon=False, fontsize=7)

    ax2.annotate("stable", xy=(0.8982, s_hat1), xytext=(0, -12), textcoords="offset points",
                 fontsize=7, color=GREY)
    ax2.annotate("1 unstable", xy=(c_max, fs[j]), xytext=(-46, -4), textcoords="offset points",
                 fontsize=7, color=GREY)
    save(fig, outdir, "fig1_kinetic_relation")


# --------------------------------------------------------------------- 2 ----
def fig2(outdir: Path) -> None:
    """The identity itself, and the leading-order eigenvalue near the threshold."""
    curve = load_csv("01_kinetic_curve.csv")
    thr = load("03_threshold.json")
    kin = load("01_kinetic.json")
    c_hat1 = kin["critical"]["c_hat1"]

    two_pi_sigma = 2.0 * np.pi * curve["sigma_prime"]
    rel = np.abs(curve["kappa"] - two_pi_sigma) / np.maximum(np.abs(two_pi_sigma), 1e-300)

    rows = sorted(thr["rows"].values(), key=lambda r: r["c"])
    tc = np.array([r["c"] for r in rows])
    pred = np.array([r["nu2_pred"] for r in rows])
    nu2 = np.array([np.nan if r["nu2"] is None else r["nu2"] for r in rows])
    valid = np.array([r["m"] > 0 for r in rows])

    fig = plt.figure(figsize=(WIDTH, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2 = fig.add_subplot(gs[:, 1])

    ax.plot(curve["c"], curve["kappa"], color=BLUE, lw=1.6, label=r"$\kappa(c)=\langle\hat p,M_1\phi'\rangle$")
    ax.plot(curve["c"], two_pi_sigma, "--", color=ORANGE, lw=1.0, label=r"$2\pi\mu\,\sigma'(c)$")
    ax.set_ylabel(r"$\kappa$")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelbottom=False)

    axr.semilogy(curve["c"], np.maximum(rel, 1e-17), color=GREY, lw=0.9)
    axr.set_xlabel(r"$c$")
    axr.set_ylabel("rel. diff.")
    axr.grid(alpha=0.25, lw=0.5)

    ax2.axhline(0.0, color=GREY, lw=0.6)
    ax2.axvline(c_hat1, color=ORANGE, lw=0.8, ls=":", label=r"$\hat c_1$")
    ax2.plot(tc[valid], pred[valid], "s--", color=ORANGE, ms=4, lw=0.9,
             label=r"$-\kappa/m$ (leading order)")
    ax2.plot(tc, nu2, "o-", color=BLUE, ms=4, lw=1.0, label=r"$\nu_2$ (Arnoldi)")
    ax2.set_xlabel(r"$c$")
    ax2.set_ylabel(r"$\nu_2$")
    ax2.set_title("the crossing at the first extremum", fontsize=9)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(loc="upper left", frameon=False, fontsize=7.5)
    save(fig, outdir, "fig2_jordan_identity")


# --------------------------------------------------------------------- 3 ----
def fig3(outdir: Path) -> None:
    """The fold: cdot and <phat,1> vanish together while kappa and nu do not."""
    fold = load("04_fold.json")
    rows = sorted(fold["rows"].values(), key=lambda r: int(r.get("step", 0)) if "step" in r else r["c"])
    keys = sorted(fold["rows"], key=lambda k: int(k))
    rows = [fold["rows"][k] for k in keys]
    step = np.array([int(k) for k in keys], dtype=float)
    cdot = np.array([r["cdot"] for r in rows])
    one = np.array([r["phat_one"] for r in rows])
    kappa = np.array([r["kappa"] for r in rows])
    nu = np.array([np.nan if r.get("nu") is None else r["nu"] for r in rows])
    fold_step = fold.get("fold_step")

    # phat is defined up to sign, and the stored orientation (<phat, phi'> > 0) flips once along
    # this stretch, taking kappa and <phat,1> with it.  That flip is a convention, not a zero:
    # kappa is ~2.5 in magnitude on both sides.  Restore continuity by carrying one orientation
    # through -- the arclength invariant cdot*kappa + mu*sigmadot*<phat,1> is homogeneous in
    # phat, so this changes no statement the figure makes.
    sign = np.ones_like(kappa)
    for i in range(1, kappa.size):
        flipped = (np.sign(kappa[i]) != np.sign(kappa[i - 1])
                   and min(abs(kappa[i]), abs(kappa[i - 1])) > 0.5)
        sign[i] = -sign[i - 1] if flipped else sign[i - 1]
    kappa = kappa * sign
    one = one * sign

    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 4.6), sharex=True, constrained_layout=True)
    for ax, y, label, colour in (
        (axes[0], cdot, r"$\dot c$", BLUE),
        (axes[1], one, r"$\langle\hat p,\mathbf{1}\rangle$", ORANGE),
        (axes[2], kappa, r"$\kappa$", GREEN),
        (axes[3], nu, r"$\nu$ (unstable)", PURPLE),
    ):
        ax.plot(step, y, "-", color=colour, lw=1.2)
        ax.axhline(0.0, color=GREY, lw=0.6)
        if fold_step is not None:
            ax.axvline(float(fold_step), color=GREY, lw=0.8, ls=":")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.5)
    axes[3].set_xlabel("pseudo-arclength step")
    axes[0].set_title(
        r"through the turning point: $\dot c$ and $\langle\hat p,\mathbf{1}\rangle$ "
        r"cross zero together, $\kappa$ and $\nu$ do not", fontsize=8.5
    )
    save(fig, outdir, "fig3_fold_dichotomy")


# --------------------------------------------------------------------- 4 ----
def fig4(outdir: Path) -> None:
    """The null vectors: their tails mirror each other, and no exponential weight relates them."""
    prof = load_csv("05_null_profiles.csv")
    nv = load("05_nullvec.json")
    c = nv["decay"]["c"]
    gamma = nv["metadata"]["config_contents"]["model"]["gamma"]
    xi = prof["xi"]
    p0 = np.abs(prof["phi_prime"])
    ph = np.abs(prof["phat"])

    # running maximum over ~3 xi units: the tails oscillate, only their envelopes compare
    h = xi[1] - xi[0]
    half = max(1, int(round(1.5 / h)))
    def env(v):
        padded = np.concatenate([v[-half:], v, v[:half]])
        return np.max(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=-1)
    e0, eh = env(p0), env(ph)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.7), constrained_layout=True)
    ax1.semilogy(xi, np.maximum(p0, 1e-18), color=BLUE, lw=0.6, label=r"$|\phi'|$")
    ax1.semilogy(xi, np.maximum(ph, 1e-18), color=ORANGE, lw=0.6, alpha=0.85, label=r"$|\hat p|$")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel("max-normalized")
    ax1.set_ylim(1e-14, 3.0)
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax1.set_title(rf"null vectors at $c={c}$", fontsize=9)

    right = xi >= 0
    xr = xi[right]
    mirrored = np.interp(-xr, xi, e0)
    ax2.semilogy(xr, mirrored, color=BLUE, lw=1.4, label=r"env $|\phi'|(-\xi)$")
    ax2.semilogy(xr, eh[right], "--", color=ORANGE, lw=1.2, label=r"env $|\hat p|(+\xi)$")
    weight = e0[right] * np.exp(-gamma * xr / c)
    weight *= eh[right][0] / max(weight[0], 1e-300)
    ax2.semilogy(xr, np.maximum(weight, 1e-30), ":", color=GREY, lw=1.2,
                 label=r"env $|\phi'|(+\xi)\,e^{-\gamma\xi/c}$ (weighted space)")
    ax2.set_xlabel(r"$\xi\geq0$")
    ax2.set_ylim(1e-14, 3.0)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(loc="lower left", frameon=False, fontsize=7)
    ax2.set_title("a reflection, not a tilt", fontsize=9)
    save(fig, outdir, "fig4_null_vectors")


# --------------------------------------------------------------------- 5 ----
def fig5(outdir: Path) -> None:
    """The spectrum in both planes, at a stable and an unstable velocity."""
    spec = load("06_spectrum.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), constrained_layout=True)
    gamma = spec["metadata"]["config_contents"]["model"]["gamma"]

    styles = {"c089": (BLUE, "o", "stable"), "c08995": (ORANGE, "^", "unstable")}
    for key, (colour, marker, tag) in styles.items():
        if key not in spec:
            continue
        block = spec[key]
        data = load_csv(Path(block["csv"]).name)
        c = block["c"]
        axes[0].plot(data["re_nu"], data["im_nu"], marker, color=colour, ms=3.5,
                     mfc="none", lw=0, label=rf"$c={c}$ ({tag})")
        axes[1].plot(data["abs_rho"] * np.cos(data["arg_rho"]),
                     data["abs_rho"] * np.sin(data["arg_rho"]), marker, color=colour, ms=3.5,
                     mfc="none", lw=0, label=rf"$c={c}$")

    axes[0].axvline(-gamma / 2.0, color=GREY, ls="--", lw=0.8)
    axes[0].annotate(r"$\mathrm{Re}\,\nu=-\gamma/2$", xy=(-gamma / 2.0, -0.06), xytext=(4, 0),
                     textcoords="offset points", fontsize=7, color=GREY)
    axes[0].axvline(0.0, color="k", lw=0.5)
    axes[0].set_xlabel(r"$\mathrm{Re}\,\nu$")
    axes[0].set_ylabel(r"$\mathrm{Im}\,\nu$")
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    axes[0].grid(alpha=0.25, lw=0.5)

    theta = np.linspace(0, 2 * np.pi, 400)
    for key, (colour, _, _) in styles.items():
        if key in spec:
            r = spec[key]["circle_radius"]
            axes[1].plot(r * np.cos(theta), r * np.sin(theta), "--", color=colour, lw=0.7, alpha=0.6)
    axes[1].plot(np.cos(theta), np.sin(theta), "-", color="k", lw=0.7, label=r"$|\rho|=1$")
    axes[1].set_xlim(0.86, 1.04)
    axes[1].set_ylim(-0.09, 0.09)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$\mathrm{Re}\,\rho$")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\rho$")
    axes[1].set_title(r"near $\rho=1$", fontsize=9)
    axes[1].grid(alpha=0.25, lw=0.5)
    save(fig, outdir, "fig5_spectrum")


# --------------------------------------------------------------------- 6 ----
def fig6(outdir: Path) -> None:
    """Spectral convergence of the identity's residual, and the kernel residual beside it."""
    ident = load("02_identity.json")
    conv = ident["convergence"]
    rows = []
    for name, rec in conv.items():
        L = rec.get("L") or float(name.split("_")[0][1:])
        N = rec.get("N") or float(name.split("_")[1][1:])
        rows.append((L, N, 2 * L / N, rec["relerr"], rec["res_M0p0"]))
    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.8), constrained_layout=True)
    for L, marker, colour in ((150.0, "s", GREEN), (200.0, "o", BLUE), (300.0, "^", ORANGE)):
        sel = [r for r in rows if abs(r[0] - L) < 1e-9]
        if not sel:
            continue
        h = [r[2] for r in sel]
        ax.loglog(h, [max(r[3], 1e-17) for r in sel], marker + "-", color=colour, ms=5,
                  label=rf"identity, $L={L:g}$")
        ax.loglog(h, [r[4] for r in sel], marker + ":", color=colour, ms=4, mfc="none",
                  alpha=0.7, label=rf"$\|M_0\phi'\|/\|\phi'\|$, $L={L:g}$")
    ax.set_xlabel(r"$h = 2L/N$")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_title("spectral convergence; floor at machine precision", fontsize=9)
    save(fig, outdir, "fig6_convergence")


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="comma-separated subset of " + ", ".join(FIGURES))
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args(argv)
    wanted = list(FIGURES) if args.only is None else [s.strip() for s in args.only.split(",")]
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        parser.error(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")

    outdir = REPO_ROOT / args.outdir
    with plt.rc_context(RC):
        for name in wanted:
            FIGURES[name](outdir)
    print(f"\n{len(wanted)} figure(s) in {outdir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
