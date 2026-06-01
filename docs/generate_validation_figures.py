#!/usr/bin/env python
"""Generate the WSWUP cross-validation figure for the monograph.

Reads validation/results/fruita_pairs.npz (paired per-pixel UAV vs WSWUP
reference values, produced by validation/run_validation.py) and renders a
three-panel 1:1 comparison of H, LE, and instantaneous ET. House style matches
generate_figures.py (9 pt, 150 dpi, study red).

Run:  python docs/generate_validation_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 9
rcParams["axes.titlesize"] = 10
rcParams["axes.labelsize"] = 9
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 150
rcParams["savefig.bbox"] = "tight"
STUDY_RED = "#c1272d"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figures")
PAIRS = os.path.join(ROOT, "validation", "results", "fruita_pairs.npz")
os.makedirs(FIG_DIR, exist_ok=True)


def panel(ax, x, y, label, unit, max_pts=8000):
    """1:1 scatter of UAV (y) vs WSWUP reference (x) with OLS + metrics."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size > max_pts:
        idx = np.random.default_rng(0).choice(x.size, max_pts, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    pad = 0.04 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1.0, zorder=1)
    ax.scatter(xs, ys, s=3, alpha=0.18, color=STUDY_RED, edgecolors="none",
               zorder=2)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.array([lo, hi])
    ax.plot(xx, slope * xx + intercept, color="k", lw=1.0, ls="--", zorder=3)
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    ss_res = np.sum((y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("WSWUP reference %s (%s)" % (label, unit))
    ax.set_ylabel("pyMETRIC-UAV %s (%s)" % (label, unit))
    ax.text(0.04, 0.96, "$R^2$=%.3f\nRMSE=%.3g\nslope=%.3f" % (r2, rmse, slope),
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                      alpha=0.85))
    ax.set_title(label)


def main():
    d = np.load(PAIRS)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    panel(axes[0], d["H_ref"], d["H_uav"], "H", "W m$^{-2}$")
    panel(axes[1], d["LE_ref"], d["LE_uav"], "LE", "W m$^{-2}$")
    panel(axes[2], d["ET_ref"], d["ET_uav"], "ET", "mm hr$^{-1}$")
    fig.suptitle("pyMETRIC-UAV vs WSWUP reference solver on the Fruita scene "
                 "(identical $R_n$, $G$, $T_s$, anchors)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, "fig_val_fruita." + ext))
    plt.close(fig)
    print("wrote fig_val_fruita.png and .pdf")


if __name__ == "__main__":
    main()
