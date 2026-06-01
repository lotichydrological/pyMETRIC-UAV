#!/usr/bin/env python
"""Generate the Quarto validation section mechanically from metrics.json.

Every number in the output originates from validation/results/metrics.json —
nothing is transcribed by hand. This eliminates the possibility of fabricated
or stale figures in the document: re-run after any change to the validation and
the prose + table update themselves.

Writes to a path given on the command line (default: a LOCAL, non-Drive path,
so Google Drive sync cannot revert it). Paste/move the result into
docs/pyMETRIC_UAV_overview.qmd immediately before "## The Fruita worked
example" once Drive sync is stable.

Usage:
    python make_section.py [output_path]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(HERE, "results", "metrics.json")
DEFAULT_OUT = os.path.expanduser("~/pymetric_validation_section.qmd")


def m(scene, prefix):
    """Return the metric dict for a variable in a scene, or None."""
    for row in DATA[scene]["metrics"]:
        if row["var"].startswith(prefix):
            return row
    return None


def fmt(x, nd=3):
    return ("%." + str(nd) + "f") % x


DATA = json.load(open(METRICS))
fr, sy = DATA["fruita"], DATA["synthetic"]

fr_et, fr_h, fr_rah = m("fruita", "ET"), m("fruita", "H"), m("fruita", "rah")
sy_et, sy_h, sy_rah = m("synthetic", "ET"), m("synthetic", "H"), m("synthetic", "rah")

# Mean ET percentage differences, computed from the file
fr_et_pct = 100.0 * (fr_et["test_mean"] - fr_et["ref_mean"]) / fr_et["ref_mean"]
sy_et_pct = 100.0 * (sy_et["test_mean"] - sy_et["ref_mean"]) / sy_et["ref_mean"]
fr_et_dir = "lower" if fr_et_pct < 0 else "higher"
sy_et_dir = "lower" if sy_et_pct < 0 else "higher"
min_et_r2 = min(fr_et["r2"], sy_et["r2"])


def table_row(scene_label, mrow, unit, nd_rmse=2):
    return ("| %s | %s | %s | %s | %+.4g | %s |"
            % (scene_label, mrow["var"], fmt(mrow["r2"], 3),
               ("%." + str(nd_rmse) + "g") % mrow["rmse"], mrow["mbe"],
               fmt(mrow["slope"], 3)))


SECTION = r"""## Validation against WSWUP/pyMETRIC

The modifications described above change *how* anchor pixels are found and how
the calibration is protected from degenerate forcing; they are not intended to
change the underlying METRIC energy balance. This section cross-checks the
adapted solver against [WSWUP/pyMETRIC](https://github.com/WSWUP/pymetric) —
the University of Idaho / Desert Research Institute implementation widely used
for operational Landsat ET and a *de facto* reference for the algorithm
[@allen2007metric].

### Method

WSWUP/pyMETRIC and pyMETRIC-UAV are independent codebases with different
lineages (WSWUP descends from the original Idaho METRIC; pyMETRIC-UAV derives
from `hectornieto/pyMETRIC`, built on pyTSEB). They also derive their inputs
differently — WSWUP computes surface variables from Landsat bands and ingests
gridded reference ET, whereas pyMETRIC-UAV consumes user NDVI/LST rasters and
computes reference ET internally. A meaningful comparison therefore has to
**isolate the energy-balance solver** from those upstream differences.

The genuine WSWUP energy-balance functions (`et_numpy.py`, imported unmodified
from a pinned clone) are driven through the exact control flow of
`metric_model2_func.py` — the 20-iteration pixel calibration loop and the
6-iteration raster stability loop, with the WSWUP resistance heights
$z = \{0.1, 2, 200\}\,\text{m}$ — by a thin reference harness
(`validation/wswup_reference.py`). Both solvers are then handed **identical**
net radiation $R_n$, soil heat flux $G$, surface temperature $T_s$, the same
two anchor pixels, and the same anchor latent-heat targets
($\mathrm{LE_{cold}}, \mathrm{LE_{hot}}$). Each solver then performs its *own*
native $dT$ calibration, aerodynamic-resistance calculation, and
Monin–Obukhov stability iteration. Disagreement is thus attributable to the
solver internals rather than to upstream input derivation.

At the level of individual physics functions the two implementations are
identical: driven on common scalars, WSWUP's `h_func`, `dt_calibration_func`,
and `le_func` reproduce pyMETRIC-UAV's `calc_H`, `calc_dT`, and the
$\mathrm{LE} = R_n - G - H$ residual to machine precision. The comparison below
therefore tests the *assembled* solver — calibration plus resistance plus
stability — not the equations in isolation.

Two scenes are used: the real Fruita UAV scene ({FR_N} px at 0.25 m), and a
synthetic Landsat-scale agricultural scene ({SY_N} px at 30 m) spanning a
wet-crop-to-bare-soil gradient — i.e. WSWUP's native design domain. On the
synthetic scene the comparison drives the *integrated* `METRIC.METRIC()`
routine directly, so it exercises the production UAV solver end-to-end.

### Results

@fig-val-fruita and @fig-val-synthetic show the per-pixel correspondence of
sensible heat $H$, latent heat $\mathrm{LE}$, and instantaneous ET; the headline
metrics are collected in @tbl-validation. Given identical energy inputs and
anchor constraints, the two independent solvers reproduce the same flux fields.
Instantaneous ET agrees closely on both scenes — $R^2 = {FR_ET_R2}$ on Fruita
and ${SY_ET_R2}$ on the synthetic scene — with mean ET within a few percent:
{FR_ET_REF} vs {FR_ET_UAV} mm hr⁻¹ on Fruita (UAV {FR_ET_PCT}% {FR_ET_DIR}) and
{SY_ET_REF} vs {SY_ET_UAV} mm hr⁻¹ on the synthetic scene
(UAV {SY_ET_PCT}% {SY_ET_DIR}). Sensible and latent heat agree comparably.

![pyMETRIC-UAV versus the WSWUP reference solver on the Fruita scene
({FR_N} px, 0.25 m), each run with its native calibration and resistance
scheme on identical $R_n$, $G$, $T_s$, and anchors. Grey is the 1:1 line,
dashed is the OLS fit.](figures/fig_val_fruita.png){#fig-val-fruita}

![The same comparison on a synthetic Landsat-scale agricultural scene
({SY_N} px, 30 m), driving the integrated `METRIC.METRIC()` solver
end-to-end.](figures/fig_val_synthetic.png){#fig-val-synthetic}

{TABLE}

: Solver cross-validation. MBE and slope are pyMETRIC-UAV relative to the WSWUP
reference (MBE = UAV − WSWUP). Generated mechanically from
`validation/results/metrics.json`. {#tbl-validation}

### Interpretation

The closest agreement is in the fluxes that METRIC self-calibration constrains
most directly. Because both solvers are pinned to the *same* two anchor
sensible-heat targets ($H = R_n - G - \mathrm{LE}$ at the cold and hot pixels)
and are driven with identical $T_s$, $R_n$, and $G$, the spatial $H$ field is
largely determined — and the two reproduce it almost exactly (Fruita $H$
$R^2 = {FR_H_R2}$, slope {FR_H_SLOPE}; @fig-val-fruita). Latent heat and ET
follow as the energy residual.

The genuine solver-to-solver difference lives in the **aerodynamic resistance**
$r_{ah}$ (@fig-val-rah). On Fruita the UAV model's $r_{ah}$ runs systematically
higher than the reference (means {FR_RAH_UAV} vs {FR_RAH_REF} s m⁻¹, OLS slope
{FR_RAH_SLOPE}), though still spatially well-correlated ($R^2 = {FR_RAH_R2}$).
This is the expected consequence of two genuinely different aerodynamic
treatments: pyMETRIC-UAV iterates $r_{ah}$ with pyTSEB's Monin–Obukhov scheme
using the measured 2 m wind over each pixel's own roughness, whereas WSWUP
extrapolates wind to a 200 m blending height over a fixed station roughness
(0.015 m).

Crucially, this $r_{ah}$ difference is *absorbed by the self-calibration*
rather than propagating into ET. Because METRIC inverts the anchors through
$r_{ah}$ to fit a fixed anchor $H$, a solver with larger $r_{ah}$ simply
recovers a larger $dT$ slope to compensate: on Fruita the UAV calibration slope
is {UAV_A} against the reference's {WSWUP_A}, and the
larger-$dT$/larger-$r_{ah}$ (UAV) and smaller-$dT$/smaller-$r_{ah}$ (WSWUP)
pairings yield nearly the same $H = \rho c_p\, dT / r_{ah}$. The compensation is
why ET agreement survives the difference in the underlying resistance.

Two caveats bound this result. First, the agreement is *partly by
construction*: feeding both solvers identical $R_n$, $G$, $T_s$, anchor pixels,
and anchor LE targets is exactly the design that isolates the solver, but it
also means the comparison tests the assembled calibration–resistance–stability
machinery given common inputs, not the independent derivation of those inputs
(which the two packages do differently by design, and which is held fixed
here). Second, the Fruita "UAV" side is the model's own stored output, whereas
the synthetic scene drives `METRIC.METRIC()` from scratch — and both routes land
in close agreement with the reference.

The bounded conclusion: pyMETRIC-UAV implements the same METRIC energy-balance
core as WSWUP/pyMETRIC — identical at the individual-function level, and
agreeing on instantaneous ET to within a few percent in the mean
($R^2 \ge {MIN_ET_R2}$) when both are given the same energy and anchors. The
aerodynamic-resistance scheme is where the two implementations genuinely
differ, but METRIC's internalized calibration absorbs that difference. The
UAV-specific changes documented earlier therefore alter how anchors are
selected and protected, not the physics of the solution they feed. The
reproduction harness (pinned WSWUP commit, driver, and figure generator) lives
in `validation/`.

![Aerodynamic resistance $r_{ah}$ agreement on both scenes — the largest
single source of solver-to-solver difference, reflecting the distinct
Monin–Obukhov / blending-height wind treatments.](figures/fig_val_rah.png){#fig-val-rah}

"""

table = "\n".join([
    "| Scene | Variable | $R^2$ | RMSE | MBE | OLS slope |",
    "|:------|:---------|------:|-----:|------:|----------:|",
    table_row("Fruita (0.25 m, n=%d)" % fr["n_pixels"], fr_h, "W/m2"),
    table_row("", m("fruita", "LE"), "W/m2"),
    table_row("", fr_et, "mm/hr", nd_rmse=2),
    table_row("", fr_rah, "s/m"),
    table_row("Synthetic (30 m, n=%d)" % sy["n_pixels"], sy_h, "W/m2"),
    table_row("", m("synthetic", "LE"), "W/m2"),
    table_row("", sy_et, "mm/hr", nd_rmse=2),
    table_row("", sy_rah, "s/m"),
])

text = SECTION
repl = {
    "{FR_N}": "{:,}".format(fr["n_pixels"]),
    "{SY_N}": "{:,}".format(sy["n_pixels"]),
    "{FR_ET_R2}": fmt(fr_et["r2"], 3),
    "{SY_ET_R2}": fmt(sy_et["r2"], 3),
    "{FR_ET_REF}": fmt(fr_et["ref_mean"], 3),
    "{FR_ET_UAV}": fmt(fr_et["test_mean"], 3),
    "{SY_ET_REF}": fmt(sy_et["ref_mean"], 3),
    "{SY_ET_UAV}": fmt(sy_et["test_mean"], 3),
    "{FR_ET_PCT}": fmt(abs(fr_et_pct), 1),
    "{SY_ET_PCT}": fmt(abs(sy_et_pct), 1),
    "{FR_ET_DIR}": fr_et_dir,
    "{SY_ET_DIR}": sy_et_dir,
    "{FR_H_R2}": fmt(fr_h["r2"], 4),
    "{FR_H_SLOPE}": fmt(fr_h["slope"], 3),
    "{FR_RAH_UAV}": fmt(fr_rah["test_mean"], 1),
    "{FR_RAH_REF}": fmt(fr_rah["ref_mean"], 1),
    "{FR_RAH_SLOPE}": fmt(fr_rah["slope"], 3),
    "{FR_RAH_R2}": fmt(fr_rah["r2"], 3),
    "{UAV_A}": fmt(fr["uav_a"], 4),
    "{WSWUP_A}": fmt(fr["wswup_a"], 4),
    "{MIN_ET_R2}": fmt(min_et_r2, 2),
    "{TABLE}": table,
}
for k, v in repl.items():
    text = text.replace(k, v)

out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
with open(out_path, "w") as f:
    f.write(text)

# Console summary (also written to a sibling .txt for reliable inspection)
summary = "Wrote validation section (%d chars) to %s\n" % (len(text), out_path)
summary += "Numbers used (all from metrics.json):\n"
for sc, lbl in [("fruita", "Fruita"), ("synthetic", "Synthetic")]:
    summary += "  %s n=%d\n" % (lbl, DATA[sc]["n_pixels"])
    for row in DATA[sc]["metrics"]:
        summary += ("    %-11s R2=%.3f RMSE=%.4g MBE=%+.4g slope=%.3f "
                    "ref=%.4g uav=%.4g\n" % (
                        row["var"], row["r2"], row["rmse"], row["mbe"],
                        row["slope"], row["ref_mean"], row["test_mean"]))
with open(out_path + ".summary.txt", "w") as f:
    f.write(summary)
print(summary)
