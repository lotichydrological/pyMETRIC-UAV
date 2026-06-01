"""
Cross-validate the pyMETRIC-UAV energy-balance solver against the genuine
WSWUP/pyMETRIC solver.

Strategy (solver cross-check):
  Both solvers are driven with IDENTICAL net radiation (Rn), soil heat flux
  (G), surface temperature (Ts), momentum roughness (zom), anchor pixel
  locations, and anchor latent-heat targets (LE_cold, LE_hot). Each then
  applies its own native dT-calibration + aerodynamic-resistance + Monin-
  Obukhov stability solver. Differences therefore isolate THE SOLVER, not
  the upstream derivation of Rn/G/Ts/anchors (which differ between the two
  packages by design and are held fixed here).

Two scenes:
  1. Fruita (real UAV scene, 0.25 m): the UAV solution is the model's own
     stored output (Rn, H, LE, G, rah, ET); the WSWUP reference is driven on
     the same Rn/G/Ts/anchors/anchor-LE.
  2. Synthetic Landsat-scale agricultural scene (30 m): WSWUP's design
     domain. Both solvers are driven from scratch on the same generated
     fields, so this independently exercises the UAV's integrated
     METRIC.METRIC() against the WSWUP reference.

Outputs (written to validation/results/):
  metrics.json         per-variable RMSE / MBE / R2 / slope / n
  fruita_pairs.npz     subsampled paired arrays for plotting
  synthetic_pairs.npz  paired arrays for plotting
"""

import json
import os
import re
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
UAV_ROOT = os.path.dirname(HERE)
sys.path.insert(0, UAV_ROOT)
sys.path.insert(0, HERE)

import wswup_reference as wswup  # noqa: E402

from pyTSEB import meteo_utils as met  # noqa: E402
from pyTSEB import resistances as res  # noqa: E402
from pyTSEB import net_radiation as rad  # noqa: E402
from pyMETRIC import METRIC as uav_metric  # noqa: E402

FRUITA = os.path.join(UAV_ROOT, "Batch_Data", "Fruita_Example")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(ref, test, name):
    """Agreement metrics of `test` (UAV) vs `ref` (WSWUP) over finite pairs."""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    m = np.isfinite(ref) & np.isfinite(test)
    ref, test = ref[m], test[m]
    n = ref.size
    if n < 2:
        return dict(var=name, n=int(n), rmse=None, mbe=None, r2=None,
                    slope=None, intercept=None, ref_mean=None, test_mean=None)
    diff = test - ref
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mbe = float(np.mean(diff))
    # OLS test = slope*ref + intercept
    slope, intercept = np.polyfit(ref, test, 1)
    ss_res = np.sum((test - (slope * ref + intercept)) ** 2)
    ss_tot = np.sum((test - np.mean(test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    return dict(var=name, n=int(n), rmse=rmse, mbe=mbe, r2=r2,
                slope=float(slope), intercept=float(intercept),
                ref_mean=float(np.mean(ref)), test_mean=float(np.mean(test)))


# ---------------------------------------------------------------------------
# Roughness (shared by both solvers)
# ---------------------------------------------------------------------------
def compute_zom(lai, h_c, w_c, landcover, f_c, z0_soil=0.01):
    """Momentum roughness via pyTSEB (the UAV model's own method).

    Vegetated pixels (LAI>0) use res.calc_roughness; bare pixels use z0_soil.
    Falls back to 0.018*LAI (clamped) if pyTSEB errors on a pixel set.
    """
    zom = np.full(lai.shape, z0_soil, dtype=np.float64)
    veg = lai > 0
    if np.any(veg):
        try:
            # pyTSEB calc_roughness returns (z_0M, d_0, ...); take the first.
            rough = res.calc_roughness(
                lai[veg],
                np.full(np.sum(veg), h_c),
                w_C=np.full(np.sum(veg), w_c),
                landcover=np.full(np.sum(veg), landcover, dtype=int),
                f_c=f_c[veg] if hasattr(f_c, "__len__") else
                np.full(np.sum(veg), f_c))
            z0m = np.asarray(rough[0], dtype=np.float64)
            z0m = np.where(np.isfinite(z0m) & (z0m > 0), z0m, z0_soil)
            zom[veg] = np.maximum(z0m, 1e-4)
        except Exception as exc:  # pragma: no cover
            print("  pyTSEB roughness fallback (%s); using 0.018*LAI" % exc)
            zom[veg] = np.maximum(0.018 * lai[veg], z0_soil)
    return zom


# ---------------------------------------------------------------------------
# Scene 1 — Fruita
# ---------------------------------------------------------------------------
def run_fruita():
    print("\n=== Scene 1: Fruita (real UAV scene, 0.25 m) ===")
    out_tif = os.path.join(FRUITA, "Output", "Fruita_Example_METRIC.tif")
    anc_tif = os.path.join(FRUITA, "Output",
                           "Fruita_Example_METRIC_ancillary.tif")
    et_tif = os.path.join(FRUITA, "Output", "Fruita_Example_ET_mm_hour.tif")

    prim = gdal.Open(out_tif)
    Rn = prim.GetRasterBand(1).ReadAsArray().astype(np.float64)   # net radiation
    H_uav = prim.GetRasterBand(2).ReadAsArray().astype(np.float64)
    LE_uav = prim.GetRasterBand(3).ReadAsArray().astype(np.float64)
    G = prim.GetRasterBand(4).ReadAsArray().astype(np.float64)    # soil heat flux
    anc = gdal.Open(anc_tif)
    rah_uav = anc.GetRasterBand(3).ReadAsArray().astype(np.float64)
    etd = gdal.Open(et_tif)
    ET_uav = etd.GetRasterBand(1).ReadAsArray().astype(np.float64)

    ts = gdal.Open(os.path.join(FRUITA, "Input", "TRAD.tif")
                   ).ReadAsArray().astype(np.float64)
    lai = gdal.Open(os.path.join(FRUITA, "Input", "LAI_NDVI.tif")
                    ).ReadAsArray().astype(np.float64)
    fc = gdal.Open(os.path.join(FRUITA, "Input", "FC.tif")
                   ).ReadAsArray().astype(np.float64)

    # Met + site (from config.txt)
    T_A = 27.38 + 273.15
    ea = 7.2
    p = 1019.0
    u_obs = 1.63
    alt = 1404.0

    # No elevation delapse: the UAV solver now calibrates dT against the raw
    # radiometric surface temperature (common-elevation UAV scenes). Drive the
    # WSWUP reference on the same raw Ts so the two solvers are compared on an
    # identical calibration temperature. elev is still passed (the WSWUP density
    # helper uses it) but no datum offset is applied to Ts.
    tsdem = ts
    elev = np.full(ts.shape, alt, dtype=np.float64)

    # Anchors, anchor LE, and the UAV's own calibration are read from the
    # processing log so this harness never drifts from the current run.
    log_txt = open(os.path.join(FRUITA, "Output",
                                "Fruita_Example_log.txt")).read()
    m_cold = re.search(r"Cold pixel at \((\d+), (\d+)\)", log_txt)
    m_hot = re.search(r"Hot pixel at \((\d+), (\d+)\)", log_txt)
    cold_rc = (int(m_cold.group(1)), int(m_cold.group(2)))
    hot_rc = (int(m_hot.group(1)), int(m_hot.group(2)))
    m_cap = re.search(r"capping LE_cold from [\d.]+ to ([\d.]+)", log_txt)
    le_cold_anchor = float(m_cap.group(1)) if m_cap else float(LE_uav[cold_rc])
    le_hot_anchor = float(LE_uav[hot_rc])
    le_anchor = np.array([le_cold_anchor, le_hot_anchor])
    m_cal = re.search(r"Calibrated dT: a=([-\d.]+), b=([-\d.]+)", log_txt)
    uav_a_run = float(m_cal.group(1))   # intercept (UAV convention)
    uav_b_run = float(m_cal.group(2))   # slope (UAV convention)

    # Shared roughness (UAV's own pyTSEB method)
    zom = compute_zom(lai, h_c=0.4, w_c=1.0, landcover=12, f_c=fc, z0_soil=0.01)

    print("  Anchors: cold=%s Ts=%.2fK  hot=%s Ts=%.2fK"
          % (cold_rc, ts[cold_rc], hot_rc, ts[hot_rc]))
    print("  Anchor Rn/G: cold=%.1f/%.1f  hot=%.1f/%.1f"
          % (Rn[cold_rc], G[cold_rc], Rn[hot_rc], G[hot_rc]))

    ref = wswup.solve(ts, tsdem, elev, Rn, G, (cold_rc, hot_rc), le_anchor,
                      u_obs, zom)

    # WSWUP calibration in terms of Tr_datum: dt = a*tsdem + b
    print("  WSWUP reference calibration: a(slope)=%.6f  b(intercept)=%.4f"
          % (ref["a"], ref["b"]))
    # UAV convention: dT = a_intercept + b_slope*Ts. Print in slope/intercept
    # order to match the WSWUP line above (WSWUP a=slope, b=intercept).
    print("  UAV run calibration:         a(slope)=%.6f  b(intercept)=%.4f"
          % (uav_b_run, uav_a_run))
    print("  WSWUP u3 (200 m wind)=%.3f m/s" % ref["u3"])

    # Restrict comparison to valid pixels in both
    valid = (np.isfinite(H_uav) & np.isfinite(ref["h"]) &
             np.isfinite(LE_uav) & np.isfinite(ET_uav))

    res_metrics = [
        metrics(ref["h"][valid], H_uav[valid], "H (W/m2)"),
        metrics(ref["le"][valid], LE_uav[valid], "LE (W/m2)"),
        metrics(ref["et_inst"][valid], ET_uav[valid], "ET (mm/hr)"),
        metrics(ref["rah"][valid], rah_uav[valid], "rah (s/m)"),
    ]
    for r in res_metrics:
        print("  %-12s n=%d RMSE=%.3f MBE=%.3f R2=%.4f slope=%.3f"
              % (r["var"], r["n"], r["rmse"], r["mbe"], r["r2"], r["slope"]))

    # Subsample paired arrays for plotting (cap ~40k points)
    idx = np.where(valid.ravel())[0]
    if idx.size > 40000:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, 40000, replace=False))
    np.savez_compressed(
        os.path.join(RESULTS, "fruita_pairs.npz"),
        H_ref=ref["h"].ravel()[idx], H_uav=H_uav.ravel()[idx],
        LE_ref=ref["le"].ravel()[idx], LE_uav=LE_uav.ravel()[idx],
        ET_ref=ref["et_inst"].ravel()[idx], ET_uav=ET_uav.ravel()[idx],
        rah_ref=ref["rah"].ravel()[idx], rah_uav=rah_uav.ravel()[idx])

    # wswup_a/wswup_b follow WSWUP convention (a=slope, b=intercept).
    # uav_slope/uav_intercept are the UAV run's dT = intercept + slope*Ts.
    return dict(scene="Fruita", n_pixels=int(valid.sum()),
                wswup_a=ref["a"], wswup_b=ref["b"],
                uav_slope=uav_b_run, uav_intercept=uav_a_run,
                wswup_u3=ref["u3"], metrics=res_metrics)


# ---------------------------------------------------------------------------
# Scene 2 — Synthetic Landsat-scale agricultural scene
# ---------------------------------------------------------------------------
def run_synthetic():
    print("\n=== Scene 2: Synthetic Landsat-scale ag scene (30 m) ===")
    ny, nx = 80, 80
    rng = np.random.default_rng(7)

    # NDVI gradient: wet dense crop (NW) -> hot bare soil (SE), + noise
    yy, xx = np.mgrid[0:ny, 0:nx]
    yy = yy / float(ny)
    xx = xx / float(nx)
    grad = 1.0 - 0.5 * (yy + xx)            # 1 (NW) .. 0 (SE)
    ndvi = np.clip(0.15 + 0.78 * grad + rng.normal(0, 0.03, (ny, nx)), 0.05, 0.95)
    lai = np.clip(7.0 * ndvi ** 3, 0.0, 6.0)

    # Surface temperature anti-correlated with NDVI (cool veg, hot soil)
    ts = 300.0 + (1.0 - ndvi) * 28.0 + rng.normal(0, 0.4, (ny, nx))  # ~300-326 K

    # Met / site (semi-arid summer midday)
    T_A = 298.0
    ea = 12.0
    p = 1013.0
    u_obs = 3.0
    alt = 100.0
    S_dn = 850.0
    doy, lon, lat, stdlon, t_decimal = 200, -119.0, 43.0, -120.0, 11.0

    elev = np.full(ts.shape, alt, dtype=np.float64)
    emis = 0.97 + 0.01 * ndvi                       # 0.97-0.98
    albedo = np.clip(0.18 + 0.10 * (1 - ndvi), 0.15, 0.30)

    # --- Shared energy inputs: Sn, Rn, G ---
    Sn = (1.0 - albedo) * S_dn
    L_dn = rad.calc_emiss_atm(np.array([ea]), np.array([T_A]))[0] * \
        met.calc_stephan_boltzmann(np.array([T_A]))[0]
    Ln = emis * L_dn - emis * met.calc_stephan_boltzmann(ts)
    Rn = Sn + Ln
    # G: Allen SEBAL/empirical form (identical for both)
    G = uav_metric.calc_G_Allen(Rn, ts, albedo, ndvi)

    # Roughness (shared)
    zom = compute_zom(lai, h_c=0.5, w_c=1.0, landcover=12, f_c=ndvi,
                      z0_soil=0.01)

    # Anchors: coldest high-NDVI pixel; hottest low-NDVI pixel
    cold_rc = np.unravel_index(
        np.argmax(np.where(ndvi > 0.7, -ts, -1e9)), ts.shape)
    hot_rc = np.unravel_index(
        np.argmax(np.where(ndvi < 0.25, ts, -1e9)), ts.shape)
    print("  Anchors: cold=%s Ts=%.2fK NDVI=%.2f  hot=%s Ts=%.2fK NDVI=%.2f"
          % (cold_rc, ts[cold_rc], ndvi[cold_rc], hot_rc, ts[hot_rc],
             ndvi[hot_rc]))

    # Reference ET (instantaneous, W/m2) via the UAV's ASCE routine; anchor LE
    # targets: cold = 1.05*ET0, hot = 0.0 (classic METRIC anchors).
    z_u = np.full(ts.shape, 2.0)
    z_T = np.full(ts.shape, 2.0)
    ET0 = uav_metric.pet_asce(np.full(ts.shape, T_A), np.full(ts.shape, u_obs),
                              np.full(ts.shape, ea), np.full(ts.shape, p),
                              np.full(ts.shape, S_dn), z_u, z_T,
                              reference=uav_metric.TALL_REFERENCE)
    le_cold = 1.05 * ET0
    le_hot = np.zeros(ts.shape)

    # --- UAV solver: call the real integrated METRIC.METRIC() ---
    # METRIC.METRIC expects 1-D masked arrays (PyMETRIC always calls it that
    # way), so flatten every field and pass flat anchor indices.
    shp = ts.shape
    flat = lambda a: np.asarray(a, dtype=np.float64).ravel()
    cold_flat = cold_rc[0] * nx + cold_rc[1]
    hot_flat = hot_rc[0] * nx + hot_rc[1]
    g_ratio_flat = (G / Rn).ravel()
    out = uav_metric.METRIC(
        flat(ts), np.full(ts.size, T_A), np.full(ts.size, u_obs),
        np.full(ts.size, ea), np.full(ts.size, p), flat(Sn),
        np.full(ts.size, L_dn), flat(emis), flat(zom), np.zeros(ts.size),
        np.full(ts.size, 2.0), np.full(ts.size, 2.0),
        cold_pixel=cold_flat, hot_pixel=hot_flat,
        LE_cold=flat(le_cold), LE_hot=flat(le_hot),
        # Pass int 1 (not bool True) exactly as PyMETRIC._call_flux_model does:
        # METRIC.py tests `use_METRIC_resistance is True`, so the int takes the
        # scaling `else` branch (z_T[i]/d_0[i]/z_0H[i]); literal True would hit a
        # latent dead branch hardcoded to length-2 anchor arrays.
        use_METRIC_resistance=1,
        calcG_params=[[1], g_ratio_flat],   # identical G as a ratio of Rn
        UseDEM=False)
    flag, Ln_u, LE_u, H_u, G_u, RA_u, ustar_u, L_u, nit = out
    H_u = H_u.reshape(shp); LE_u = LE_u.reshape(shp); RA_u = RA_u.reshape(shp)
    ET_u = (LE_u * 3600.0) / ((2.501 - 0.002361 * (ts - 273.15)) * 1e6)

    # --- WSWUP reference: identical Rn, G, Ts, zom, anchors, anchor-LE ---
    le_anchor = np.array([le_cold[cold_rc], le_hot[hot_rc]])
    ref = wswup.solve(ts, ts, elev, Rn, G, (cold_rc, hot_rc), le_anchor,
                      u_obs, zom)
    ET_ref = ref["et_inst"]

    print("  UAV  calibration (dT=a+b*Ts): from METRIC.METRIC log above")
    print("  WSWUP calibration: a(slope)=%.6f b(intercept)=%.4f u3=%.3f"
          % (ref["a"], ref["b"], ref["u3"]))

    valid = np.isfinite(H_u) & np.isfinite(ref["h"]) & np.isfinite(LE_u)
    res_metrics = [
        metrics(ref["h"][valid], H_u[valid], "H (W/m2)"),
        metrics(ref["le"][valid], LE_u[valid], "LE (W/m2)"),
        metrics(ET_ref[valid], ET_u[valid], "ET (mm/hr)"),
        metrics(ref["rah"][valid], RA_u[valid], "rah (s/m)"),
    ]
    for r in res_metrics:
        print("  %-12s n=%d RMSE=%.3f MBE=%.3f R2=%.4f slope=%.3f"
              % (r["var"], r["n"], r["rmse"], r["mbe"], r["r2"], r["slope"]))

    np.savez_compressed(
        os.path.join(RESULTS, "synthetic_pairs.npz"),
        H_ref=ref["h"][valid], H_uav=H_u[valid],
        LE_ref=ref["le"][valid], LE_uav=LE_u[valid],
        ET_ref=ET_ref[valid], ET_uav=ET_u[valid],
        rah_ref=ref["rah"][valid], rah_uav=RA_u[valid],
        ndvi=ndvi[valid], ts=ts[valid])

    return dict(scene="Synthetic_30m", n_pixels=int(valid.sum()),
                wswup_a=ref["a"], wswup_b=ref["b"], wswup_u3=ref["u3"],
                metrics=res_metrics)


if __name__ == "__main__":
    summary = {}
    summary["fruita"] = run_fruita()
    summary["synthetic"] = run_synthetic()
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote", os.path.join(RESULTS, "metrics.json"))
