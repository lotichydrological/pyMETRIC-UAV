#!/usr/bin/env python
"""Export every Fruita worked-example value cited in the monograph to JSON.

Single source of truth for the Quarto document. The qmd reads
``docs/_fruita_metrics.json`` and prints values inline; nothing is transcribed
by hand. Re-run this after any model re-run, then re-render the qmd, and the
prose, tables, and figures stay mutually consistent by construction.

Reads only the committed run outputs under Batch_Data/Fruita_Example/ plus the
processing log (for the exact calibration coefficients). Per-strategy dT
coefficients are recovered from each strategy's output rasters by the exact
identity dT = H * rah / (rho * cp), regressed on Ts_datum (r^2 = 1 by
construction for a linear calibration).

Usage:  python export_metrics.py        # writes docs/_fruita_metrics.json
"""
import json
import os
import re
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FRUITA = os.path.join(ROOT, "Batch_Data", "Fruita_Example")
OUT = os.path.join(FRUITA, "Output")
INP = os.path.join(FRUITA, "Input")
CONFIG = os.path.join(FRUITA, "config.txt")
LOG = os.path.join(OUT, "Fruita_Example_log.txt")
DEST = os.path.join(HERE, "_fruita_metrics.json")

sys.path.insert(0, ROOT)
from pyTSEB import meteo_utils as met  # noqa: E402  (after sys.path insert)


def band(path, b):
    """Read one band as float64; keep the dataset alive until after read."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    arr = ds.GetRasterBand(b).ReadAsArray().astype(np.float64)
    ds = None
    return arr


def stats(a):
    v = a[np.isfinite(a)]
    return dict(mean=float(np.mean(v)), median=float(np.median(v)),
                min=float(np.min(v)), max=float(np.max(v)))


def parse_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            s = line.split("#")[0].strip()
            if "=" in s:
                k, v = s.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def parse_log(path):
    """Pull exact, run-emitted values from the processing log."""
    txt = open(path).read()
    out = {}
    m = re.search(r"Calibrated dT: a=([-\d.]+), b=([-\d.]+)", txt)
    if m:
        out["a"] = float(m.group(1))
        out["b"] = float(m.group(2))
    m = re.search(r"Cold pixel at \((\d+), (\d+)\): ([\d.]+) K, ([\d.]+) VI", txt)
    if m:
        out["cold_rc"] = [int(m.group(1)), int(m.group(2))]
        out["cold_ts"] = float(m.group(3))
        out["cold_vi"] = float(m.group(4))
    m = re.search(r"Hot pixel at \((\d+), (\d+)\): ([\d.]+) K, ([\d.]+) VI", txt)
    if m:
        out["hot_rc"] = [int(m.group(1)), int(m.group(2))]
        out["hot_ts"] = float(m.group(3))
        out["hot_vi"] = float(m.group(4))
    m = re.search(r"LST range = ([\d.]+) K", txt)
    if m:
        out["lst_range"] = float(m.group(1))
    m = re.search(r"capping LE_cold from ([\d.]+) to ([\d.]+) "
                  r"\(Rn-G=([\d.]+), H_cold floored at ([\d.]+)", txt)
    if m:
        out["le_cold_precap"] = float(m.group(1))
        out["le_cold_postcap"] = float(m.group(2))
        out["rn_minus_g_cold"] = float(m.group(3))
        out["h_cold_floor"] = float(m.group(4))
    return out


def recover_ab(stem, tsd, rho_cp):
    """Recover dT = a + b*Ts_datum for one strategy from its output rasters."""
    H = band(os.path.join(OUT, stem + ".tif"), 2)
    rah = band(os.path.join(OUT, stem + "_ancillary.tif"), 3)
    dT = H * rah / rho_cp
    m = np.isfinite(dT) & np.isfinite(tsd) & np.isfinite(H) & (rah > 0)
    b, a = np.polyfit(tsd[m], dT[m], 1)  # slope, intercept
    pred = a + b * tsd[m]
    ss_res = np.sum((dT[m] - pred) ** 2)
    ss_tot = np.sum((dT[m] - np.mean(dT[m])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def main():
    cfg = parse_config(CONFIG)
    log = parse_log(LOG)

    # Met / site, exactly as the model uses them
    T_A = float(cfg["T_A1"]) + 273.15  # config T_A1 is in C
    ea = float(cfg["ea"])
    p = float(cfg["p"])
    alt = float(cfg["alt"])
    rho = float(met.calc_rho(p, ea, T_A))
    cp = float(met.calc_c_p(p, ea))
    rho_cp = rho * cp
    # No elevation delapse: the model now calibrates dT against the raw
    # radiometric surface temperature (common-elevation UAV scenes), so there
    # is no datum offset. gamma_w retained as 0 for provenance.
    gamma_w = 0.0

    # Primary + ancillary fluxes (main run)
    prim = os.path.join(OUT, "Fruita_Example_METRIC.tif")
    anc = os.path.join(OUT, "Fruita_Example_METRIC_ancillary.tif")
    Rn = band(prim, 1)
    H = band(prim, 2)
    LE = band(prim, 3)
    G = band(prim, 4)
    rah = band(anc, 3)
    fetr = band(anc, 9)
    ts = band(os.path.join(INP, "TRAD.tif"), 1)
    ndvi = band(os.path.join(INP, "NDVI.tif"), 1)
    lai = band(os.path.join(INP, "LAI_NDVI.tif"), 1)

    # ET unfloored (matches the LE/fETr table rows) and floored (the saved map)
    lam = (2.501 - 0.002361 * (ts - 273.15)) * 1e6
    et_unfloored = LE * 3600.0 / lam
    et_floored = np.where(et_unfloored < 0, 0.0, et_unfloored)

    # Reference ET from the Penman CSV
    etr = None
    pcsv = os.path.join(OUT, "Fruita_Example_Penman_ET.csv")
    if os.path.exists(pcsv):
        lines = open(pcsv).read().strip().split("\n")
        hdr = lines[0].split(",")
        row = lines[1].split(",")
        etr = float(row[hdr.index("ETr_mm_hr")])

    # dT at anchors from the calibration line. No datum offset: tsd is the raw
    # radiometric surface temperature the model now calibrates against.
    cold_rc = tuple(log["cold_rc"])
    hot_rc = tuple(log["hot_rc"])
    tsd = ts
    a, b = log["a"], log["b"]
    dt_cold = a + b * tsd[cold_rc]
    dt_hot = a + b * tsd[hot_rc]

    # Sensitivity sweep — ET stats from CSV, a/b recovered from rasters
    scsv = os.path.join(OUT, "Fruita_Example_sensitivity.csv")
    strat_rows = {}
    if os.path.exists(scsv):
        lines = open(scsv).read().strip().split("\n")
        hdr = lines[0].split(",")
        for ln in lines[1:]:
            vals = ln.split(",")
            d = dict(zip(hdr, vals))
            label = d["Strategy"]
            strat_rows[label] = dict(
                et_mean=float(d["ET_mean_mm_hr"]),
                et_median=float(d["ET_median_mm_hr"]),
                et_std=float(d["ET_std_mm_hr"]),
                stem=os.path.splitext(d["Output_file"])[0])

    # Recover a/b and read anchor Ts/VI per strategy
    for label, d in strat_rows.items():
        try:
            sa, sb, sr2 = recover_ab(d["stem"], tsd, rho_cp)
            d["a"], d["b"], d["ab_r2"] = sa, sb, sr2
        except Exception as exc:  # pragma: no cover
            d["a"] = d["b"] = d["ab_r2"] = None
            d["ab_error"] = str(exc)

    et_means = [d["et_mean"] for d in strat_rows.values()]

    M = {
        "_provenance": {
            "source": "docs/export_metrics.py",
            "reads": "Batch_Data/Fruita_Example/Output + Input + config.txt",
            "note": ("All worked-example numbers in the monograph derive from "
                     "this file. Regenerate with: python docs/export_metrics.py"),
        },
        "site": {
            "doy": int(cfg.get("DOY", 0)),
            "time": float(cfg.get("time", 0)),
            "T_A_K": T_A, "ea_mb": ea, "p_mb": p, "alt_m": alt,
            "rho": rho, "cp": cp, "rho_cp": rho_cp, "gamma_w": gamma_w,
            "n_pixels": int(np.sum(np.isfinite(ts))),
            "pixel_m": 0.25, "nx": int(ts.shape[1]), "ny": int(ts.shape[0]),
        },
        "cal": {
            "a": a, "b": b,
            "dt_cold": float(dt_cold), "dt_hot": float(dt_hot),
            "cold_rc": list(cold_rc), "hot_rc": list(hot_rc),
            "cold_ts_datum": float(tsd[cold_rc]),
            "hot_ts_datum": float(tsd[hot_rc]),
            "cold_ndvi": float(ndvi[cold_rc]), "hot_ndvi": float(ndvi[hot_rc]),
            "cold_vi_log": log.get("cold_vi"), "hot_vi_log": log.get("hot_vi"),
            "lst_range": log.get("lst_range"),
            "le_cold_precap": log.get("le_cold_precap"),
            "le_cold_postcap": log.get("le_cold_postcap"),
            "rn_minus_g_cold": log.get("rn_minus_g_cold"),
            "h_cold_floor": log.get("h_cold_floor"),
        },
        "flux": {
            "Rn": stats(Rn), "H": stats(H), "LE": stats(LE), "G": stats(G),
            "ET": stats(et_unfloored), "fETr": stats(fetr),
            "rah": stats(rah),
            "ET_floored_mean": float(np.nanmean(et_floored)),
            "neg_LE_pct": float(100.0 * np.mean(LE[np.isfinite(LE)] < 0)),
            "neg_ET_pct": float(100.0 * np.mean(
                et_unfloored[np.isfinite(et_unfloored)] < 0)),
        },
        "etr": {"value": etr,
                "et_over_etr": (float(np.nanmean(et_floored) / etr)
                                if etr else None)},
        "anchors": {
            "cold": dict(rc=list(cold_rc), ts=float(ts[cold_rc]),
                         ndvi=float(ndvi[cold_rc]), lai=float(lai[cold_rc]),
                         Rn=float(Rn[cold_rc]), G=float(G[cold_rc]),
                         H=float(H[cold_rc]), LE=float(LE[cold_rc]),
                         rah=float(rah[cold_rc])),
            "hot": dict(rc=list(hot_rc), ts=float(ts[hot_rc]),
                        ndvi=float(ndvi[hot_rc]), lai=float(lai[hot_rc]),
                        Rn=float(Rn[hot_rc]), G=float(G[hot_rc]),
                        H=float(H[hot_rc]), LE=float(LE[hot_rc]),
                        rah=float(rah[hot_rc])),
        },
        "sensitivity": {
            "strategies": strat_rows,
            "et_min": float(min(et_means)) if et_means else None,
            "et_max": float(max(et_means)) if et_means else None,
            "et_spread_pct": (float(100.0 * (max(et_means) - min(et_means))
                                    / max(et_means)) if et_means else None),
        },
    }

    with open(DEST, "w") as f:
        json.dump(M, f, indent=2)
    print("Wrote", DEST)
    print("  calibration: a=%.4f b=%.5f  (dT_cold=%.3f dT_hot=%.3f)"
          % (a, b, dt_cold, dt_hot))
    print("  mean ET=%.4f  ET/ETr=%.4f  neg-LE=%.2f%%"
          % (M["flux"]["ET_floored_mean"], M["etr"]["et_over_etr"],
             M["flux"]["neg_LE_pct"]))
    print("  sensitivity ET means:",
          {k: v["et_mean"] for k, v in strat_rows.items()})


if __name__ == "__main__":
    main()
