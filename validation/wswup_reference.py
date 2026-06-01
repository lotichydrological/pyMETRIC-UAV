"""
WSWUP/pyMETRIC reference energy-balance solver.

This module is a faithful re-expression of the calibration + stability
solver in WSWUP/pymetric ``code/metric_functions/metric_model2_func.py``.
It does NOT re-implement any physics: every numerical step calls the genuine
WSWUP functions imported from ``code/support/et_numpy.py`` and
``code/support/et_common.py`` (clone pinned at commit c3a47fb,
et_numpy.py sha256 4cbf022c…). This file only reproduces the *control flow*
of metric_model2_func — the two iteration loops, in the same order, with the
same constants — so the WSWUP solver can be driven on an arbitrary in-memory
scene without the full GDAL/INI pipeline.

Loop 1 (pixel calibration), metric_model2_func.py L1067-1156:
    init a=1, b=-1000, psi=0, dt = ts - 293
    repeat 20x:
        u* = u_star_func(u3, z3, zom, psi3)
        rah = rah_func(z, psi2, psi1, u*)
        rho = density_func(elev, ts, dt)
        dt  = dt_calibration_func(h_anchor, rah, rho)      # dt = h*rah/(rho*1004)
        a   = (dt_hot-dt_cold)/(tsdem_hot-tsdem_cold)
        b   = -a*tsdem_cold + dt_cold
        L   = l_calibration_func(h_anchor, rho, u*, ts)
        psi_i = psi_func(L, i, z_i)   for i in {1:0.1m, 2:2m, 3:200m}

Loop 2 (raster application), metric_model2_func.py L1578-1648:
    dt = dt_func(ts_dem, a, b)                              # dt = a*ts_dem + b
    psi3=psi2=psi1 = 0
    repeat 6x:
        u* = u_star_func(u3, z3, zom, psi3)
        rah = rah_func(z, psi2, psi1, u*)
        L   = l_func(dt, u*, ts, rah)
        psi3=psi_func(L,3,z3); psi2=psi_func(L,2,z2); psi1=psi_func(L,1,z1)
    # final
    u* = u_star_func(...); rah = rah_func(...)
    rho = density_func(elev, ts, dt)
    H  = h_func(rho, dt, rah)                               # H = rho*1004*dt/rah
    LE = le_func(rn, g, H)                                  # LE = rn - g - H
    ET_inst = et_inst_func(LE, ts)                          # mm/hr

Wind to the 200 m blending height uses WSWUP's native station log-profile
(et_common.u_star_station_func + u3_func) with station roughness 0.015 m,
exactly as metric_model2_func.py L851-858.
"""

import os
import sys

import numpy as np

# Locate the WSWUP clone's support modules.
_WSWUP_SUPPORT = os.environ.get(
    "WSWUP_SUPPORT",
    os.path.expanduser("~/wswup-validation/code/support"))
if _WSWUP_SUPPORT not in sys.path:
    sys.path.insert(0, _WSWUP_SUPPORT)

import et_numpy   # noqa: E402  genuine WSWUP energy-balance functions
import et_common  # noqa: E402  genuine WSWUP station-wind helpers

# Heights (m) for the resistance profile, metric_model2_func.py L836.
Z_FLT_DICT = {1: 0.1, 2: 2.0, 3: 200.0}

# Station wind defaults, metric_model2_func.py L700-703.
STATION_ROUGHNESS = 0.015
WIND_SPEED_HEIGHT = 2.0

PIXEL_ITERS = 20    # stability_pixel_iters default, L329-330
RASTER_ITERS = 6    # stability_raster_iters default, L339-340


def wind_to_blending_height(u_obs, wind_speed_height=WIND_SPEED_HEIGHT,
                            station_roughness=STATION_ROUGHNESS):
    """Extrapolate observed wind to the 200 m blending height (u3).

    Genuine WSWUP path: u_star_station_func -> u3_func.
    """
    u_star_station = et_common.u_star_station_func(
        wind_speed_height, station_roughness, u_obs)
    return et_common.u3_func(u_star_station, Z_FLT_DICT[3], station_roughness)


def calibrate(ts_anchor, tsdem_anchor, elev_anchor, h_anchor, u3, zom_anchor,
              n_iter=PIXEL_ITERS):
    """Loop 1 — invert the two anchor pixels to dT calibration (a, b).

    Parameters
    ----------
    ts_anchor : (2,) array   [cold, hot] surface temperature (K)
    tsdem_anchor : (2,) array  delapsed surface temperature (K)
    elev_anchor : (2,) array  elevation (m)
    h_anchor : (2,) array   anchor sensible heat = Rn - G - LE (W m-2)
    u3 : float              wind speed at 200 m (m s-1)
    zom_anchor : (2,) array  momentum roughness length (m)

    Returns
    -------
    a, b : float   dt = a*ts_dem + b   (a = slope, b = intercept)
    diag : dict    final-iteration u*, rah, rho, dt, L at the anchors
    """
    ts_anchor = np.asarray(ts_anchor, dtype=np.float64)
    tsdem_anchor = np.asarray(tsdem_anchor, dtype=np.float64)
    elev_anchor = np.asarray(elev_anchor, dtype=np.float64)
    h_anchor = np.asarray(h_anchor, dtype=np.float64)
    zom_anchor = np.asarray(zom_anchor, dtype=np.float64)

    a, b = 1.0, -1000.0
    psi = {1: np.zeros(2), 2: np.zeros(2), 3: np.zeros(2)}
    dt = ts_anchor - 293.0

    for _ in range(n_iter):
        u_star = et_numpy.u_star_func(u3, Z_FLT_DICT[3], zom_anchor, psi[3])
        rah = et_numpy.rah_func(Z_FLT_DICT, psi[2], psi[1], u_star)
        rho = et_numpy.density_func(elev_anchor, ts_anchor, dt)
        dt = et_numpy.dt_calibration_func(h_anchor, rah, rho)
        a = (dt[1] - dt[0]) / (tsdem_anchor[1] - tsdem_anchor[0])
        b = -(a * tsdem_anchor[0]) + dt[0]
        L = et_numpy.l_calibration_func(h_anchor, rho, u_star, ts_anchor)
        for zi, zf in Z_FLT_DICT.items():
            psi[zi] = et_numpy.psi_func(L, zi, zf)

    diag = dict(u_star=np.asarray(u_star), rah=np.asarray(rah),
                rho=np.asarray(rho), dt=np.asarray(dt), L=np.asarray(L))
    return float(a), float(b), diag


def apply_image(ts, tsdem, elev, rn, g, a, b, u3, zom, n_iter=RASTER_ITERS):
    """Loop 2 — apply (a, b) over the image and solve H, LE, ET.

    All arrays are 2-D (or 1-D) and broadcast together. ``rn`` and ``g`` are
    the per-pixel net radiation and soil heat flux (W m-2) — supplied
    externally so both solvers share identical energy inputs.
    """
    ts = np.asarray(ts, dtype=np.float64)
    tsdem = np.asarray(tsdem, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)
    rn = np.asarray(rn, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    zom = np.asarray(zom, dtype=np.float64)

    dt = et_numpy.dt_func(tsdem, a, b)

    psi3 = np.zeros(dt.shape)
    psi2 = np.zeros(dt.shape)
    psi1 = np.zeros(dt.shape)
    for _ in range(n_iter):
        u_star = et_numpy.u_star_func(u3, Z_FLT_DICT[3], zom, psi3)
        rah = et_numpy.rah_func(Z_FLT_DICT, psi2, psi1, u_star)
        # l_func mutates dt in place (zeros -> -1000); WSWUP reuses dt_array
        # across iterations unchanged, so we mirror that exactly.
        L = et_numpy.l_func(dt, u_star, ts, rah)
        psi3 = et_numpy.psi_func(L, 3, Z_FLT_DICT[3])
        psi2 = et_numpy.psi_func(L, 2, Z_FLT_DICT[2])
        psi1 = et_numpy.psi_func(L, 1, Z_FLT_DICT[1])

    u_star = et_numpy.u_star_func(u3, Z_FLT_DICT[3], zom, psi3)
    rah = et_numpy.rah_func(Z_FLT_DICT, psi2, psi1, u_star)
    rho = et_numpy.density_func(elev, ts, dt)
    h = et_numpy.h_func(rho, dt, rah)
    le = et_numpy.le_func(rn, g, h)
    et_inst = et_numpy.et_inst_func(le, ts)

    return dict(dt=np.asarray(dt, dtype=np.float64),
                rah=np.asarray(rah, dtype=np.float64),
                u_star=np.asarray(u_star, dtype=np.float64),
                rho=np.asarray(rho, dtype=np.float64),
                h=np.asarray(h, dtype=np.float64),
                le=np.asarray(le, dtype=np.float64),
                et_inst=np.asarray(et_inst, dtype=np.float64),
                a=a, b=b)


def solve(ts, tsdem, elev, rn, g, anchor_idx, le_anchor, u_obs, zom,
          rn_anchor=None, g_anchor=None):
    """Full WSWUP reference solve: calibrate on anchors, apply over image.

    Parameters
    ----------
    ts, tsdem, elev, rn, g, zom : 2-D arrays (image)
    anchor_idx : (cold_rc, hot_rc) tuples of (row, col)
    le_anchor : (2,) array  anchor latent-heat targets [cold, hot] (W m-2)
    u_obs : float  observed wind speed at 2 m (m s-1)
    rn_anchor, g_anchor : optional explicit anchor Rn/G; default reads rn/g
        at anchor_idx.
    """
    cold_rc, hot_rc = anchor_idx
    u3 = wind_to_blending_height(u_obs)

    ts_a = np.array([ts[cold_rc], ts[hot_rc]], dtype=np.float64)
    tsdem_a = np.array([tsdem[cold_rc], tsdem[hot_rc]], dtype=np.float64)
    elev_a = np.array([_at(elev, cold_rc), _at(elev, hot_rc)], dtype=np.float64)
    zom_a = np.array([zom[cold_rc], zom[hot_rc]], dtype=np.float64)
    if rn_anchor is None:
        rn_anchor = np.array([rn[cold_rc], rn[hot_rc]], dtype=np.float64)
    if g_anchor is None:
        g_anchor = np.array([g[cold_rc], g[hot_rc]], dtype=np.float64)
    h_anchor = rn_anchor - g_anchor - np.asarray(le_anchor, dtype=np.float64)

    a, b, cal_diag = calibrate(ts_a, tsdem_a, elev_a, h_anchor, u3, zom_a)
    out = apply_image(ts, tsdem, elev, rn, g, a, b, u3, zom)
    out["cal_diag"] = cal_diag
    out["h_anchor"] = h_anchor
    out["u3"] = u3
    return out


def _at(arr, rc):
    """Index a 2-D array or return a scalar/0-d as-is."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return float(a)
    return a[rc]
