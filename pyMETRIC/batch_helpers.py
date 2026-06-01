# -*- coding: utf-8 -*-
"""
pyMETRIC batch helpers
======================

All the supporting machinery used by ``pyMETRIC_Batch_Run.ipynb``:
preflight checks, multiband-image derivation, Sanitization, post-processing,
reference ET, zonal stats, diagnostics, sanity warnings, and a final summary
table. Keeping these out of the notebook makes the notebook short and
readable; a user editing it sees the workflow, not the plumbing.
"""

import datetime
import glob
import io
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from pyMETRIC.METRICConfigFileInterface import apply_crop_preset


# Project root = the folder that contains the pyMETRIC/ package (this file is
# pyMETRIC/batch_helpers.py). Used to rewrite absolute paths in the saved log
# as project-root-relative, so logs are portable and not machine-specific.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _relativize_paths(text, root=PROJECT_ROOT):
    """Rewrite any absolute paths under `root` as project-root-relative.

    Replaces occurrences of ``<root>/`` with ``''`` so e.g.
    ``/abs/.../pyMETRIC-uav/Batch_Data/x/Input/f.gpkg`` becomes
    ``Batch_Data/x/Input/f.gpkg``. Only affects strings written to the log
    file; console output is untouched. Anything outside `root` is left as-is.
    """
    if not root:
        return text
    return text.replace(root + os.sep, '')


# ============================================================================
# Logging
# ============================================================================

class DatasetLogger:
    """Capture print output for a dataset and write to ``Output/<name>_log.txt``.

    Usage::

        logger = DatasetLogger()
        with logger.capture('Derivation'):
            print('computing NDVI...')
        logger.save(output_dir, folder_name)
    """

    def __init__(self):
        self.sections = []

    @contextmanager
    def capture(self, section_name):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = _TeeWriter(old_stdout, buf)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            text = buf.getvalue()
            if text.strip():
                self.sections.append((section_name, text))

    def save(self, output_dir, folder_name):
        if not self.sections:
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_path = os.path.join(output_dir, folder_name + '_log.txt')
        with open(log_path, 'w') as f:
            for section_name, text in self.sections:
                # Rewrite absolute paths under the project root as relative,
                # so the saved log is portable and not machine-specific.
                text = _relativize_paths(text)
                f.write('=' * 60 + '\n')
                f.write(section_name + '\n')
                f.write('=' * 60 + '\n')
                f.write(text)
                if not text.endswith('\n'):
                    f.write('\n')
                f.write('\n')
        print(f'  Log saved: {log_path}')

    def clear(self):
        self.sections = []


class _TeeWriter:
    """Writes to two streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================================
# Config parsing + preflight
# ============================================================================

def parse_config_simple(config_path):
    """Read a config.txt into a flat ``{key: value}`` dict.

    Applies the active crop preset (silently) so callers see the resolved
    parameter set --- e.g. ``alfalfa.k_ext`` becomes ``k_ext`` when
    ``crop_type=alfalfa``.
    """
    params = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('#')[0].split('=', 1)
            params[key.strip()] = val.strip()
    apply_crop_preset(params, verbose=False)
    return params


def resolve_config_path(cfg, key, folder_path):
    """Resolve a file path from config, relative to the dataset folder."""
    val = cfg.get(key, '').strip().strip('"')
    if not val:
        return None
    if os.path.isabs(val):
        return val
    return os.path.abspath(os.path.join(folder_path, val))


def run_preflight_check(batch_dir):
    """Scan ``batch_dir`` for dataset folders and report readiness.

    Returns a list of ``(name, folder_path, config_path)`` for datasets that
    pass the basic file-existence and band-range checks.
    """
    folders = sorted([d for d in os.listdir(batch_dir)
                      if os.path.isdir(os.path.join(batch_dir, d))
                      and not d.startswith('.')])

    if not folders:
        print(f'No dataset folders found in {batch_dir}')
        return []

    ready = []
    for folder in folders:
        folder_path = os.path.join(batch_dir, folder)
        issues = []

        config_path = os.path.join(folder_path, 'config.txt')
        if not os.path.exists(config_path):
            txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
            config_candidates = [f for f in txt_files
                                 if 'config' in os.path.basename(f).lower()]
            if config_candidates:
                config_path = config_candidates[0]
            elif txt_files:
                config_path = txt_files[0]
            else:
                issues.append('No config.txt or .txt config file found')
                config_path = None

        if config_path is None:
            print(f'  [{folder}] ISSUES:')
            for issue in issues:
                print(f'    - {issue}')
            continue

        cfg = parse_config_simple(config_path)
        multiband_image = cfg.get('multiband_image', '').strip()

        if multiband_image:
            img_path = os.path.join(folder_path, multiband_image)
            if not os.path.exists(img_path):
                issues.append(f'Multiband image not found: {multiband_image}')
            else:
                ds = gdal.Open(img_path, gdal.GA_ReadOnly)
                if ds is None:
                    issues.append(f'Cannot open multiband image: {multiband_image}')
                else:
                    n_bands = ds.RasterCount
                    ds = None
                    for key, label in [('red_band', 'Red'), ('nir_band', 'NIR'),
                                       ('thermal_band', 'Thermal')]:
                        band_str = cfg.get(key, '').strip()
                        if not band_str:
                            issues.append(f'{label} band number ({key}) not set in config')
                        else:
                            band_num = int(band_str)
                            if band_num < 1 or band_num > n_bands:
                                issues.append(
                                    f'{label} band {band_num} out of range '
                                    f'(image has {n_bands} bands)')
            mode_label = f'multiband: {multiband_image}'
        else:
            input_dir = os.path.join(folder_path, 'Input')
            if not os.path.isdir(input_dir):
                issues.append('No Input/ subdirectory and no multiband_image configured')
            else:
                required = {'TRAD': ['TRAD.tif', 'T_R1.tif', 'LST.tif'],
                            'NDVI': ['NDVI.tif', 'VI.tif'],
                            'LAI':  ['LAI_NDVI.tif', 'LAI.tif']}
                input_files = os.listdir(input_dir)
                for name, patterns in required.items():
                    found = any(p in input_files for p in patterns)
                    if not found:
                        issues.append(
                            f'Missing {name} file ({" or ".join(patterns)})')
            mode_label = 'legacy (pre-computed NDVI/LAI/TRAD)'

        if issues:
            print(f'  [{folder}] ISSUES:')
            for issue in issues:
                print(f'    - {issue}')
        else:
            print(f'  [{folder}] READY — {mode_label} '
                  f'(config: {os.path.basename(config_path)})')
            ready.append((folder, folder_path, config_path))

    print(f'\n{len(ready)}/{len(folders)} datasets ready for processing.')
    return ready


# ============================================================================
# Sanity checks (human-readable warnings shown during preflight)
# ============================================================================

def _doy_time_to_datetime(doy, time_decimal, year=None):
    """Convert (DOY, decimal hour) to a calendar datetime."""
    if year is None:
        year = datetime.datetime.now().year
    base = datetime.datetime(year, 1, 1)
    when = base + datetime.timedelta(days=doy - 1, hours=time_decimal)
    return when


def sanity_checks(name, cfg):
    """Print plain-English checks on a parsed config so the user catches typos.

    Doesn't return anything — just emits warnings to stdout. Designed to run
    inside the preflight cell after a dataset is marked READY.
    """
    msgs = []

    # --- Date / time interpretation ---
    try:
        doy = int(cfg['DOY'])
        t = float(cfg['time'])
        when = _doy_time_to_datetime(doy, t)
        msgs.append(f"  Acquisition: DOY {doy}, {t:.2f} hr -> "
                    f"{when.strftime('%b %-d at %H:%M')} (local standard time)")
        if doy < 1 or doy > 366:
            msgs.append(f"  WARNING: DOY={doy} outside 1-366")
        if t < 0 or t > 24:
            msgs.append(f"  WARNING: time={t} outside 0-24")
    except (KeyError, ValueError):
        pass

    # --- Standard-longitude sanity (rough US heuristic) ---
    try:
        lon = float(cfg['lon'])
        stdlon = float(cfg['stdlon'])
        # US time zones: -75 Eastern, -90 Central, -105 Mountain, -120 Pacific
        expected = round(lon / 15) * 15
        if abs(stdlon - expected) > 15:
            msgs.append(f"  WARNING: stdlon={stdlon} looks far from lon={lon}. "
                        f"Local time-zone meridian is usually within 15° of "
                        f"the site longitude.")
    except (KeyError, ValueError):
        pass

    # --- Met value plausibility (light bounds, just to catch unit mix-ups) ---
    try:
        T_units = cfg.get('T_A1_units', 'K').strip().upper()
        T = float(cfg['T_A1'])
        if T_units == 'C' and not (-40 <= T <= 60):
            msgs.append(f"  WARNING: T_A1={T}°C unusual for daytime UAV flight")
        elif T_units == 'K' and not (233 <= T <= 333):
            msgs.append(f"  WARNING: T_A1={T} K unusual for daytime UAV flight")
    except (KeyError, ValueError):
        pass

    try:
        u = float(cfg['u'])
        if u < 0 or u > 25:
            msgs.append(f"  WARNING: u={u} m/s outside typical 0-25 m/s")
    except (KeyError, ValueError):
        pass

    try:
        S = float(cfg['S_dn'])
        if S < 0 or S > 1400:
            msgs.append(f"  WARNING: S_dn={S} W/m2 outside physical 0-1400 range")
    except (KeyError, ValueError):
        pass

    try:
        p = float(cfg['p'])
        if p < 500 or p > 1100:
            msgs.append(f"  WARNING: p={p} mbar unusual (mid-altitude sites "
                        f"typically 700-900 mbar, sea level ~1013 mbar)")
    except (KeyError, ValueError):
        pass

    # --- Endmember polygons require zone mode ---
    if cfg.get('endmember_zones', '').strip() and \
       cfg.get('endmember_mode', '').strip().lower() == 'auto':
        msgs.append("  NOTE: endmember_zones set with mode=auto — "
                    "will be auto-promoted to 'zone' at run time")

    # --- Reference type consistency ---
    ref = cfg.get('reference_type', 'tall').strip().lower()
    if ref not in ('tall', 'short'):
        msgs.append(f"  WARNING: reference_type={ref!r} unrecognised; "
                    f"will default to 'tall' (alfalfa)")

    if msgs:
        print(f"  Sanity check [{name}]:")
        for m in msgs:
            print(m)


# ============================================================================
# NDVI calibration from zone polygons
# ============================================================================

def calibrate_ndvi_from_zones(ndvi, folder_path, cfg, geo, prj, dims):
    """Re-derive NDVI_soil and NDVI_full from the hot/cold zone polygons.

    Falls back to config defaults if no polygon file is provided or zones
    are empty. The same polygons later constrain the endmember search inside
    the model — see the dual-purpose note in the config template.
    """
    from pyMETRIC.endmember_search import create_zone_masks_from_polygons

    ndvi_soil_default = float(cfg.get('NDVI_soil', 0.15))
    ndvi_full_default = float(cfg.get('NDVI_full', 0.90))

    ez_path = cfg.get('endmember_zones', '').strip().strip('"')
    if not ez_path:
        return ndvi_soil_default, ndvi_full_default

    if not os.path.isabs(ez_path):
        ez_path = os.path.abspath(os.path.join(folder_path, ez_path))

    if not os.path.exists(ez_path):
        print('  NDVI calibration: zone file not found, using defaults')
        return ndvi_soil_default, ndvi_full_default

    zone_field = cfg.get('endmember_zone_field', 'zone').strip()
    cold_mask, hot_mask = create_zone_masks_from_polygons(
        dims, geo, prj, ez_path, zone_field=zone_field)

    ndvi_soil = ndvi_soil_default
    ndvi_full = ndvi_full_default

    if hot_mask is not None and np.any(hot_mask):
        hot_vals = ndvi[hot_mask & np.isfinite(ndvi)]
        if len(hot_vals) > 0:
            ndvi_soil = float(np.nanmedian(hot_vals))
            print(f'  NDVI_soil calibrated from hot zone: {ndvi_soil:.3f} '
                  f'(median of {len(hot_vals)} pixels, '
                  f'default was {ndvi_soil_default:.3f})')
    else:
        print(f'  NDVI_soil: no hot zone polygons, using default '
              f'{ndvi_soil_default:.3f}')

    if cold_mask is not None and np.any(cold_mask):
        cold_vals = ndvi[cold_mask & np.isfinite(ndvi)]
        if len(cold_vals) > 0:
            ndvi_full = float(np.nanmedian(cold_vals))
            print(f'  NDVI_full calibrated from cold zone: {ndvi_full:.3f} '
                  f'(median of {len(cold_vals)} pixels, '
                  f'default was {ndvi_full_default:.3f})')
    else:
        print(f'  NDVI_full: no cold zone polygons, using default '
              f'{ndvi_full_default:.3f}')

    if ndvi_soil >= ndvi_full:
        print(f'  WARNING: calibrated NDVI_soil ({ndvi_soil:.3f}) >= '
              f'NDVI_full ({ndvi_full:.3f}). Reverting to defaults.')
        return ndvi_soil_default, ndvi_full_default

    return ndvi_soil, ndvi_full


# ============================================================================
# Multiband image derivation: NDVI, LAI, FC, TRAD
# ============================================================================

def derive_inputs_from_multiband(folder_path, config_path):
    """Compute NDVI, LAI, FC, TRAD from a multiband UAV image.

    Returns True on success, False on failure. Skips (returns True) if no
    multiband_image is configured.
    """
    cfg = parse_config_simple(config_path)

    multiband_image = cfg.get('multiband_image', '').strip()
    if not multiband_image:
        print('  No multiband_image configured, skipping derivation')
        return True

    img_path = os.path.join(folder_path, multiband_image)
    ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f'  ERROR: cannot open {img_path}')
        return False

    geo = ds.GetGeoTransform()
    prj = ds.GetProjection()
    rows = ds.RasterYSize
    cols = ds.RasterXSize

    red_band = int(cfg['red_band'])
    nir_band = int(cfg['nir_band'])
    thermal_band = int(cfg['thermal_band'])
    thermal_units = cfg.get('thermal_units', 'C').strip().upper()

    print(f'  Image: {multiband_image} ({ds.RasterCount} bands, '
          f'{cols}x{rows}, {abs(geo[1]):.4f} m)')
    print(f'  Bands: Red={red_band}, NIR={nir_band}, Thermal={thermal_band}')

    nodata = ds.GetRasterBand(1).GetNoDataValue()

    red = ds.GetRasterBand(red_band).ReadAsArray().astype(np.float64)
    nir = ds.GetRasterBand(nir_band).ReadAsArray().astype(np.float64)
    thermal = ds.GetRasterBand(thermal_band).ReadAsArray().astype(np.float64)
    ds = None

    if nodata is not None:
        nodata_mask = (red == nodata) | (nir == nodata) | (thermal == nodata)
        red[nodata_mask] = np.nan
        nir[nodata_mask] = np.nan
        thermal[nodata_mask] = np.nan

    # Reflectance of exactly 0.0 is physically implausible — treat as nodata.
    zero_mask = (red <= 0) | (nir <= 0)
    n_zero = np.sum(zero_mask & np.isfinite(red))
    if n_zero > 0:
        red[zero_mask] = np.nan
        nir[zero_mask] = np.nan
        thermal[zero_mask] = np.nan
        print(f'  Masked {n_zero} pixels with zero/negative reflectance as nodata')

    # NDVI
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    n_valid = np.sum(np.isfinite(ndvi))
    print(f'  NDVI: min={np.nanmin(ndvi):.3f}, max={np.nanmax(ndvi):.3f}, '
          f'mean={np.nanmean(ndvi):.3f} ({n_valid} valid pixels)')

    # LAI + FC via Beer's law, optionally with zone-calibrated NDVI bounds
    ndvi_soil, ndvi_full = calibrate_ndvi_from_zones(
        ndvi, folder_path, cfg, geo, prj, (rows, cols))
    k_ext = float(cfg.get('k_ext', 0.6))
    lai_max = float(cfg.get('LAI_max', 8.0))

    fc = ((ndvi - ndvi_soil) / (ndvi_full - ndvi_soil)) ** 2
    fc = np.clip(fc, 0.001, 0.99)

    lai = -np.log(1.0 - fc) / k_ext
    lai = np.clip(lai, 0.0, lai_max)
    lai[~np.isfinite(ndvi)] = np.nan
    fc[~np.isfinite(ndvi)] = np.nan

    print(f'  LAI: min={np.nanmin(lai):.2f}, max={np.nanmax(lai):.2f}, '
          f'mean={np.nanmean(lai):.2f} '
          f'(NDVI_soil={ndvi_soil}, NDVI_full={ndvi_full}, k={k_ext})')
    print(f'  FC:  min={np.nanmin(fc):.3f}, max={np.nanmax(fc):.3f}, '
          f'mean={np.nanmean(fc):.3f}')

    # TRAD
    if thermal_units == 'C':
        trad = thermal + 273.15
        print('  Thermal: converted C -> K')
    elif thermal_units == 'K':
        trad = thermal.copy()
    else:
        print(f'  WARNING: unknown thermal_units={thermal_units}, '
              'assuming Celsius')
        trad = thermal + 273.15

    trad_valid = trad[np.isfinite(trad)]
    print(f'  TRAD: min={np.nanmin(trad_valid):.2f} K, '
          f'max={np.nanmax(trad_valid):.2f} K, '
          f'mean={np.nanmean(trad_valid):.2f} K')

    input_dir = os.path.join(folder_path, 'Input')
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    def write_band(arr, fname):
        out_path = os.path.join(input_dir, fname)
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
        ds_out.SetGeoTransform(geo)
        ds_out.SetProjection(prj)
        band = ds_out.GetRasterBand(1)
        band.SetNoDataValue(float('nan'))
        band.WriteArray(arr.astype(np.float32))
        band.FlushCache()
        ds_out = None

    write_band(ndvi, 'NDVI.tif')
    write_band(lai, 'LAI_NDVI.tif')
    write_band(fc, 'FC.tif')
    write_band(trad, 'TRAD.tif')

    print('  Wrote NDVI.tif, LAI_NDVI.tif, FC.tif, TRAD.tif to Input/')
    return True


# ============================================================================
# Per-raster Sanitization
# ============================================================================

def sanitize_lai(folder_path, lai_max=8.0):
    """Clamp LAI: negatives -> 0, values > 10 -> lai_max."""
    fpath = os.path.join(folder_path, 'Input', 'LAI_NDVI.tif')
    if not os.path.exists(fpath):
        return
    ds = gdal.Open(fpath, gdal.GA_Update)
    if ds is None:
        return
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    n_neg = np.sum(arr[np.isfinite(arr)] < 0)
    n_high = np.sum(arr[np.isfinite(arr)] > 10)
    if n_neg > 0 or n_high > 0:
        arr[arr < 0] = 0
        arr[arr > 10] = lai_max
        band.WriteArray(arr)
        band.FlushCache()
        print(f'  LAI sanitized: {n_neg} negatives clamped, '
              f'{n_high} high values capped')
    ds = None


def sanitize_temperature(folder_path):
    """Replace temperatures < 200K with NaN."""
    fpath = os.path.join(folder_path, 'Input', 'TRAD.tif')
    if not os.path.exists(fpath):
        return
    ds = gdal.Open(fpath, gdal.GA_Update)
    if ds is None:
        return
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    n_bad = np.sum(arr[np.isfinite(arr)] < 200)
    if n_bad > 0:
        arr[arr < 200] = np.nan
        band.SetNoDataValue(float('nan'))
        band.WriteArray(arr)
        band.FlushCache()
        print(f'  TRAD sanitized: {n_bad} pixels < 200K replaced with NaN')
    ds = None


def sanitize_ndvi(folder_path):
    """Replace NDVI values outside [-1, 1] with NaN."""
    fpath = os.path.join(folder_path, 'Input', 'NDVI.tif')
    if not os.path.exists(fpath):
        return
    ds = gdal.Open(fpath, gdal.GA_Update)
    if ds is None:
        return
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    n_bad = np.sum((arr < -1.0) | (arr > 1.0))
    if n_bad > 0:
        arr[(arr < -1.0) | (arr > 1.0)] = np.nan
        band.WriteArray(arr)
        band.FlushCache()
        print(f'  NDVI sanitized: {n_bad} out-of-range pixels replaced with NaN')
    ds = None


# ============================================================================
# Post-processing
# ============================================================================

def validate_metric_output(folder_path, folder_name):
    """Inspect METRIC output for NaN/zero pathology.

    Returns ``(status, message)`` where status is OK / WARNING / NO_OUTPUT /
    UNREADABLE.
    """
    output_dir = os.path.join(folder_path, 'Output')
    metric_files = glob.glob(os.path.join(output_dir, '*_METRIC.tif'))
    metric_files = [f for f in metric_files
                    if '_ancillary' not in os.path.basename(f)
                    and '_CIMEC_' not in os.path.basename(f)
                    and '_ESA_' not in os.path.basename(f)
                    and '_MinMax_' not in os.path.basename(f)]
    if not metric_files:
        return 'NO_OUTPUT', 'No METRIC output file found'

    ds = gdal.Open(metric_files[0], gdal.GA_ReadOnly)
    if ds is None or ds.RasterCount < 4:
        return 'UNREADABLE', 'Cannot read METRIC output or fewer than 4 bands'

    issues = []
    band_names = ['Rn', 'H', 'LE', 'G']
    for b in range(1, 5):
        arr = ds.GetRasterBand(b).ReadAsArray().astype(np.float64)
        n_total = arr.size
        n_nan = np.sum(~np.isfinite(arr))
        n_zero = np.sum(arr[np.isfinite(arr)] == 0)
        pct_nan = 100.0 * n_nan / n_total
        pct_zero = 100.0 * n_zero / n_total

        if pct_nan > 90:
            issues.append(f'{band_names[b-1]}: {pct_nan:.0f}% NaN')
        elif pct_nan > 50:
            issues.append(f'{band_names[b-1]}: {pct_nan:.0f}% NaN - partial failure')

        if b == 3 and pct_zero > 80:
            issues.append(f'LE: {pct_zero:.0f}% zero')

    ds = None
    if issues:
        return 'WARNING', '; '.join(issues)
    return 'OK', 'Output validates clean'


def convert_le_to_et(folder_path, folder_name):
    """Convert METRIC band 3 (LE in W/m2) to instantaneous ET (mm/hr)."""
    output_dir = os.path.join(folder_path, 'Output')
    if not os.path.isdir(output_dir):
        return None

    metric_files = glob.glob(os.path.join(output_dir, '*_METRIC.tif'))
    metric_files = [f for f in metric_files
                    if '_ancillary' not in os.path.basename(f)
                    and '_CIMEC_' not in os.path.basename(f)
                    and '_ESA_' not in os.path.basename(f)
                    and '_MinMax_' not in os.path.basename(f)]
    if not metric_files:
        return None
    metric_file = metric_files[0]

    trad_file = os.path.join(folder_path, 'Input', 'TRAD.tif')
    if not os.path.exists(trad_file):
        print('  Skipped LE->ET: no TRAD.tif')
        return None

    ds_metric = gdal.Open(metric_file, gdal.GA_ReadOnly)
    if ds_metric is None or ds_metric.RasterCount < 3:
        return None
    le = ds_metric.GetRasterBand(3).ReadAsArray().astype(np.float64)
    geo = ds_metric.GetGeoTransform()
    prj = ds_metric.GetProjection()
    rows, cols = le.shape
    ds_metric = None

    ds_trad = gdal.Open(trad_file, gdal.GA_ReadOnly)
    trad = ds_trad.GetRasterBand(1).ReadAsArray().astype(np.float64)
    ds_trad = None

    if trad.shape != le.shape:
        from scipy.ndimage import zoom
        zoom_factors = (le.shape[0] / trad.shape[0],
                        le.shape[1] / trad.shape[1])
        trad = zoom(trad, zoom_factors, order=1)

    t_celsius = trad - 273.15
    lambda_j = (2.501 - 0.002361 * t_celsius) * 1e6

    et_mm_hr = (le * 3600.0) / lambda_j
    et_mm_hr[et_mm_hr < 0] = 0.0
    et_mm_hr[~np.isfinite(et_mm_hr)] = np.nan

    out_path = os.path.join(output_dir, folder_name + '_ET_mm_hour.tif')
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(geo)
    ds_out.SetProjection(prj)
    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(float('nan'))
    band.WriteArray(et_mm_hr.astype(np.float32))
    band.FlushCache()
    ds_out = None

    mean_et = np.nanmean(et_mm_hr)
    print(f'  ET mean = {mean_et:.4f} mm/hr')
    return {'Dataset': folder_name, 'Mean_ET_mm_hr': mean_et}


# ============================================================================
# Reference ET (ASCE Penman-Monteith)
# ============================================================================

def calc_etr_inst(cfg):
    """ASCE Penman-Monteith instantaneous ET (mm/hr).

    Reference surface picked from ``cfg['reference_type']``:
      ``'tall'``  -> alfalfa (Cn=66, Cd=0.25, G/Rn=0.04). Default.
      ``'short'`` -> grass   (Cn=37, Cd=0.24, G/Rn=0.10).

    Must match the reference_type used by the pyMETRIC model so the
    notebook's Penman_ET.csv is consistent with the model's fETr band.
    """
    ref = cfg.get('reference_type', 'tall').strip().lower()
    if ref == 'short':
        Cn, Cd, g_ratio = 37.0, 0.24, 0.10
    else:
        Cn, Cd, g_ratio = 66.0, 0.25, 0.04
    albedo = 0.23

    T_A1_units = cfg.get('T_A1_units', 'K').strip().upper()
    T_A1_val = float(cfg['T_A1'])
    if T_A1_units == 'C':
        T_C = T_A1_val
        T_K = T_C + 273.15
    else:
        T_K = T_A1_val
        T_C = T_K - 273.15

    u = float(cfg['u'])
    ea = float(cfg['ea']) * 0.1
    P = float(cfg['p']) * 0.1
    Rs = float(cfg['S_dn']) * 3600 / 1e6
    z_u = float(cfg['z_u'])
    alt = float(cfg.get('alt', 0))

    gamma = 0.000665 * P
    es = 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))
    delta = 4098.0 * es / (T_C + 237.3) ** 2

    Rns = (1.0 - albedo) * Rs
    sigma_hr = 2.042e-10
    Rso = (0.75 + 2e-5 * alt) * Rs * 1.2
    if Rs > 0 and Rso > 0:
        fcd = 1.35 * (Rs / Rso) - 0.35
        fcd = max(0.05, min(fcd, 1.0))
    else:
        fcd = 0.05
    Rnl = sigma_hr * fcd * (0.34 - 0.14 * np.sqrt(ea)) * T_K**4

    Rn = Rns - Rnl
    G = g_ratio * Rn
    u2 = u * 4.87 / np.log(67.8 * z_u - 5.42)

    ETr = (0.408 * delta * (Rn - G)
           + gamma * (Cn / (T_C + 273.0)) * u2 * (es - ea)) \
          / (delta + gamma * (1.0 + Cd * u2))

    return max(0.0, ETr)


# ============================================================================
# Zonal statistics
# ============================================================================

def compute_zonal_stats(raster_path, vector_path, dataset_name,
                        value_label='ET', value_unit='mm/hr'):
    """Per-feature mean/median/min/max/std for a raster against a polygon set.

    Returns a list of dicts ready for ``pd.DataFrame``.
    """
    ds_raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds_raster is None:
        print(f'  Cannot open raster: {raster_path}')
        return []

    gt = ds_raster.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds_raster.GetProjection())
    cols = ds_raster.RasterXSize
    rows_r = ds_raster.RasterYSize
    nodata = ds_raster.GetRasterBand(1).GetNoDataValue()

    raster_arr = ds_raster.GetRasterBand(1).ReadAsArray().astype(np.float64)
    if nodata is not None:
        raster_arr[raster_arr == nodata] = np.nan
    ds_raster = None

    pixel_area_m2 = abs(gt[1] * gt[5])
    x_min = gt[0]
    y_max = gt[3]
    x_max = gt[0] + cols * gt[1]
    y_min = gt[3] + rows_r * gt[5]

    ds_vec = ogr.Open(vector_path)
    if ds_vec is None:
        print(f'  Cannot open vector: {vector_path}')
        return []

    layer = ds_vec.GetLayer(0)
    vec_srs = layer.GetSpatialRef()

    transform = None
    if vec_srs is not None and raster_srs is not None:
        vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not vec_srs.IsSame(raster_srs):
            transform = osr.CoordinateTransformation(vec_srs, raster_srs)

    results = []
    n_skipped = 0

    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        feat_id = feat.GetField('id') if feat.GetFieldIndex('id') >= 0 \
            else feat.GetFID()
        plot_id = feat.GetField('plot_id') if feat.GetFieldIndex('plot_id') >= 0 else ''
        location = feat.GetField('location') if feat.GetFieldIndex('location') >= 0 else ''

        base_row = {
            'Dataset': dataset_name, 'ID': feat_id, 'Plot_ID': plot_id,
            'Location': location, 'Variable': value_label, 'Unit': value_unit,
        }

        geom = geom.Clone()
        if transform is not None:
            geom.Transform(transform)

        env = geom.GetEnvelope()
        feat_xmin, feat_xmax, feat_ymin, feat_ymax = env

        if (feat_xmax < x_min or feat_xmin > x_max or
                feat_ymax < y_min or feat_ymin > y_max):
            base_row.update({'Mean': np.nan, 'Median': np.nan, 'Min': np.nan,
                             'Max': np.nan, 'Std': np.nan,
                             'Count': 0, 'Total_Pixels': 0, 'Pct_Valid': 0.0,
                             'Pixel_m2': pixel_area_m2})
            results.append(base_row)
            n_skipped += 1
            continue

        px_min, py_min = gdal.ApplyGeoTransform(inv_gt, feat_xmin, feat_ymax)
        px_max, py_max = gdal.ApplyGeoTransform(inv_gt, feat_xmax, feat_ymin)

        px_min = max(0, int(np.floor(px_min)))
        py_min = max(0, int(np.floor(py_min)))
        px_max = min(cols, int(np.ceil(px_max)))
        py_max = min(rows_r, int(np.ceil(py_max)))

        win_cols = px_max - px_min
        win_rows = py_max - py_min
        if win_cols <= 0 or win_rows <= 0:
            base_row.update({'Mean': np.nan, 'Median': np.nan, 'Min': np.nan,
                             'Max': np.nan, 'Std': np.nan,
                             'Count': 0, 'Total_Pixels': 0, 'Pct_Valid': 0.0,
                             'Pixel_m2': pixel_area_m2})
            results.append(base_row)
            n_skipped += 1
            continue

        mem_drv = gdal.GetDriverByName('MEM')
        mem_ds = mem_drv.Create('', win_cols, win_rows, 1, gdal.GDT_Byte)
        win_gt = (gt[0] + px_min * gt[1], gt[1], gt[2],
                  gt[3] + py_min * gt[5], gt[4], gt[5])
        mem_ds.SetGeoTransform(win_gt)
        mem_ds.SetProjection(raster_srs.ExportToWkt())
        mem_ds.GetRasterBand(1).Fill(0)

        mem_vec_drv = ogr.GetDriverByName('MEM')
        mem_vec_ds = mem_vec_drv.CreateDataSource('')
        mem_layer = mem_vec_ds.CreateLayer('', srs=raster_srs, geom_type=ogr.wkbPolygon)
        mem_feat = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feat.SetGeometry(geom)
        mem_layer.CreateFeature(mem_feat)

        gdal.RasterizeLayer(mem_ds, [1], mem_layer, burn_values=[1])
        mask = mem_ds.GetRasterBand(1).ReadAsArray().astype(bool)

        mem_feat = None
        mem_layer = None
        mem_vec_ds = None
        mem_ds = None

        window = raster_arr[py_min:py_max, px_min:px_max]
        all_pixels = window[mask]
        total_pixels = int(len(all_pixels))
        valid = all_pixels[np.isfinite(all_pixels)]

        if len(valid) == 0:
            base_row.update({'Mean': np.nan, 'Median': np.nan, 'Min': np.nan,
                             'Max': np.nan, 'Std': np.nan,
                             'Count': 0, 'Total_Pixels': total_pixels,
                             'Pct_Valid': 0.0, 'Pixel_m2': pixel_area_m2})
            results.append(base_row)
            n_skipped += 1
            continue

        pct_valid = round(100.0 * len(valid) / total_pixels, 1)
        base_row.update({
            'Mean': round(float(np.mean(valid)), 6),
            'Median': round(float(np.median(valid)), 6),
            'Min': round(float(np.min(valid)), 6),
            'Max': round(float(np.max(valid)), 6),
            'Std': round(float(np.std(valid)), 6),
            'Count': int(len(valid)),
            'Total_Pixels': total_pixels,
            'Pct_Valid': pct_valid,
            'Pixel_m2': pixel_area_m2,
        })
        results.append(base_row)

    ds_vec = None
    if n_skipped > 0:
        print(f'  {n_skipped} features had no valid data')

    return results


# ============================================================================
# Diagnostics
# ============================================================================

def check_image_stats(folder_path):
    """Print stats for the multiband source and each derived input raster."""
    input_dir = os.path.join(folder_path, 'Input')
    folder_name = os.path.basename(folder_path)
    print(f'\n=== {folder_name} ===')

    config_path = os.path.join(folder_path, 'config.txt')
    if os.path.exists(config_path):
        cfg = parse_config_simple(config_path)
        mb_image = cfg.get('multiband_image', '').strip()
        if mb_image:
            mb_path = os.path.join(folder_path, mb_image)
            if os.path.exists(mb_path):
                ds = gdal.Open(mb_path, gdal.GA_ReadOnly)
                if ds:
                    gt = ds.GetGeoTransform()
                    print(f'  Multiband source: {mb_image}')
                    print(f'    Bands: {ds.RasterCount}, '
                          f'Size: {ds.RasterXSize}x{ds.RasterYSize}, '
                          f'Resolution: {abs(gt[1]):.4f} m')
                    ds = None

    for fname, label, expected_range in [
        ('TRAD.tif', 'Surface Temperature', (250, 350)),
        ('NDVI.tif', 'NDVI', (-1, 1)),
        ('LAI_NDVI.tif', 'LAI', (0, 10)),
        ('FC.tif', 'Fractional Cover', (0, 1)),
    ]:
        fpath = os.path.join(input_dir, fname)
        if not os.path.exists(fpath):
            print(f'  {label}: FILE NOT FOUND')
            continue

        ds = gdal.Open(fpath, gdal.GA_ReadOnly)
        if ds is None:
            print(f'  {label}: CANNOT OPEN')
            continue

        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        gt = ds.GetGeoTransform()
        ds = None

        valid = arr[np.isfinite(arr)]
        n_nan = np.sum(~np.isfinite(arr))
        lo, hi = expected_range

        print(f'  {label} ({fname}):')
        print(f'    Shape: {arr.shape}, Resolution: {abs(gt[1]):.4f} m')
        if len(valid) > 0:
            print(f'    Min: {np.min(valid):.2f}, Max: {np.max(valid):.2f}, '
                  f'Mean: {np.mean(valid):.2f}, Std: {np.std(valid):.2f}')
            print(f'    NaN/NoData pixels: {n_nan} '
                  f'({100*n_nan/arr.size:.1f}%)')
            if np.min(valid) < lo or np.max(valid) > hi:
                print(f'    WARNING: values outside expected range [{lo}, {hi}]')
        else:
            print('    All pixels are NaN/NoData')


# ============================================================================
# End-of-run summary table
# ============================================================================

def build_summary_table(batch_results, et_results, datasets):
    """Build a per-dataset summary DataFrame for end-of-run reporting.

    Combines:
      * status (from batch_results dict)
      * mean ET (from et_results list of dicts)
      * reference ETr and ET fraction if Penman_ET.csv exists
      * plot count from Zonal_Stats.csv if present
    """
    rows = []
    for name, folder_path, _ in datasets:
        status = batch_results.get(name, 'NOT RUN')
        output_dir = os.path.join(folder_path, 'Output')

        mean_et = next((r['Mean_ET_mm_hr'] for r in et_results
                        if r['Dataset'] == name), np.nan)

        etr_mm_hr = np.nan
        etr_csv = os.path.join(output_dir, name + '_Penman_ET.csv')
        if os.path.exists(etr_csv):
            try:
                etr_df = pd.read_csv(etr_csv)
                if 'ETr_mm_hr' in etr_df.columns and len(etr_df) > 0:
                    etr_mm_hr = float(etr_df['ETr_mm_hr'].iloc[0])
            except Exception:
                pass

        et_fraction = (mean_et / etr_mm_hr) if (
            np.isfinite(mean_et) and np.isfinite(etr_mm_hr) and etr_mm_hr > 0
        ) else np.nan

        n_plots = 0
        zonal_csv = os.path.join(output_dir, name + '_Zonal_Stats.csv')
        if os.path.exists(zonal_csv):
            try:
                z_df = pd.read_csv(zonal_csv)
                if 'Variable' in z_df.columns:
                    n_plots = int((z_df['Variable'] == 'ET').sum())
                else:
                    n_plots = len(z_df)
            except Exception:
                pass

        rows.append({
            'Dataset': name,
            'Status': status,
            'Mean ET (mm/hr)': round(mean_et, 4) if np.isfinite(mean_et) else None,
            'Ref ETr (mm/hr)': round(etr_mm_hr, 4) if np.isfinite(etr_mm_hr) else None,
            'ET / ETr':        round(et_fraction, 3) if np.isfinite(et_fraction) else None,
            'Plots':           n_plots if n_plots > 0 else None,
        })

    return pd.DataFrame(rows)
