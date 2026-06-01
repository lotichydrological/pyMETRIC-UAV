"""Generate all figures for ``pyMETRIC_UAV_overview.qmd`` from the Fruita
example dataset. Re-run after re-processing Fruita to refresh the document's
figures.

Usage (from the project root):

    python docs/generate_figures.py

Outputs land in ``docs/figures/`` as PDF + PNG (PDF for LaTeX, PNG for HTML).
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from osgeo import gdal, ogr, osr

# Repeatable, vector-friendly defaults
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
FRUITA = os.path.join(ROOT, 'Batch_Data', 'Fruita_Example')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_band(path, band=1):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    arr = ds.GetRasterBand(band).ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    prj = ds.GetProjection()
    ds = None
    return arr, gt, prj


def extent_from_gt(arr, gt):
    rows, cols = arr.shape
    x0 = gt[0]
    y0 = gt[3]
    x1 = gt[0] + cols * gt[1]
    y1 = gt[3] + rows * gt[5]
    return (x0, x1, y1, y0)  # (left, right, bottom, top) for imshow


def load_polygons_to_image_coords(gpkg_path, gt, prj):
    """Return list of (label, [list-of-(x,y)-rings]) in raster CRS."""
    ds = ogr.Open(gpkg_path)
    layer = ds.GetLayer(0)

    vec_srs = layer.GetSpatialRef()
    ras_srs = osr.SpatialReference()
    ras_srs.ImportFromWkt(prj)

    transform = None
    if vec_srs is not None:
        vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ras_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not vec_srs.IsSame(ras_srs):
            transform = osr.CoordinateTransformation(vec_srs, ras_srs)

    polys = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        geom = geom.Clone()
        if transform is not None:
            geom.Transform(transform)

        label = None
        for fname in ('zone', 'type', 'plot_id', 'id', 'name'):
            idx = feat.GetFieldIndex(fname)
            if idx >= 0:
                label = feat.GetField(fname)
                break
        if label is None:
            label = str(feat.GetFID())

        # Multi-polygon / polygon
        if geom.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
            rings = [_ring_to_xy(geom.GetGeometryRef(0))]
        elif geom.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
            rings = []
            for i in range(geom.GetGeometryCount()):
                sub = geom.GetGeometryRef(i)
                rings.append(_ring_to_xy(sub.GetGeometryRef(0)))
        else:
            continue
        polys.append((str(label), rings))

    ds = None
    return polys


def _ring_to_xy(ring):
    n = ring.GetPointCount()
    return np.array([[ring.GetX(i), ring.GetY(i)] for i in range(n)])


def save(fig, name):
    pdf = os.path.join(OUT, f'{name}.pdf')
    png = os.path.join(OUT, f'{name}.png')
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f'  wrote {name}.pdf and {name}.png')


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------

print('Loading Fruita data...')
ndvi, gt, prj = read_band(os.path.join(FRUITA, 'Input', 'NDVI.tif'))
lai,  _,  _  = read_band(os.path.join(FRUITA, 'Input', 'LAI_NDVI.tif'))
fc,   _,  _  = read_band(os.path.join(FRUITA, 'Input', 'FC.tif'))
trad, _,  _  = read_band(os.path.join(FRUITA, 'Input', 'TRAD.tif'))

metric_path = os.path.join(FRUITA, 'Output', 'Fruita_Example_METRIC.tif')
rn, _, _ = read_band(metric_path, band=1)
h,  _, _ = read_band(metric_path, band=2)
le, _, _ = read_band(metric_path, band=3)
g,  _, _ = read_band(metric_path, band=4)

anc_path = os.path.join(FRUITA, 'Output', 'Fruita_Example_METRIC_ancillary.tif')
fetr, _, _ = read_band(anc_path, band=9)

et, _, _ = read_band(os.path.join(FRUITA, 'Output', 'Fruita_Example_ET_mm_hour.tif'))

ext = extent_from_gt(ndvi, gt)

ez_polys = load_polygons_to_image_coords(
    os.path.join(FRUITA, 'Input', 'endmember_zones.gpkg'), gt, prj)
plot_polys = load_polygons_to_image_coords(
    os.path.join(FRUITA, 'Input', 'study_plots.gpkg'), gt, prj)

zonal_csv = os.path.join(FRUITA, 'Output', 'Fruita_Example_Zonal_Stats.csv')
zonal_df = pd.read_csv(zonal_csv) if os.path.exists(zonal_csv) else None

sensitivity_csv = os.path.join(FRUITA, 'Output', 'Fruita_Example_sensitivity.csv')
sens_df = pd.read_csv(sensitivity_csv) if os.path.exists(sensitivity_csv) else None

# Calibration coefficients, endmember pixel coords, and met/site values are
# read from the run outputs (the processing log and the dataset config) so the
# figures always reflect the most recent run — nothing about the calibration is
# hardcoded here.

def _parse_run_log(path):
    """Return (a, b, cold_rc, hot_rc) from a Fruita processing log."""
    import re
    txt = open(path).read()
    m_ab = re.search(r'Calibrated dT: a=([-\d.]+), b=([-\d.]+)', txt)
    m_cold = re.search(r'Cold pixel at \((\d+), (\d+)\)', txt)
    m_hot = re.search(r'Hot pixel at \((\d+), (\d+)\)', txt)
    if not (m_ab and m_cold and m_hot):
        raise ValueError('Could not parse calibration/anchors from %s' % path)
    a = float(m_ab.group(1))
    b = float(m_ab.group(2))
    cold_rc = (int(m_cold.group(1)), int(m_cold.group(2)))
    hot_rc = (int(m_hot.group(1)), int(m_hot.group(2)))
    return a, b, cold_rc, hot_rc


def _parse_config(path):
    """Return (T_A_K, ea_mb, p_mb, alt_m) from a dataset config.txt."""
    cfg = {}
    for line in open(path):
        s = line.split('#')[0].strip()
        if '=' in s:
            k, v = s.split('=', 1)
            cfg[k.strip()] = v.strip()
    t_val = float(cfg['T_A1'])
    t_k = t_val + 273.15 if cfg.get('T_A1_units', 'C').upper() == 'C' else t_val
    return t_k, float(cfg['ea']), float(cfg['p']), float(cfg['alt'])


DT_A, DT_B, (COLD_ROW, COLD_COL), (HOT_ROW, HOT_COL) = _parse_run_log(
    os.path.join(FRUITA, 'Output', 'Fruita_Example_log.txt'))
MET_TA_K, MET_EA, MET_P, SITE_ALT = _parse_config(
    os.path.join(FRUITA, 'config.txt'))
print('  calibration from log: dT = %.4f + %.5f*Ts; '
      'cold=(%d,%d) hot=(%d,%d)'
      % (DT_A, DT_B, COLD_ROW, COLD_COL, HOT_ROW, HOT_COL))


def rc_to_xy(row, col):
    x = gt[0] + (col + 0.5) * gt[1]
    y = gt[3] + (row + 0.5) * gt[5]
    return x, y


COLD_XY = rc_to_xy(COLD_ROW, COLD_COL)
HOT_XY  = rc_to_xy(HOT_ROW,  HOT_COL)


# ---------------------------------------------------------------------------
# Figure 1 — METRIC workflow schematic
# ---------------------------------------------------------------------------

def fig_workflow_schematic():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    def box(xy, w, h, txt, fc='#eef5ff', ec='#3a6ea5'):
        ax.add_patch(mpatches.FancyBboxPatch(
            xy, w, h, boxstyle='round,pad=0.04', linewidth=1.0,
            facecolor=fc, edgecolor=ec))
        ax.text(xy[0] + w/2, xy[1] + h/2, txt, ha='center', va='center',
                fontsize=8)

    def arrow(p0, p1):
        ax.annotate('', xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle='->', linewidth=1.0,
                                    color='#555'))

    # Inputs (left)
    box((0.02, 0.78), 0.18, 0.12, 'Multiband UAV image\n(reflectance + thermal)')
    box((0.02, 0.62), 0.18, 0.12, 'Met station\n(T, u, ea, S↓, p)')
    box((0.02, 0.46), 0.22, 0.12, 'Site geom\n(lat, lon, alt, time)')
    box((0.02, 0.30), 0.22, 0.12, 'Endmember zones\n(cold + hot polygons)', fc='#fff5e6', ec='#c47a2a')

    # Derivation (mid-left)
    box((0.30, 0.78), 0.22, 0.12, 'NDVI / LAI / FC / TRAD')

    # METRIC model (center)
    box((0.32, 0.40), 0.32, 0.30,
        'pyMETRIC\n──────\n'
        'R_n parameterisation\n'
        'Endmember search\n'
        'dT = a + b·T_s\n'
        'Monin-Obukhov\n'
        'iteration',
        fc='#e6f4ea', ec='#2c7a3e')

    # Outputs (right)
    box((0.72, 0.74), 0.26, 0.10, 'METRIC.tif\n(R_n, H, LE, G)')
    box((0.72, 0.60), 0.26, 0.10, 'ancillary.tif\n(R_A, L, u*, ETref, fETr)')
    box((0.72, 0.46), 0.26, 0.10, 'ET_mm_hour.tif')
    box((0.72, 0.32), 0.26, 0.10, 'Penman_ET.csv\n(reference ET)')
    box((0.72, 0.18), 0.26, 0.10, 'Zonal_Stats.csv\n(per plot)')

    # Arrows
    arrow((0.24, 0.84), (0.30, 0.84))
    arrow((0.52, 0.84), (0.55, 0.70))         # derived rasters into model
    arrow((0.24, 0.68), (0.32, 0.58))         # met
    arrow((0.24, 0.52), (0.32, 0.50))         # site geom
    arrow((0.24, 0.36), (0.32, 0.45))         # zones
    for dy in (0.79, 0.65, 0.51, 0.37, 0.23):
        arrow((0.64, 0.55), (0.72, dy + 0.05))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    save(fig, 'fig01_workflow')


# ---------------------------------------------------------------------------
# Figure 2 — Derived inputs (NDVI, LAI, FC, TRAD)
# ---------------------------------------------------------------------------

def fig_derived_inputs():
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    axes = axes.ravel()

    panels = [
        ('NDVI',                ndvi, 'RdYlGn', (-0.2, 1.0),  '–'),
        ('LAI (m²/m²)',         lai,  'YlGn',   (0, 6),       'm²/m²'),
        ('Fractional cover FC', fc,   'YlGn',   (0, 1),       '–'),
        ('TRAD (°C)',           trad - 273.15, 'inferno', (10, 60), '°C'),
    ]
    for ax, (title, arr, cmap, (lo, hi), units) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, extent=ext, origin='upper')
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(units)

    save(fig, 'fig02_derived_inputs')


# ---------------------------------------------------------------------------
# Figure 3 — Endmember zones over TRAD with selected pixels
# ---------------------------------------------------------------------------

def fig_endmembers():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(trad - 273.15, cmap='inferno', vmin=10, vmax=60,
                   extent=ext, origin='upper')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Surface temperature (°C)')

    for label, rings in ez_polys:
        for ring in rings:
            color = 'cyan' if label.lower() in ('cold', 'wet') else 'red'
            ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.8)

    ax.plot(*COLD_XY, marker='o', markersize=9, markerfacecolor='cyan',
            markeredgecolor='black', linestyle='None', label='Cold pixel')
    ax.plot(*HOT_XY,  marker='s', markersize=9, markerfacecolor='red',
            markeredgecolor='black', linestyle='None', label='Hot pixel')

    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', framealpha=0.95)
    save(fig, 'fig03_endmembers')


# ---------------------------------------------------------------------------
# Figure 4 — dT calibration line
# ---------------------------------------------------------------------------

def fig_dT_calibration():
    """The dT calibration line — two endmembers anchor a straight line in
    (T_s, dT) space. For these common-elevation UAV scenes the model applies
    no elevation delapse, so the calibration temperature is the raw radiometric
    surface temperature T_s (no datum adjustment).
    """
    # Calibration coefficients from the run log (parsed at module load).
    a, b = DT_A, DT_B

    # No datum adjustment: plot against the raw surface temperature the model
    # actually calibrates against.
    t_cold = float(trad[COLD_ROW, COLD_COL])
    t_hot  = float(trad[HOT_ROW,  HOT_COL])
    dT_cold = a + b * t_cold
    dT_hot  = a + b * t_hot

    ts = np.linspace(np.nanmin(trad), np.nanmax(trad), 100)
    dT_line = a + b * ts

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    valid = np.isfinite(trad)
    dT_pixels = a + b * trad[valid]
    ax.hexbin(trad[valid], dT_pixels, gridsize=30,
              cmap='Greys', mincnt=1, alpha=0.5)

    ax.plot(ts, dT_line, '-', color='#2c7a3e', linewidth=2.0,
            label=f'dT = {a:.2f} + {b:.4f}·T$_s$')

    ax.plot(t_cold, dT_cold, 'o', markersize=10, markerfacecolor='cyan',
            markeredgecolor='black',
            label=f'Cold pixel ({t_cold:.1f} K)')
    ax.plot(t_hot, dT_hot, 's', markersize=10, markerfacecolor='red',
            markeredgecolor='black',
            label=f'Hot pixel ({t_hot:.1f} K)')

    ax.set_xlabel('Surface temperature, T$_s$ (K)')
    ax.set_ylabel('Air-surface gradient, dT (K)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, 'fig04_dT_calibration')


# ---------------------------------------------------------------------------
# Figure 5 — Energy balance components
# ---------------------------------------------------------------------------

def fig_energy_balance():
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    axes = axes.ravel()

    panels = [
        ('Net radiation R$_n$',        rn,  'viridis', (100, 500)),
        ('Sensible heat H',            h,   'plasma',  (0, 250)),
        ('Latent heat LE',             le,  'YlGnBu',  (0, 600)),
        ('Soil heat flux G',           g,   'cividis', (0, 20)),
    ]
    for ax, (title, arr, cmap, (lo, hi)) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, extent=ext, origin='upper')
        ax.set_title(title + ' (W m$^{-2}$)')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    save(fig, 'fig05_energy_balance')


# ---------------------------------------------------------------------------
# Figure 6 — ET map
# ---------------------------------------------------------------------------

def fig_et_map():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(et, cmap='YlGnBu', vmin=0, vmax=1.0,
                   extent=ext, origin='upper')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Instantaneous ET (mm hr$^{-1}$)')

    # Overlay plot polygons
    for label, rings in plot_polys:
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], color='red', linewidth=1.2, alpha=1.0)

    ax.set_xticks([]); ax.set_yticks([])
    save(fig, 'fig06_et_map')


# ---------------------------------------------------------------------------
# Figure 7 — fETr map
# ---------------------------------------------------------------------------

def fig_fetr_map():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(fetr, cmap='YlGnBu', vmin=0, vmax=1.05,
                   extent=ext, origin='upper')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('$fET_r$')

        # Overlay plot polygons
    for label, rings in plot_polys:
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], color='red', linewidth=1.2, alpha=1.0)

    ax.set_xticks([]); ax.set_yticks([])
    save(fig, 'fig07_fetr_map')


# ---------------------------------------------------------------------------
# Figure 8 — Histogram of pixel-wise ET
# ---------------------------------------------------------------------------

def fig_et_histogram():
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    vals = et[np.isfinite(et)]
    ax.hist(vals, bins=80, color='#3a6ea5', edgecolor='white', linewidth=0.3)
    mean_et = float(np.nanmean(et))
    ax.axvline(mean_et, color='crimson', linewidth=1.8,
               label=f'mean = {mean_et:.3f} mm hr$^{{-1}}$')
    ax.set_xlabel('Instantaneous ET (mm hr$^{-1}$)')
    ax.set_ylabel('Pixel count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, 'fig08_et_histogram')


# ---------------------------------------------------------------------------
# Figure 9 — Sensitivity analysis bar chart
# ---------------------------------------------------------------------------

def fig_sensitivity():
    if sens_df is None or len(sens_df) == 0:
        print('  skipping sensitivity figure (no CSV)')
        return
    df = sens_df.copy()
    df = df.sort_values('Strategy')

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ['#3a6ea5' if 'zone' in s else '#c47a2a' for s in df['Strategy']]
    x = np.arange(len(df))
    ax.bar(x, df['ET_mean_mm_hr'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Strategy'], rotation=30, ha='right')
    ax.set_ylabel('Mean ET (mm hr$^{-1}$)')
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    zone_patch = mpatches.Patch(color='#3a6ea5', label='Zone-restricted search')
    auto_patch = mpatches.Patch(color='#c47a2a', label='Auto search (full image)')
    ax.legend(handles=[zone_patch, auto_patch], loc='upper right', fontsize=8)
    save(fig, 'fig09_sensitivity')


# ---------------------------------------------------------------------------
# Figure 10 — Zonal stats per plot
# ---------------------------------------------------------------------------

def fig_zonal_stats():
    if zonal_df is None or len(zonal_df) == 0:
        print('  skipping zonal-stats figure (no CSV)')
        return
    df = zonal_df.copy()
    if 'Variable' in df.columns:
        df = df[df['Variable'] == 'ET']
    df = df.sort_values('ID')

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(df))
    ax.bar(x, df['Mean'], yerr=df['Std'],
           color='#3a6ea5', edgecolor='black', linewidth=0.4,
           error_kw={'linewidth': 0.8, 'ecolor': '#222'})
    ax.set_xticks(x)
    ax.set_xticklabels(df['ID'].astype(str), rotation=0, fontsize=7)
    ax.set_xlabel('Plot ID')
    ax.set_ylabel('Mean ET (mm hr$^{-1}$) ± 1 SD')
    ax.grid(True, axis='y', alpha=0.3)
    save(fig, 'fig10_zonal_stats')


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Generating figures to', OUT)
    fig_workflow_schematic()
    fig_derived_inputs()
    fig_endmembers()
    fig_dT_calibration()
    fig_energy_balance()
    fig_et_map()
    fig_fetr_map()
    fig_et_histogram()
    fig_sensitivity()
    fig_zonal_stats()
    print('Done.')
