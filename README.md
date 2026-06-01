# pyMETRIC-UAV

Batch processing workflow for estimating evapotranspiration (ET) from UAV multispectral and thermal imagery using the METRIC (Mapping EvapoTranspiration at high Resolution with Internalized Calibration) energy balance model.

Based on [hectornieto/pyMETRIC](https://github.com/hectornieto/pyMETRIC), with modifications for UAV-scale imagery including zone-based endmember search, prescribed calibration coefficients, resolution-adaptive spatial filtering, and streamlined batch I/O.

---

## Directory Structure

```
pyMETRIC-uav/
  pyMETRIC/                      Modified pyMETRIC package source
    __init__.py
    METRIC.py                    Core METRIC energy balance algorithm
    PyMETRIC.py                  Image processing wrapper
    endmember_search.py          Hot/cold pixel selection algorithms
    METRICConfigFileInterface.py Config parser with file auto-discovery
    batch_helpers.py             Notebook-side helpers (preflight, derivation,
                                 sanitization, ET conversion, zonal stats, etc.)
  pyMETRIC_Batch_Run.ipynb       Jupyter notebook for batch processing
  config_template.txt            Annotated configuration file template
  METRIC_local_image_main.py     Command-line entry point (single image)
  Batch_Data/                    Active datasets for processing
  Processed_Data/                Archive of completed runs
  all_sample_plots.gpkg          Polygon boundaries for zonal statistics (user-provided)
  README.md                      This file
```

Each dataset lives in its own folder inside `Batch_Data/`. The default workflow takes a multiband aerial image and derives NDVI, LAI, and surface temperature automatically:

```
Batch_Data/
  Fruita_Example/
    config.txt                   Configuration file (site + met + band assignments)
    example_image.tif            Multiband aerial image (reflectance + thermal)
    Input/                       Auto-created by the derivation step
      TRAD.tif                   Derived: surface temperature (Kelvin)
      NDVI.tif                   Derived: vegetation index
      LAI_NDVI.tif               Derived: leaf area index (Beer's law)
      FC.tif                     Derived: fractional vegetation cover
    Output/                      Auto-created by the model
      Fruita_Example_METRIC.tif
      Fruita_Example_METRIC_ancillary.tif
      Fruita_Example_ET_mm_hour.tif
```

If `multiband_image` is not set in `config.txt`, the workflow falls back to expecting pre-computed TRAD, NDVI, and LAI rasters in `Input/`. All must share the same spatial extent, resolution, and coordinate reference system.

| File | Variable | Units | Description |
|------|----------|-------|-------------|
| `TRAD.tif` | T_R1 | Kelvin | Radiometric land surface temperature. Must be atmospherically corrected and converted from raw thermal camera output to absolute temperature. |
| `NDVI.tif` | VI | unitless (-1 to 1) | Normalized Difference Vegetation Index derived from multispectral bands. |
| `LAI_NDVI.tif` | LAI | m2/m2 | Leaf Area Index. Can be derived from NDVI using an empirical relationship or provided from other sources. |
| `FC.tif` (optional) | f_c | unitless (0-1) | Fractional vegetation cover. If present, used as a per-pixel input to `res.calc_roughness`. If absent, the scalar `f_c` in `config.txt` is broadcast uniformly. |

**File discovery:** If `T_R1`, `VI`, `LAI`, or `f_c` are not explicitly set in `config.txt`, the model auto-discovers them from `Input/` by standard filenames. Alternate names accepted: `T_R1.tif`, `LST.tif` (for temperature); `VI.tif` (for vegetation index); `LAI.tif` (for leaf area index); `fc.tif` (for fractional cover).

---

## Configuration File

Each dataset requires a `config.txt` file in its folder root. Copy `config_template.txt` as a starting point.

The file is organised into three sections ordered by edit frequency:

| Section | Purpose | Typical edit cadence |
|---------|---------|----------------------|
| 1. FLIGHT INFO | Everything specific to this dataset: input image + sensor band assignments (1.1), date/time (1.2), met values (1.3), polygon files (1.4), endmember-search settings (1.5) | every flight |
| 2. SITE INFO | lat, lon, alt, stdlon, instrument heights, `crop_type`, `reference_type` | each new site |
| 3. ADVANCED SETTINGS | Crop presets (3.1), site-level canopy/leaf defaults (3.2), soil heat flux (3.3), QC (3.4), soil optics (3.5), aerodynamic resistance (3.6), optional file overrides (3.7), hot-pixel ETrF tuning (3.8), alternate endmember specifications for manual/prescribed modes (3.9) | rarely |

---

## Input Multiband Aerial Image

The default input is a single multiband GeoTIFF containing reflectance and thermal bands from a UAV flight. The `config.txt` file is used to specify which bands to use:

| Config Key | Description | Example |
|------------|-------------|---------|
| `multiband_image` | Filename of the multiband image (in dataset folder) | `example_image.tif` |
| `red_band` | Band number for Red reflectance (~668 nm) | `3` |
| `nir_band` | Band number for NIR reflectance (~840 nm) | `5` |
| `thermal_band` | Band number for surface temperature | `6` |
| `thermal_units` | Temperature units in thermal band: `C` or `K` | `C` |

The notebook's derivation step computes:
- **NDVI** = (NIR - Red) / (NIR + Red)
- **LAI** via Beer's law inversion (see below)
- **TRAD** = thermal band converted to Kelvin

### LAI from NDVI (Beer's Law)

Fractional vegetation cover and LAI are derived from NDVI:

```
fc = ((NDVI - NDVI_soil) / (NDVI_full - NDVI_soil))^2
LAI = -ln(1 - fc) / k_ext
```

Both rasters are written to `Input/`: `LAI_NDVI.tif` and `FC.tif`. The model auto-discovers FC.tif and uses it as a per-pixel `f_c` input to `res.calc_roughness`, so roughness/displacement vary by canopy density rather than being broadcast from a single scalar.

| Config Key | Default | Description |
|------------|---------|-------------|
| `NDVI_soil` | 0.15 | NDVI of bare soil |
| `NDVI_full` | 0.90 | NDVI of full vegetation cover |
| `k_ext` | 0.6 | Canopy extinction coefficient (0.5-0.65 for alfalfa/grass hay) |
| `LAI_max` | 8.0 | Maximum LAI clamp value |

---

## Output Files

### Primary Output (`*_METRIC.tif`)

4-band GeoTIFF containing the instantaneous surface energy balance components. All bands are in **W/m2**.

| Band | Variable | Units | Description |
|------|----------|-------|-------------|
| 1 | R_n | W/m2 | Net radiation at the surface |
| 2 | H | W/m2 | Sensible heat flux |
| 3 | LE | W/m2 | Latent heat flux (energy equivalent of ET) |
| 4 | G | W/m2 | Soil heat flux |

The energy balance closure is: **R_n = H + LE + G**

### Ancillary Output (`*_METRIC_ancillary.tif`)

9-band GeoTIFF with diagnostic and intermediate variables.

| Band | Variable | Units | Description |
|------|----------|-------|-------------|
| 1 | R_ns | W/m2 | Net shortwave radiation |
| 2 | R_nl | W/m2 | Net longwave radiation |
| 3 | R_A | s/m | Aerodynamic resistance to heat transport |
| 4 | L | m | Monin-Obukhov stability length |
| 5 | u* | m/s | Friction velocity |
| 6 | flag | -- | Quality flag (0 = valid, 255 = invalid) |
| 7 | ETref_datum | W/m2 | Reference ET at datum elevation (surface set by `reference_type`) |
| 8 | ETref | W/m2 | Reference ET at measurement elevation (surface set by `reference_type`) |
| 9 | fETr | -- | ET fraction (LE / ET0; unitless, typically 0 to 1.05). Surface matches `reference_type`. |

### Converted ET Output (`*_ET_mm_hour.tif`)

Single-band GeoTIFF produced by the notebook's post-processing step.

| Band | Variable | Units | Description |
|------|----------|-------|-------------|
| 1 | ET | mm/hr | Instantaneous evapotranspiration rate |

Conversion from LE (W/m2) uses the temperature-dependent latent heat of vaporization:

```
lambda = (2.501 - 0.002361 * T_celsius) * 1e6   [J/kg]
ET_mm_hr = (LE * 3600) / lambda
```

---

## Batch Image Processing

The code includes a Jupyter Notebook that allows users to execute batch processing of multiple datasets from different dates and/or locations. 
To implement a batch run, open `pyMETRIC_Batch_Run.ipynb` in Jupyter Lab and run the code blocks in order. The workflow scans `Batch_Data/` for dataset folders and reports which are ready vs. which have issues. Then runs `sanity_checks` on each config — prints the calendar date implied by the user-entered DOY+time, warns if  `stdlon` looks far from your `lon`, and flags out-of-range met values. For each ready dataset: the workflow then derives NDVI/LAI/FC/TRAD from the multiband image,  sanitizes the output, runs the METRIC model, and computes zonal stats for delineated study plots. All outputs and a per-dataset processing log are written to the `Output/` directory.

---

## Single Image Processing

To run a single image from the command line, set your working directory to the location of the pyMETRIC-UAV code and enter the command below:

```bash
python METRIC_local_image_main.py /path/to/dataset/config.txt
```

---

## Installation and dependencies

The simplest setup is the bundled conda environment file, which pins every
package to the versions this code was developed and validated against. From the
project folder, run once:

```bash
conda env create -f environment.yml
conda activate pymetric-uav
jupyter lab        # then open pyMETRIC_Batch_Run.ipynb
```

The runtime dependencies (all provided by `environment.yml`) are:

- Python 3.10
- GDAL 3.11 with Python bindings (`osgeo`), from **conda-forge** (not pip)
- numpy, scipy, pandas, netCDF4, tqdm
- matplotlib (for the documentation figure scripts)
- [pyTSEB](https://github.com/hectornieto/pyTSEB) 2.1 (pip; the energy-balance
  and resistance routines `pyMETRIC` builds on)

> The validation harness under `validation/` additionally uses `drigo` and
> `refet`; these are not needed for the notebook workflow and are not in
> `environment.yml`.

Do **not** install the `pyMETRIC` package itself. The notebook's first cell
loads the bundled `pyMETRIC/` source directly (it places this folder first on
the import path), so the notebook always runs the code that ships in this
repository — no copying into site-packages, and any other `pyMETRIC` that
happens to be installed in the environment is ignored. If the layout is ever
disturbed, the Setup cell stops with a clear error rather than silently running
stale code.

A step-by-step guide of environment creation through
a worked example is available in `docs/pyMETRIC_UAV_getting_started.qmd`.

---

## References

- Allen, R.G., Tasumi, M., Trezza, R. (2007). Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC) — Model. *Journal of Irrigation and Drainage Engineering*, 133(4), 380-394.
- Allen, R.G., et al. (2013). Automated calibration of the METRIC-Landsat evapotranspiration process. *JAWRA*, 49(3), 563-576.
- Bhattarai, N., et al. (2017). A new optimized algorithm for automating endmember pixel selection in the SEBAL and METRIC models. *Remote Sensing of Environment*, 196, 178-192.
- Nieto, H., et al. pyTSEB/pyMETRIC. [github.com/hectornieto/pyMETRIC](https://github.com/hectornieto/pyMETRIC)
