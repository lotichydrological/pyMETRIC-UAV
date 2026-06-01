# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:56:06 2018

@author: hector

Modified 2026: Added zone-based endmember search, quality validation,
resolution-adaptive CV filtering, and utility functions for UAV imagery.
"""

import os
from functools import lru_cache

import numpy as np
from scipy.signal import convolve2d

VI_SOIL = 0.0
VI_FULL = 0.95

# Minimum recommended LST range between endmembers (K)
DEFAULT_MIN_LST_RANGE = 3.0


def maxmin_temperature(vi_array, lst_array, vi_lower_limit=VI_SOIL,
                       cold_percentile=2.0, hot_percentile=98.0,
                       search_mask=None, cold_zone=None, hot_zone=None):
    """Pick cold and hot pixels by LST percentile, with VI as a tie-breaker.

    Originally this picked the absolute coldest pixel in the search area for
    cold, and the absolute hottest for hot. That collapses on UAV scenes
    because raw thermal imagery often contains a handful of artifact pixels
    (specular reflections, sensor glitches, cold-edge effects) tens of K
    away from the real surface distribution — and the model dutifully
    calibrates dT against them.

    The robust version picks the coldest ``cold_percentile`` fraction of
    vegetated pixels and, among those candidates, returns the one with the
    highest VI (most fully-vegetated). Symmetric for hot. In zone mode the
    polygon already constrains the search, so results shift very little; in
    auto mode it eliminates the collapse.

    Parameters
    ----------
    vi_array : numpy array (2D)
        Vegetation Index array.
    lst_array : numpy array (2D)
        Land Surface Temperature array (Kelvin).
    vi_lower_limit : float
        VI partition between "vegetated" and "bare" pixels (default 0.0).
        Cold candidates must have VI >= this, hot candidates VI <= this.
    cold_percentile : float
        Coldest fraction of vegetated pixels considered as cold candidates
        (default 2.0, i.e. coldest 2%).
    hot_percentile : float
        Hottest fraction of bare pixels considered as hot candidates
        (default 98.0, i.e. hottest 2%).
    search_mask : numpy array (bool, 2D), optional
        Base mask of valid pixels to search within.
    cold_zone : numpy array (bool, 2D), optional
        Mask restricting cold pixel search area.
    hot_zone : numpy array (bool, 2D), optional
        Mask restricting hot pixel search area.

    Returns
    -------
    cold_pixel : tuple (row, col) or None
    hot_pixel : tuple (row, col) or None
    """
    if search_mask is None:
        search_mask = np.isfinite(vi_array) & np.isfinite(lst_array)

    # ===== Cold pixel =====
    cold_mask = np.logical_and(search_mask, vi_array >= vi_lower_limit)
    if cold_zone is not None:
        cold_mask = np.logical_and(cold_mask, cold_zone)

    if not np.any(cold_mask):
        print('No valid cold pixel candidates found')
        return None, None

    n_cold_pool = int(np.sum(cold_mask))
    lst_cold_thresh = float(np.nanpercentile(lst_array[cold_mask],
                                             cold_percentile))
    cold_candidates = np.logical_and(cold_mask, lst_array <= lst_cold_thresh)

    # Tie-break: among the coldest candidates, pick the one with highest VI.
    vi_within = np.where(cold_candidates, vi_array, -np.inf)
    cold_pixel = tuple(np.unravel_index(int(np.argmax(vi_within)),
                                        vi_array.shape))
    print('Cold pixel at (%d, %d): %.2f K, %.3f VI '
          '(coldest %.1f%% of %d veg pixels, highest-VI tie-break)' % (
        cold_pixel[0], cold_pixel[1],
        float(lst_array[cold_pixel]), float(vi_array[cold_pixel]),
        cold_percentile, n_cold_pool))

    # ===== Hot pixel =====
    hot_mask = np.logical_and(search_mask, vi_array <= vi_lower_limit)
    if hot_zone is not None:
        hot_mask = np.logical_and(hot_mask, hot_zone)

    if not np.any(hot_mask):
        print('No pixels with VI <= %.2f in hot search area' % vi_lower_limit)
        if hot_zone is not None and np.any(np.logical_and(search_mask, hot_zone)):
            print('Falling back to hottest pixel in hot zone regardless of VI')
            hot_mask = np.logical_and(search_mask, hot_zone)
        else:
            print('Falling back to hottest pixel in full search area')
            hot_mask = search_mask.copy()

    n_hot_pool = int(np.sum(hot_mask))
    lst_hot_thresh = float(np.nanpercentile(lst_array[hot_mask],
                                            hot_percentile))
    hot_candidates = np.logical_and(hot_mask, lst_array >= lst_hot_thresh)

    # Tie-break: among the hottest candidates, pick the one with lowest VI.
    vi_within = np.where(hot_candidates, vi_array, np.inf)
    hot_pixel = tuple(np.unravel_index(int(np.argmin(vi_within)),
                                       vi_array.shape))
    print('Hot pixel at (%d, %d): %.2f K, %.3f VI '
          '(hottest %.1f%% of %d bare pixels, lowest-VI tie-break)' % (
        hot_pixel[0], hot_pixel[1],
        float(lst_array[hot_pixel]), float(vi_array[hot_pixel]),
        100.0 - hot_percentile, n_hot_pool))

    return cold_pixel, hot_pixel


def cimec(vi_array, lst_array, albedo_array, sza_array, cv_ndvi, cv_lst,
          adjust_rainfall=False, search_mask=None, cold_zone=None, hot_zone=None):
    '''Find hot and cold pixels using CIMEC algorithm.

    Calibration using Inverse Modelling at Extreme Conditions.
    Optionally restricts search to user-defined zones.

    Parameters
    ----------
    vi_array : numpy array (2D)
        Vegetation Index array (-)
    lst_array : numpy array (2D)
        Land Surface Temperature array (Kelvin)
    albedo_array : numpy array (2D)
        Surface albedo
    sza_array : numpy array (2D)
        Solar zenith angle (degrees)
    cv_ndvi : numpy array (2D)
        Coefficient of variation of NDVI
    cv_lst : numpy array (2D)
        Coefficient of variation of LST
    adjust_rainfall : bool or tuple
        If tuple (rainfall_60, ETr_60): adjust hot temperature
    search_mask : numpy array (bool, 2D), optional
        Base mask of valid pixels (e.g. landcover-filtered AOI)
    cold_zone : numpy array (bool, 2D), optional
        Geographic mask restricting cold pixel search
    hot_zone : numpy array (bool, 2D), optional
        Geographic mask restricting hot pixel search

    Returns
    -------
    cold_pixel : tuple (row, col) or None
    hot_pixel : tuple (row, col) or None

    References
    ----------
    Allen et al. 2013, JAWRA 49(3):563-576
    '''

    valid = np.isfinite(vi_array) & np.isfinite(lst_array)
    if search_mask is not None:
        valid = np.logical_and(valid, search_mask)

    # =========================================================================
    # Cold pixel search
    # =========================================================================
    cold_valid = valid.copy()
    if cold_zone is not None:
        cold_valid = np.logical_and(cold_valid, cold_zone)

    if not np.any(cold_valid):
        print('No valid pixels in cold search area')
        return None, None

    # Step 1. Find the 5% top NDVI pixels within cold search area
    ndvi_top = np.nanpercentile(vi_array[cold_valid], 95)
    ndvi_index = np.logical_and(vi_array >= ndvi_top, cold_valid)

    if not np.any(ndvi_index):
        print('No pixels with NDVI >= 95th percentile in cold zone')
        return None, None

    # Step 2. Coldest 20% LST from high-NDVI pixels
    lst_low = np.nanpercentile(lst_array[ndvi_index], 20)
    lst_index = np.logical_and(lst_array <= lst_low, cold_valid)
    lst_cold = np.nanmean(lst_array[lst_index])

    # Step 3. Filter by temperature tolerance and albedo threshold
    beta = 90.0 - sza_array  # Solar elevation angle
    albedo_thres = 0.001343 * beta + 0.3281 * np.exp(-0.0188 * beta)  # Eq. 7
    cold_candidates = np.logical_and.reduce((
        lst_index,
        np.abs(lst_array - lst_cold) <= 0.2,
        np.abs(albedo_array - albedo_thres) <= 0.02))

    if not np.any(cold_candidates):
        # Relax albedo constraint
        print('Relaxing albedo constraint for cold pixel search')
        cold_candidates = np.logical_and(
            lst_index, np.abs(lst_array - lst_cold) <= 0.2)

    if not np.any(cold_candidates):
        print('No cold pixel candidates found after filtering')
        return None, None

    # Step 5. Select most homogeneous pixel
    cold_candidates = np.logical_and(
        cold_candidates, cv_lst == np.nanmin(cv_lst[cold_candidates]))

    cold_pixel = tuple(np.argwhere(cold_candidates)[0])
    print('Cold pixel found at (%d, %d) with %.2f K and %.3f VI' % (
        cold_pixel[0], cold_pixel[1],
        float(lst_array[cold_pixel]), float(vi_array[cold_pixel])))

    # =========================================================================
    # Hot pixel search
    # =========================================================================
    hot_valid = valid.copy()
    if hot_zone is not None:
        hot_valid = np.logical_and(hot_valid, hot_zone)

    if not np.any(hot_valid):
        print('No valid pixels in hot search area')
        return cold_pixel, None

    # Step 1. Find the 10% lowest NDVI within hot search area
    ndvi_low = np.nanpercentile(vi_array[hot_valid], 10)
    ndvi_index = np.logical_and(vi_array <= ndvi_low, hot_valid)

    if not np.any(ndvi_index):
        ndvi_low = np.nanpercentile(vi_array[hot_valid], 20)
        ndvi_index = np.logical_and(vi_array <= ndvi_low, hot_valid)

    # Step 2. Hottest 20% LST from low-NDVI pixels
    lst_high = np.nanpercentile(lst_array[ndvi_index], 80)
    lst_index = np.logical_and(lst_array >= lst_high, hot_valid)
    lst_hot = np.nanmean(lst_array[lst_index])

    if not isinstance(adjust_rainfall, bool):
        lst_hot -= 2.6 - 13.0 * adjust_rainfall[0] / adjust_rainfall[1]  # Eq. 8

    # Step 4. Filter by temperature tolerance and homogeneity
    hot_candidates = np.logical_and(
        lst_index, np.abs(lst_array - lst_hot) <= 0.2)

    if not np.any(hot_candidates):
        print('Relaxing temperature tolerance for hot pixel search')
        hot_candidates = np.logical_and(
            lst_index, np.abs(lst_array - lst_hot) <= 0.5)

    if not np.any(hot_candidates):
        print('No hot pixel candidates found after filtering')
        return cold_pixel, None

    hot_candidates = np.logical_and(
        hot_candidates, cv_ndvi == np.nanmin(cv_ndvi[hot_candidates]))

    hot_pixel = tuple(np.argwhere(hot_candidates)[0])
    print('Hot pixel found at (%d, %d) with %.2f K and %.3f VI' % (
        hot_pixel[0], hot_pixel[1],
        float(lst_array[hot_pixel]), float(vi_array[hot_pixel])))

    return cold_pixel, hot_pixel


def esa(vi_array, lst_array, cv_vi, std_lst, cv_albedo,
        search_mask=None, cold_zone=None, hot_zone=None):
    '''Find hot and cold pixels using the Exhaustive Search Algorithm.

    Parameters
    ----------
    vi_array : numpy array (2D)
        Vegetation Index array (-)
    lst_array : numpy array (2D)
        Land Surface Temperature array (Kelvin)
    cv_vi : numpy array (2D)
        Coefficient of variation of VI
    std_lst : numpy array (2D)
        Standard deviation of LST
    cv_albedo : numpy array (2D)
        Coefficient of variation of albedo
    search_mask : numpy array (bool, 2D), optional
        Base mask of valid pixels
    cold_zone : numpy array (bool, 2D), optional
        Mask restricting cold pixel search area
    hot_zone : numpy array (bool, 2D), optional
        Mask restricting hot pixel search area

    Returns
    -------
    cold_pixel : tuple (row, col) or None
    hot_pixel : tuple (row, col) or None

    References
    ----------
    Bhattarai et al. 2017, RSE 196:178-192
    '''

    lst_nan = np.isnan(lst_array)
    vi_nan = np.isnan(vi_array)
    if np.all(lst_nan) or np.all(vi_nan):
        print('No valid LST or VI pixels')
        return None, None

    base_valid = ~lst_nan & ~vi_nan
    if search_mask is not None:
        base_valid = np.logical_and(base_valid, search_mask)

    # Step 1. Find homogeneous pixels
    print('Filtering pixels by homogeneity')
    homogeneous = np.logical_and.reduce((
        base_valid,
        cv_vi <= 0.25,
        cv_albedo <= 0.25,
        std_lst < 1.5))

    print('Found %s homogeneous pixels' % np.sum(homogeneous))
    if np.sum(homogeneous) == 0:
        return None, None

    # Step 2. Filter outliers by histogram
    lst_min, lst_max, vi_min, vi_max = histogram_filter(
        vi_array[base_valid], lst_array[base_valid])

    print('Removing outliers by histogram')
    mask = np.logical_and.reduce((
        homogeneous,
        lst_array >= lst_min,
        lst_array <= lst_max,
        vi_array >= vi_min,
        vi_array <= vi_max))

    print('Keep %s pixels after outlier removal' % np.sum(mask))
    if np.sum(mask) == 0:
        return None, None

    # Step 3. Search for cold pixels (restricted to cold_zone)
    cold_mask = mask.copy()
    if cold_zone is not None:
        cold_mask = np.logical_and(cold_mask, cold_zone)

    print('Searching for cold pixel candidates (%d pixels in search area)'
          % np.sum(cold_mask))
    cold_pixels = incremental_search(vi_array, lst_array, cold_mask,
                                     is_cold=True)
    n_cold = np.sum(cold_pixels) if isinstance(cold_pixels, np.ndarray) else 0
    print('Found %s candidate cold pixels' % n_cold)
    if n_cold == 0:
        return None, None

    # Search for hot pixels (restricted to hot_zone)
    hot_mask = mask.copy()
    if hot_zone is not None:
        hot_mask = np.logical_and(hot_mask, hot_zone)

    print('Searching for hot pixel candidates (%d pixels in search area)'
          % np.sum(hot_mask))
    hot_pixels = incremental_search(vi_array, lst_array, hot_mask,
                                    is_cold=False)
    n_hot = np.sum(hot_pixels) if isinstance(hot_pixels, np.ndarray) else 0
    print('Found %s candidate hot pixels' % n_hot)
    if n_hot == 0:
        return None, None

    # Step 4. Rank candidates
    print('Ranking candidate anchor pixels')
    lst_rank = rank_array(lst_array)
    vi_rank = rank_array(vi_array)

    rank = vi_rank - lst_rank
    cold_best = np.logical_and(cold_pixels,
                               rank == np.max(rank[cold_pixels]))
    cold_pixel = tuple(np.argwhere(cold_best)[0])
    print('Cold pixel found at (%d, %d) with %.2f K and %.3f VI' % (
        cold_pixel[0], cold_pixel[1],
        float(lst_array[cold_pixel]), float(vi_array[cold_pixel])))

    rank = lst_rank - vi_rank
    hot_best = np.logical_and(hot_pixels,
                              rank == np.max(rank[hot_pixels]))
    hot_pixel = tuple(np.argwhere(hot_best)[0])
    print('Hot pixel found at (%d, %d) with %.2f K and %.3f VI' % (
        hot_pixel[0], hot_pixel[1],
        float(lst_array[hot_pixel]), float(vi_array[hot_pixel])))

    return cold_pixel, hot_pixel


# =============================================================================
# Utility functions
# =============================================================================

def validate_endmembers(lst_array, vi_array, cold_pixel, hot_pixel,
                        min_lst_range=DEFAULT_MIN_LST_RANGE):
    """Validate selected endmembers and print quality warnings.

    Parameters
    ----------
    lst_array : numpy array
        Land Surface Temperature (K) - can be 1D or 2D
    vi_array : numpy array
        Vegetation Index - same shape as lst_array
    cold_pixel : tuple
        Cold pixel index/coordinates
    hot_pixel : tuple
        Hot pixel index/coordinates
    min_lst_range : float
        Minimum recommended LST difference (K)

    Returns
    -------
    is_valid : bool
    warnings : list of str
    """
    if cold_pixel is None or hot_pixel is None:
        return False, ['Endmember search failed to find valid pixels']

    lst_cold = float(lst_array[cold_pixel])
    lst_hot = float(lst_array[hot_pixel])
    vi_cold = float(vi_array[cold_pixel])
    vi_hot = float(vi_array[hot_pixel])
    lst_range = lst_hot - lst_cold

    warnings = []

    if lst_range <= 0:
        warnings.append(
            'ERROR: Hot pixel (%.1f K) is not warmer than cold pixel '
            '(%.1f K). Check endmember zones or input data.'
            % (lst_hot, lst_cold))

    elif lst_range < min_lst_range:
        warnings.append(
            'WARNING: LST range between endmembers is only %.1f K '
            '(hot=%.1f K, cold=%.1f K). Minimum recommended: %.1f K. '
            'Results may be unreliable.'
            % (lst_range, lst_hot, lst_cold, min_lst_range))

    if vi_hot > 0.4:
        warnings.append(
            'WARNING: Hot pixel VI is %.2f (expected < 0.2 for bare soil). '
            'This pixel may have significant vegetation cover.' % vi_hot)

    if vi_cold < 0.3:
        warnings.append(
            'WARNING: Cold pixel VI is %.2f (expected > 0.5 for full cover). '
            'This pixel may not represent well-watered vegetation.' % vi_cold)

    for w in warnings:
        print(w)

    if not warnings:
        print('Endmember quality OK: LST range = %.1f K, '
              'cold VI = %.2f, hot VI = %.2f'
              % (lst_range, vi_cold, vi_hot))

    return len(warnings) == 0, warnings


def create_zone_masks_from_polygons(dims, geo_transform, projection,
                                    polygon_path, zone_field='zone'):
    """Create cold and hot zone boolean masks from a polygon feature file.

    The polygon file should contain features with an attribute field
    identifying each polygon as 'cold' or 'hot' (case-insensitive).

    Results are memoised by (abspath, mtime, dims, geo_transform,
    projection, zone_field) — within a single process, a second call with
    the same inputs returns the cached masks immediately. The cache holds
    up to 4 entries. Callers must not mutate the returned arrays.

    Parameters
    ----------
    dims : tuple
        Image dimensions (rows, cols)
    geo_transform : tuple
        GDAL GeoTransform
    projection : str
        Image projection as WKT
    polygon_path : str
        Path to the polygon feature file (GeoPackage, Shapefile, etc.)
    zone_field : str
        Name of the attribute field containing 'cold' or 'hot' labels.
        If the field is not found, falls back to checking common field
        names: 'zone', 'type', 'Zone', 'Type', 'name', 'Name'.

    Returns
    -------
    cold_mask : numpy array (bool) or None
    hot_mask : numpy array (bool) or None
    """
    try:
        mtime = os.path.getmtime(polygon_path)
    except OSError:
        # File missing or unreadable — bypass the cache so the helper's
        # ogr.Open path reports the error to the user.
        mtime = -1.0
    return _create_zone_masks_cached(
        tuple(dims),
        tuple(geo_transform),
        projection,
        os.path.abspath(polygon_path),
        mtime,
        zone_field,
    )


@lru_cache(maxsize=4)
def _create_zone_masks_cached(dims, geo_transform, projection,
                              polygon_path, mtime, zone_field):
    """Memoised body for create_zone_masks_from_polygons."""
    from osgeo import gdal, ogr, osr

    ds_vec = ogr.Open(polygon_path)
    if ds_vec is None:
        print('ERROR: cannot open polygon file: %s' % polygon_path)
        return None, None

    layer = ds_vec.GetLayer(0)
    if layer is None:
        print('ERROR: no layer found in %s' % polygon_path)
        ds_vec = None
        return None, None

    # Find the zone attribute field
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName()
                   for i in range(layer_defn.GetFieldCount())]

    candidates = [zone_field, 'zone', 'type', 'Zone', 'Type', 'name', 'Name']
    found_field = None
    for candidate in candidates:
        if candidate in field_names:
            found_field = candidate
            break

    if found_field is None:
        print('ERROR: no zone attribute field found in %s' % polygon_path)
        print('  Available fields: %s' % ', '.join(field_names))
        print('  Expected one of: %s' % ', '.join(candidates))
        ds_vec = None
        return None, None

    print('Using zone field: "%s" from %s' % (found_field, polygon_path))

    # Set up coordinate transformation if needed
    vec_srs = layer.GetSpatialRef()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(projection)

    transform = None
    if vec_srs is not None and raster_srs is not None:
        vec_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not vec_srs.IsSame(raster_srs):
            transform = osr.CoordinateTransformation(vec_srs, raster_srs)
            print('  Reprojecting polygon features to image CRS')

    # Separate features into cold and hot
    cold_geoms = []
    hot_geoms = []
    for feat in layer:
        zone_val = str(feat.GetField(found_field)).strip().lower()
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        geom = geom.Clone()
        if transform is not None:
            geom.Transform(transform)

        if zone_val in ('cold', 'wet', 'c'):
            cold_geoms.append(geom)
        elif zone_val in ('hot', 'dry', 'h'):
            hot_geoms.append(geom)
        else:
            print('  WARNING: skipping feature with zone="%s" '
                  '(expected cold/hot/wet/dry)' % zone_val)

    ds_vec = None

    if not cold_geoms and not hot_geoms:
        print('ERROR: no cold or hot features found in %s' % polygon_path)
        return None, None

    def rasterize_geoms(geoms, dims, geo_transform, projection):
        """Rasterize a list of geometries into a boolean mask."""
        if not geoms:
            return None

        mem_drv = ogr.GetDriverByName('MEM')
        mem_ds = mem_drv.CreateDataSource('')
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        mem_layer = mem_ds.CreateLayer('', srs=srs, geom_type=ogr.wkbPolygon)

        for geom in geoms:
            feat = ogr.Feature(mem_layer.GetLayerDefn())
            feat.SetGeometry(geom)
            mem_layer.CreateFeature(feat)
            feat = None

        rast_drv = gdal.GetDriverByName('MEM')
        rast_ds = rast_drv.Create('', dims[1], dims[0], 1, gdal.GDT_Byte)
        rast_ds.SetGeoTransform(geo_transform)
        rast_ds.SetProjection(projection)
        rast_ds.GetRasterBand(1).Fill(0)

        gdal.RasterizeLayer(rast_ds, [1], mem_layer, burn_values=[1])
        mask = rast_ds.GetRasterBand(1).ReadAsArray().astype(bool)

        mem_ds = None
        rast_ds = None
        return mask

    cold_mask = rasterize_geoms(cold_geoms, dims, geo_transform, projection)
    hot_mask = rasterize_geoms(hot_geoms, dims, geo_transform, projection)

    if cold_mask is not None:
        print('  Cold zone: %d pixels from %d polygon(s)'
              % (np.sum(cold_mask), len(cold_geoms)))
    if hot_mask is not None:
        print('  Hot zone: %d pixels from %d polygon(s)'
              % (np.sum(hot_mask), len(hot_geoms)))

    return cold_mask, hot_mask


def create_zone_mask(dims, geo_transform, zone_coords):
    """Create a boolean mask from geographic bounding box coordinates.

    Parameters
    ----------
    dims : tuple
        Image dimensions (rows, cols)
    geo_transform : tuple
        GDAL GeoTransform (origin_x, pixel_w, x_rot, origin_y, y_rot, pixel_h)
    zone_coords : str or list
        Bounding box as 'xmin,ymin,xmax,ymax' in the image coordinate system

    Returns
    -------
    mask : numpy array (bool)
    """
    if isinstance(zone_coords, str):
        coords = [float(c.strip()) for c in zone_coords.split(',')]
    else:
        coords = list(zone_coords)

    xmin, ymin, xmax, ymax = coords

    col_min = int((xmin - geo_transform[0]) / geo_transform[1])
    col_max = int((xmax - geo_transform[0]) / geo_transform[1])

    # For north-up images, geo_transform[5] is negative
    if geo_transform[5] < 0:
        row_min = int((ymax - geo_transform[3]) / geo_transform[5])
        row_max = int((ymin - geo_transform[3]) / geo_transform[5])
    else:
        row_min = int((ymin - geo_transform[3]) / geo_transform[5])
        row_max = int((ymax - geo_transform[3]) / geo_transform[5])

    # Clamp to image bounds
    row_min = max(0, min(row_min, dims[0] - 1))
    row_max = max(0, min(row_max, dims[0] - 1))
    col_min = max(0, min(col_min, dims[1] - 1))
    col_max = max(0, min(col_max, dims[1] - 1))

    n_pixels = (row_max - row_min + 1) * (col_max - col_min + 1)
    print('Zone mask: rows %d-%d, cols %d-%d (%d pixels)'
          % (row_min, row_max, col_min, col_max, n_pixels))

    mask = np.zeros(dims, dtype=bool)
    mask[row_min:row_max + 1, col_min:col_max + 1] = True

    return mask


def geo_to_pixel(geo_transform, x, y):
    """Convert geographic coordinates to pixel (row, col).

    Parameters
    ----------
    geo_transform : tuple
        GDAL GeoTransform
    x, y : float
        Geographic coordinates

    Returns
    -------
    row, col : int
    """
    col = int((x - geo_transform[0]) / geo_transform[1])
    row = int((y - geo_transform[3]) / geo_transform[5])
    return row, col


def global_to_aoi_index(global_coord, aoi_mask):
    """Map a global (row, col) pixel to its position in the 1D ``array[aoi_mask]`` view.

    ``METRIC.METRIC`` is called with arrays already sliced by the AOI bool mask
    (1D, shape ``(n_aoi,)``). This helper translates a full-image (row, col)
    coordinate to the integer index into that 1D view.

    Parameters
    ----------
    global_coord : tuple
        (row, col) in the full 2D image.
    aoi_mask : numpy array (bool, 2D)
        The AOI mask used to create the 1D array.

    Returns
    -------
    int or None
        Index into ``array[aoi_mask]``. Returns None if the pixel is out of
        bounds or outside the AOI (with an explanatory message).
    """
    row, col = global_coord
    rows, cols = aoi_mask.shape
    if not (0 <= row < rows and 0 <= col < cols):
        print('ERROR: pixel (%d, %d) outside image bounds (%d, %d)'
              % (row, col, rows, cols))
        return None
    if not aoi_mask[row, col]:
        print('WARNING: pixel (%d, %d) is not within the AOI mask'
              % (row, col))
        return None

    # numpy's array[bool_mask] returns elements in row-major (ravel) order,
    # so the local 1D index equals the count of True cells preceding (row, col).
    flat_idx = row * cols + col
    local_idx = int(np.count_nonzero(aoi_mask.ravel()[:flat_idx]))
    return local_idx


def compute_cv_window(pixel_size_meters, target_meters=5.0,
                      min_pixels=3, max_pixels=51):
    """Compute CV filter window size based on pixel resolution.

    For UAV imagery at sub-meter resolution, the default 11x11 window
    covers only ~1.8m x 1.8m, which is too small for meaningful
    homogeneity assessment. This function scales the window to a
    physically meaningful size.

    Parameters
    ----------
    pixel_size_meters : float
        Pixel size in meters
    target_meters : float
        Target physical window size in meters (default 5.0m)
    min_pixels : int
        Minimum window size in pixels
    max_pixels : int
        Maximum window size in pixels

    Returns
    -------
    window_size : int
        Odd-numbered window size in pixels
    """
    window = int(np.ceil(target_meters / pixel_size_meters))
    if window % 2 == 0:
        window += 1
    window = max(min_pixels, min(window, max_pixels))
    if window % 2 == 0:
        window += 1

    actual_meters = window * pixel_size_meters
    print('CV window: %d x %d pixels (%.1f x %.1f m at %.3f m resolution)'
          % (window, window, actual_meters, actual_meters, pixel_size_meters))

    return window


# =============================================================================
# Internal helpers
# =============================================================================

def histogram_filter(vi_array, lst_array):
    """Filter outliers using histogram tail analysis.

    Trims histogram tails until each extreme bin has >= 50 pixels.

    Parameters
    ----------
    vi_array : numpy array (1D)
        Vegetation index values (pre-masked)
    lst_array : numpy array (1D)
        LST values (pre-masked)

    Returns
    -------
    lst_min, lst_max, vi_min, vi_max : float
        Valid data bounds after outlier removal
    """
    cold_bin_pixels = 0
    hot_bin_pixels = 0
    bare_bin_pixels = 0
    full_bin_pixels = 0

    max_iter = 100
    for iteration in range(max_iter):
        if (cold_bin_pixels >= 50 and hot_bin_pixels >= 50
                and bare_bin_pixels >= 50 and full_bin_pixels >= 50):
            break

        if len(lst_array) < 50 or len(vi_array) < 50:
            print('WARNING: too few pixels (%d LST, %d VI) for histogram filter'
                  % (len(lst_array), len(vi_array)))
            break

        max_lst = np.nanmax(lst_array)
        min_lst = np.nanmin(lst_array)
        max_vi = np.nanmax(vi_array)
        min_vi = np.nanmin(vi_array)

        n_bins = max(1, int(np.ceil((max_lst - min_lst) / 0.25)))
        lst_hist, lst_edges = np.histogram(lst_array, n_bins)

        n_bins = max(1, int(np.ceil((max_vi - min_vi) / 0.01)))
        vi_hist, vi_edges = np.histogram(vi_array, n_bins)

        cold_bin_pixels = lst_hist[0]
        hot_bin_pixels = lst_hist[-1]
        bare_bin_pixels = vi_hist[0]
        full_bin_pixels = vi_hist[-1]

        if cold_bin_pixels < 50:
            lst_array = lst_array[lst_array >= lst_edges[1]]
        if hot_bin_pixels < 50:
            lst_array = lst_array[lst_array <= lst_edges[-2]]
        if bare_bin_pixels < 50:
            vi_array = vi_array[vi_array >= vi_edges[1]]
        if full_bin_pixels < 50:
            vi_array = vi_array[vi_array <= vi_edges[-2]]
    else:
        print('WARNING: histogram filter did not converge after %d iterations'
              % max_iter)

    return lst_edges[0], lst_edges[-1], vi_edges[0], vi_edges[-1]


def rank_array(array):
    """Rank array elements."""
    temp = array.argsort(axis=None)
    ranks = np.arange(np.size(array))[temp.argsort()].reshape(array.shape)
    return ranks


def incremental_search(vi_array, lst_array, mask, is_cold=True):
    """Iteratively search for endmember candidates by expanding percentile range.

    Parameters
    ----------
    vi_array : numpy array
        Vegetation index
    lst_array : numpy array
        Land surface temperature
    mask : numpy array (bool)
        Valid pixel mask (includes zone restriction if any)
    is_cold : bool
        True for cold pixel search, False for hot

    Returns
    -------
    candidates : numpy array (bool)
        Mask of candidate pixels, or zero-filled array on failure
    """
    if np.sum(mask) < 10:
        print('WARNING: fewer than 10 pixels in search mask')
        return np.zeros(vi_array.shape, dtype=bool)

    step = 0
    if is_cold:
        while True:
            for n_lst in range(1, 11 + step):
                for n_vi in range(1, 11 + step):
                    vi_high = np.nanpercentile(vi_array[mask], 100 - n_vi)
                    lst_cold = np.nanpercentile(lst_array[mask], n_lst)
                    cold_index = np.logical_and.reduce((
                        mask,
                        vi_array >= vi_high,
                        lst_array <= lst_cold))
                    if np.sum(cold_index) >= 10:
                        return cold_index
            step += 5
            if step > 90:
                return np.zeros(vi_array.shape, dtype=bool)
    else:
        while True:
            for n_lst in range(1, 11 + step):
                for n_vi in range(1, 11 + step):
                    vi_low = np.nanpercentile(vi_array[mask], n_vi)
                    lst_hot = np.nanpercentile(lst_array[mask], 100 - n_lst)
                    hot_index = np.logical_and.reduce((
                        mask,
                        vi_array <= vi_low,
                        lst_array >= lst_hot))
                    if np.sum(hot_index) >= 10:
                        return hot_index
            step += 5
            if step > 90:
                return np.zeros(vi_array.shape, dtype=bool)


def moving_cv_filter(data, window):
    '''Compute coefficient of variation in a moving window.

    Parameters
    ----------
    data : numpy array (2D)
        Input data
    window : tuple or int
        Moving window dimensions (rows, columns). If int, used for both.

    Returns
    -------
    cv : numpy array
        Coefficient of variation (std/mean)
    mean : numpy array
        Local mean
    std : numpy array
        Local standard deviation
    '''
    if isinstance(window, (int, np.integer)):
        window = (window, window)

    kernel = np.ones(window) / np.prod(np.asarray(window))
    mean = convolve2d(data, kernel, mode='same', boundary='symm')
    distance = (data - mean) ** 2
    std = np.sqrt(convolve2d(distance, kernel, mode='same', boundary='symm'))
    cv = std / mean

    return cv, mean, std
