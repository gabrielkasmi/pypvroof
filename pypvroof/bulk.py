# -*- coding: utf-8 -*-
"""
Bulk characterization of roof planes (tilt, azimuth) under PV polygons, from
a DSM service and the Theil-Sen estimator.

    from pypvroof import characterize_roofs, IGNLidarService
    out = characterize_roofs("detections.parquet", IGNLidarService(), workers=8)

The output is always a GeoDataFrame (same CRS as the input), one row per
input polygon, or several when `split=True` and a polygon covers several roof
planes. pypvroof never applies a fallback: rows without a value carry a
`status` explaining why, and the service metadata (`meta_*` columns), so that
the caller decides what to do.

Statuses:
  ok, flat                          -> estimated (azimuth NaN when flat)
  multimodal                        -> several roof planes: NaN, inspect=True
                                       (or split into parts with split=True)
  too_few_points, fit_failed        -> estimator could not conclude
  no_coverage, service_error,
  too_large                         -> from the DSM service (see dsm.py)
  error                             -> unexpected exception (see meta_error)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy import ndimage

from . import dsm, theilsen
from .geometry import to_geodataframe, reproject

ERROR = "error"

RESULT_COLUMNS = ["tilt", "azimuth", "status", "inspect", "surface_proj", "surface",
                  "n_pixels", "n_modes", "primary_share", "inlier_frac", "rmse",
                  "roughness", "offset_suspected", "modes"]


def _empty(status):
    r = theilsen._nan_result(status)
    r.update(surface_proj=np.nan, surface=np.nan)
    return r


def _with_surfaces(result, area):
    result["surface_proj"] = float(area)
    tilt = result.get("tilt", np.nan)
    result["surface"] = float(area / np.cos(np.radians(tilt))) if np.isfinite(tilt) else np.nan
    return result


def polygon_mask(geom, patch):
    """
    Boolean grid of the pixels whose centre lies in `geom`.

    Implemented with shapely rather than rasterio.features, which is not
    thread safe (it intermittently ignores the transform when called from
    several threads).
    """
    h, w = patch.z.shape
    rows, cols = np.mgrid[0:h, 0:w]
    x, y = patch.pixel_centers(rows, cols)
    shapely.prepare(geom)
    return shapely.contains_xy(geom, x, y)


def _masks(geom, patch, p):
    """(full polygon mask, estimation mask, edge distance in metres) on the patch grid."""
    full = polygon_mask(geom, patch)
    xres, _ = patch.resolution
    mask = full
    if p.erosion > 0:
        with np.errstate(invalid="ignore"):  # spurious numpy warning in shapely's buffer
            inner = geom.buffer(-p.erosion)
        # keep the erosion only if what remains is still large enough
        if not inner.is_empty and inner.area >= max(p.min_pixels * xres * xres, p.min_area):
            mask = polygon_mask(inner, patch)
    edge = ndimage.distance_transform_edt(full) * xres
    return full, mask, edge


def _polygonal(geom):
    """Repairs invalid polygons (self-intersections), keeping polygonal parts only."""
    if geom.is_valid:
        return geom
    fixed = shapely.make_valid(geom)
    if fixed.geom_type == "GeometryCollection":
        parts = [g for g in fixed.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        fixed = shapely.union_all(parts) if parts else geom
    return fixed


def characterize_polygon(geometry, service, params=None, split=False):
    """
    Characterizes one polygon, expressed in `service.crs`.

    Returns a list of (geometry, result dict, meta dict), with one element, or
    one per part if `split=True` and the polygon is multimodal. Geometries
    are in `service.crs`.
    """
    p = theilsen.TheilSenParams.from_dict(params)
    try:
        fr = service.fetch(geometry)
    except Exception as e:  # a buggy service must not kill a bulk run
        return [(geometry, _empty(ERROR), {"error": repr(e)})]
    meta = dict(fr.meta or {})
    if fr.status != dsm.OK:
        return [(geometry, _with_surfaces(_empty(fr.status), geometry.area), meta)]

    try:
        full, mask, edge = _masks(geometry, fr.patch, p)
        result, fit = theilsen.estimate(fr.patch.z, fr.patch.transform, mask, edge, p,
                                        return_labels=True)
        if split and result["status"] == theilsen.MULTIMODAL and fit is not None:
            from .split import split_polygon
            parts = split_polygon(geometry, fr.patch, full, mask, fit, p)
            if parts:
                return [(g, _with_surfaces(r, g.area), meta) for g, r in parts]
    except Exception as e:
        meta["error"] = repr(e)
        return [(geometry, _with_surfaces(_empty(ERROR), geometry.area), meta)]
    return [(geometry, _with_surfaces(result, geometry.area), meta)]


def characterize_roofs(polygons, service="fr-lidar", *, crs=None, params=None, split=False,
                       workers=8, id_column=None, meta_prefix="meta_", progress=None):
    """
    Characterizes the roof planes under a set of polygons.

    Parameters
    ----------
    polygons : GeoDataFrame, DataFrame with a geometry column, GeoJSON dict,
        list of features / shapely geometries, or a path (GeoJSON, GeoPackage,
        (Geo)Parquet, ...). See geometry.to_geodataframe.
    service : a DSMService, a service name from pypvroof.catalog
              ("fr-lidar", "nl-ahn"...) or a folder of GeoTIFFs
    crs : CRS of the input when it does not carry one (default EPSG:4326)
    params : dict or TheilSenParams, estimator parameters
    split : split multimodal polygons into homogeneous parts
    workers : number of polygons processed concurrently (threads). The work is
        dominated by I/O for remote services; mind their usage limits.
    id_column : column identifying the input polygons, copied into
        `parent_id` (default: the input index)
    meta_prefix : prefix of the columns holding the service metadata
    progress : optional callable(done, total)

    Returns
    -------
    GeoDataFrame with the input columns (renamed with an `_input` suffix when
    they clash with the output ones), parent_id, part, the result columns
    (tilt, azimuth, status, inspect, surface_proj, surface, diagnostics) and
    the service metadata.
    """
    gdf = to_geodataframe(polygons, crs=crs)
    service = dsm.get_service(service)
    in_crs = gdf.crs
    p = theilsen.TheilSenParams.from_dict(params)

    def work(geom):
        if geom is None or geom.is_empty:
            return [(geom, _empty(ERROR), {"error": "empty geometry"})]
        geom = _polygonal(geom)
        parts = characterize_polygon(reproject(geom, in_crs, service.crs), service, p, split)
        return [(reproject(g, service.crs, in_crs), r, m) for g, r, m in parts]

    geoms = list(gdf.geometry.values)
    total = len(geoms)
    results = [None] * total
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(work, g): i for i, g in enumerate(geoms)}
            for done, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                if progress:
                    progress(done, total)
    else:
        for i, g in enumerate(geoms):
            results[i] = work(g)
            if progress:
                progress(i + 1, total)

    attrs = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    clash = set(attrs.columns) & (set(RESULT_COLUMNS) | {"parent_id", "part"})
    attrs = attrs.rename(columns={c: c + "_input" for c in clash})
    parent = gdf[id_column].values if id_column else gdf.index.values

    rows, out_geoms = [], []
    for i, parts in enumerate(results):
        base = attrs.iloc[i].to_dict()
        for k, (g, r, m) in enumerate(parts):
            row = dict(base)
            row["parent_id"] = parent[i]
            row["part"] = k
            row.update({c: r.get(c) for c in RESULT_COLUMNS})
            row.update({meta_prefix + key: v for key, v in (m or {}).items()})
            rows.append(row)
            out_geoms.append(g)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=out_geoms, crs=in_crs)
    return out
