# -*- coding: utf-8 -*-
"""
Geometry helpers: normalisation of the inputs accepted by the bulk API and
reprojection between coordinate reference systems.
"""

import os
import threading

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from pyproj import CRS, Transformer

DEFAULT_CRS = "EPSG:4326"

# pyproj transformers are not thread safe: one cache per thread.
_local = threading.local()


def _transformer(src, dst):
    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = {}
    key = (CRS.from_user_input(src).to_wkt(), CRS.from_user_input(dst).to_wkt())
    if key not in cache:
        cache[key] = Transformer.from_crs(src, dst, always_xy=True)
    return cache[key]


def reproject(geometry, src, dst):
    """Reprojects a shapely geometry from `src` to `dst` (x = lon/easting)."""
    if CRS.from_user_input(src) == CRS.from_user_input(dst):
        return geometry
    tr = _transformer(src, dst)
    return shapely.transform(geometry, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def _from_records(records, crs):
    """List of GeoJSON features / geometries / shapely geometries -> GeoDataFrame."""
    geoms, props = [], []
    for rec in records:
        if isinstance(rec, BaseGeometry):
            geoms.append(rec)
            props.append({})
        elif isinstance(rec, dict) and rec.get("type") == "Feature":
            geoms.append(shape(rec["geometry"]))
            props.append(dict(rec.get("properties") or {}))
        elif isinstance(rec, dict) and "coordinates" in rec:
            geoms.append(shape(rec))
            props.append({})
        else:
            raise TypeError("unsupported record type: {!r}".format(type(rec)))
    return gpd.GeoDataFrame(pd.DataFrame(props), geometry=geoms, crs=crs)


def to_geodataframe(data, crs=None):
    """
    Normalises the inputs of the bulk API into a GeoDataFrame.

    Accepted inputs:
      - a GeoDataFrame or GeoSeries (its CRS is used; `crs` is used if it has none)
      - a pandas DataFrame with a `geometry` column (shapely or WKB)
      - a GeoJSON FeatureCollection, Feature or geometry (dict)
      - a shapely geometry, or a list of features / geometries
      - a path to any file readable by geopandas (.geojson, .gpkg, .shp, ...)
        or to a (Geo)Parquet file

    GeoJSON dicts and shapely geometries are assumed to be in `crs`
    (EPSG:4326 by default, as mandated by the GeoJSON spec).
    """
    if isinstance(data, gpd.GeoDataFrame):
        gdf = data.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(crs or DEFAULT_CRS)
        return gdf
    if isinstance(data, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=data)
        return gdf if gdf.crs is not None else gdf.set_crs(crs or DEFAULT_CRS)
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        if path.lower().endswith((".parquet", ".geoparquet")):
            try:
                gdf = gpd.read_parquet(path)
            except ValueError:  # plain parquet with a WKB geometry column
                return to_geodataframe(pd.read_parquet(path), crs=crs)
        else:
            gdf = gpd.read_file(path)
        return gdf if gdf.crs is not None else gdf.set_crs(crs or DEFAULT_CRS)
    if isinstance(data, pd.DataFrame):
        if "geometry" not in data.columns:
            raise ValueError("a DataFrame input needs a `geometry` column")
        geoms = data["geometry"]
        if len(geoms) and isinstance(geoms.iloc[0], (bytes, bytearray, memoryview)):
            geoms = shapely.from_wkb(geoms.values)
        return gpd.GeoDataFrame(data.drop(columns="geometry"), geometry=list(geoms),
                                crs=crs or DEFAULT_CRS)
    if isinstance(data, BaseGeometry):
        return _from_records([data], crs or DEFAULT_CRS)
    if isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            return _from_records(data["features"], crs or DEFAULT_CRS)
        return _from_records([data], crs or DEFAULT_CRS)
    if isinstance(data, (list, tuple)):
        return _from_records(data, crs or DEFAULT_CRS)
    raise TypeError("unsupported input type: {!r}".format(type(data)))
