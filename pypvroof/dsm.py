# -*- coding: utf-8 -*-
"""
Digital surface model (DSM) services.

A service retrieves and prepares the elevation data under a polygon. It knows
nothing about roofs: it returns a `FetchResult` holding a `DSMPatch` (an
elevation grid with its georeferencing) and a free-form `meta` dict whose
content depends on the service (e.g. LiDAR acquisition dates for the IGN
service, file names for the local one). pypvroof does not interpret `meta`;
it is passed through to the output so that the caller can act on it.

Statuses returned by `fetch`:
  - "ok"            : a patch with at least one valid pixel
  - "no_coverage"   : the service has no data at this location
  - "service_error" : the service could not be reached / returned an error
                      (after retries); retrying later may succeed
  - "too_large"     : the polygon exceeds what the service can return at once
"""

import glob
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rasterio
import rasterio.merge
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.strtree import STRtree

OK = "ok"
NO_COVERAGE = "no_coverage"
SERVICE_ERROR = "service_error"
TOO_LARGE = "too_large"


@dataclass
class DSMPatch:
    """
    Elevation grid around a polygon.

    z         : 2D float array of elevations (metres), NaN where there is no data
    transform : affine transform of the grid (rasterio convention, north up)
    crs       : coordinate reference system of the grid (projected, metres)
    """
    z: np.ndarray
    transform: Affine
    crs: CRS

    @property
    def resolution(self):
        return abs(self.transform.a), abs(self.transform.e)

    def pixel_centers(self, rows, cols):
        """Coordinates (x, y) of the centers of the given pixels."""
        x = self.transform.c + (np.asarray(cols) + 0.5) * self.transform.a
        y = self.transform.f + (np.asarray(rows) + 0.5) * self.transform.e
        return x, y


@dataclass
class FetchResult:
    status: str
    patch: Optional[DSMPatch] = None
    meta: dict = field(default_factory=dict)


class DSMService:
    """
    Base class of the DSM services.

    Subclasses set `crs` (the CRS of the elevation grids, in which polygons
    must be passed to `fetch`) and implement `fetch(geometry)`.
    """

    crs = None

    def fetch(self, geometry):
        """
        Returns a FetchResult for a shapely polygon expressed in `self.crs`.
        The grid must cover the polygon's bounding box (plus any margin).
        """
        raise NotImplementedError


def _snap(lo, hi, res, origin):
    """Snaps [lo, hi] outwards onto a grid of step `res` passing through `origin`."""
    lo = origin + np.floor((lo - origin) / res) * res
    hi = origin + np.ceil((hi - origin) / res) * res
    return float(lo), float(hi)


def _clean(z, nodata):
    z = z.astype(np.float64)
    if nodata is not None and not np.isnan(nodata):
        z[z == nodata] = np.nan
    z[z < -1000] = np.nan  # sentinel values such as -9999
    return z


class IGNLidarService(DSMService):
    """
    IGN LiDAR HD elevation models served by the Géoplateforme WMS-R
    (https://data.geopf.fr/wms-r), as raw float32 GeoTIFFs on any extent.

    Parameters
    ----------
    product : "MNS" (digital surface model, default), "MNT" or "MNH"
    margin : metres added around the polygon bounding box
    resolution : pixel size (0.5 m, the native resolution of the product)
    grid_origin : offset of the native pixel grid (the IGN tiles' edges are at
        x.25 / x.75), so that requested pixels match native ones (no resampling)
    timeout, retries, backoff : HTTP settings. Retries apply to network errors,
        HTTP 429 and 5xx, with exponential backoff and jitter.
    fetch_metadata : query the LiDAR HD WFS index for the acquisition metadata
        of the tile containing the polygon (cached per 1 km tile)
    user_agent : User-Agent header sent to the service

    `meta` keys: source, product, and if fetch_metadata: tile,
    date_debut_acquisition, date_fin_acquisition, date_edition, code_mission
    (or metadata_error). On errors: http_status, error.
    """

    WMS_URL = "https://data.geopf.fr/wms-r"
    WFS_URL = "https://data.geopf.fr/wfs/ows"
    WFS_LAYER = "IGNF_LIDAR-HD_METADONNEE:metadata"
    MAX_PIXELS = 5000  # WMS-R accepts up to 5010 x 5010
    METADATA_KEYS = ("date_debut_acquisition", "date_fin_acquisition", "date_edition",
                     "code_mission", "coordonnees_nw")

    crs = CRS.from_epsg(2154)

    def __init__(self, product="MNS", margin=1.0, resolution=0.5, grid_origin=0.25,
                 timeout=30, retries=3, backoff=1.0, fetch_metadata=True,
                 user_agent=None):
        from . import __version__
        product = product.upper()
        if product not in ("MNS", "MNT", "MNH"):
            raise ValueError("product must be MNS, MNT or MNH")
        self.product = product
        self.layer = "IGNF_LIDAR-HD_{}_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93".format(product)
        self.margin = margin
        self.resolution = resolution
        self.grid_origin = grid_origin
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.fetch_metadata = fetch_metadata
        self.user_agent = user_agent or "pypvroof/{}".format(__version__)
        self._metadata_cache = {}
        self._metadata_lock = threading.Lock()

    # -- HTTP ---------------------------------------------------------------

    def _get(self, url):
        """GET with retries. Returns (status_code, content_type, body)."""
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.status, r.headers.get("Content-Type", ""), r.read()
            except urllib.error.HTTPError as e:
                last = (e.code, e.headers.get("Content-Type", "") if e.headers else "", e.read())
                if e.code != 429 and e.code < 500:
                    return last
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = (None, "", repr(e).encode())
            if attempt < self.retries:
                time.sleep(self.backoff * 2 ** attempt + random.uniform(0, self.backoff / 2))
        return last

    # -- metadata -------------------------------------------------------------

    def _tile_metadata(self, x, y):
        key = (int(np.floor(x / 1000)), int(np.floor(y / 1000)))
        with self._metadata_lock:
            if key in self._metadata_cache:
                return self._metadata_cache[key]
        params = {
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": self.WFS_LAYER, "OUTPUTFORMAT": "application/json",
            "PROPERTYNAME": ",".join(self.METADATA_KEYS),
            "BBOX": "{},{},{},{},urn:ogc:def:crs:EPSG::2154".format(x - 0.5, y - 0.5, x + 0.5, y + 0.5),
        }
        code, _, body = self._get(self.WFS_URL + "?" + urllib.parse.urlencode(params))
        if code != 200:
            return {"metadata_error": "WFS HTTP {}".format(code)}  # not cached: may be transient
        import json
        try:
            features = json.loads(body).get("features", [])
        except ValueError:
            return {"metadata_error": "invalid WFS response"}
        meta = {}
        if features:
            # several editions may exist: keep the latest
            props = max((f["properties"] for f in features), key=lambda p: p.get("date_edition") or "")
            for k in self.METADATA_KEYS:
                v = props.get(k)
                if isinstance(v, str) and k.startswith("date") and v.endswith("Z"):
                    v = v[:-1]
                meta["tile" if k == "coordonnees_nw" else k] = v
        with self._metadata_lock:
            self._metadata_cache[key] = meta
        return meta

    # -- fetch ------------------------------------------------------------------

    def fetch(self, geometry):
        meta = {"source": "ign-lidar-hd", "product": self.product}
        minx, miny, maxx, maxy = geometry.bounds
        res = self.resolution
        minx, maxx = _snap(minx - self.margin, maxx + self.margin, res, self.grid_origin)
        miny, maxy = _snap(miny - self.margin, maxy + self.margin, res, self.grid_origin)
        width, height = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
        if width > self.MAX_PIXELS or height > self.MAX_PIXELS:
            return FetchResult(TOO_LARGE, None, meta)

        params = {
            "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": self.layer,
            "FORMAT": "image/geotiff", "STYLES": "", "CRS": "EPSG:2154",
            "BBOX": "{},{},{},{}".format(minx, miny, maxx, maxy),
            "WIDTH": width, "HEIGHT": height,
        }
        code, ctype, body = self._get(self.WMS_URL + "?" + urllib.parse.urlencode(params))
        if code != 200 or "tiff" not in ctype:
            meta.update(http_status=code, error=body[:300].decode("utf-8", "replace") if body else None)
            return FetchResult(SERVICE_ERROR, None, meta)

        with rasterio.MemoryFile(body) as mem, mem.open() as ds:
            z = _clean(ds.read(1), ds.nodata)
        # the CRS of the response is written as a degraded WKT: use the
        # requested one and rebuild the transform from the requested bbox
        transform = Affine(res, 0, minx, 0, -res, maxy)
        patch = DSMPatch(z, transform, self.crs)

        if self.fetch_metadata:
            c = geometry.centroid
            meta.update(self._tile_metadata(c.x, c.y))
        if not np.isfinite(z).any():
            return FetchResult(NO_COVERAGE, None, meta)
        return FetchResult(OK, patch, meta)


class LocalRasterService(DSMService):
    """
    Elevation rasters stored locally.

    Expected format: single-band GeoTIFFs whose pixel values are elevations
    (in metres), georeferenced in a projected CRS (Lambert-93 / EPSG:2154 for
    IGN products, e.g. LiDAR HD MNS tiles or a thumbnail exported from the
    WMS-R). Any file name is accepted; the folder is indexed by extent when
    the service is created. All rasters must share the same CRS. A polygon
    spanning several rasters is handled (the rasters are mosaicked).

    Parameters
    ----------
    folder : directory containing the rasters
    pattern : glob pattern of the raster files (searched recursively)
    margin : metres added around the polygon bounding box
    crs : CRS to assume for rasters without one

    `meta` keys: source ("local"), files (names of the rasters used), and
    tags (the GeoTIFF tags of the first raster, if any).
    """

    def __init__(self, folder, pattern="*.tif", margin=1.0, crs=None):
        self.folder = folder
        self.margin = margin
        paths = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
        if not paths:
            raise ValueError("no raster matching {!r} in {}".format(pattern, folder))
        self.paths, boxes, crss = [], [], set()
        self._info = []
        for p in paths:
            with rasterio.open(p) as ds:
                crs_ = ds.crs or (CRS.from_user_input(crs) if crs else None)
                if crs_ is None:
                    raise ValueError("{} has no CRS; pass `crs`".format(p))
                crss.add(crs_.to_wkt())
                self.paths.append(p)
                boxes.append(box(*ds.bounds))
                self._info.append((ds.transform, ds.res, ds.nodata))
        if len(crss) > 1:
            raise ValueError("all rasters must share the same CRS")
        self.crs = CRS.from_wkt(crss.pop())
        self._boxes = boxes
        self._tree = STRtree(boxes)

    def fetch(self, geometry):
        meta = {"source": "local"}
        minx, miny, maxx, maxy = geometry.bounds
        query = box(minx - self.margin, miny - self.margin, maxx + self.margin, maxy + self.margin)
        idx = sorted(self._tree.query(query, predicate="intersects"))
        if not idx:
            return FetchResult(NO_COVERAGE, None, meta)

        transform, (xres, yres), nodata = self._info[idx[0]]
        minx, maxx = _snap(query.bounds[0], query.bounds[2], xres, transform.c)
        miny, maxy = _snap(query.bounds[1], query.bounds[3], yres, transform.f)
        datasets = [rasterio.open(self.paths[i]) for i in idx]
        try:
            fill = -99999.0
            arr, out_transform = rasterio.merge.merge(
                datasets, bounds=(minx, miny, maxx, maxy), res=(xres, yres),
                nodata=fill, dtype="float64")
            z = _clean(arr[0], fill)
            if nodata is not None:
                z[arr[0] == nodata] = np.nan
            tags = datasets[0].tags()
        finally:
            for ds in datasets:
                ds.close()
        meta["files"] = [os.path.basename(self.paths[i]) for i in idx]
        if tags:
            meta["tags"] = tags
        if not np.isfinite(z).any():
            return FetchResult(NO_COVERAGE, None, meta)
        return FetchResult(OK, DSMPatch(z, out_transform, self.crs), meta)


def get_service(spec, **kwargs):
    """
    Builds a service from a specification:
      - a DSMService instance (returned as is)
      - "ign" / "ign-lidar-hd" -> IGNLidarService(**kwargs)
      - a path to a folder -> LocalRasterService(folder, **kwargs)
    """
    if isinstance(spec, DSMService):
        return spec
    if isinstance(spec, str) and spec.lower() in ("ign", "ign-lidar-hd", "lidar"):
        return IGNLidarService(**kwargs)
    if isinstance(spec, (str, os.PathLike)) and os.path.isdir(spec):
        return LocalRasterService(spec, **kwargs)
    raise ValueError("unknown DSM service: {!r}".format(spec))
