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
import rasterio.errors
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
    z[(z < -1000) | (z > 1e5)] = np.nan  # sentinel values such as -9999 or float32 max
    return z


class RemoteDSMService(DSMService):
    """
    Base class of the services that query a remote server for each polygon.

    It provides HTTP GETs with retries (network errors, HTTP 429 and 5xx,
    exponential backoff with jitter), the snapping of the requested extent
    onto the native pixel grid, and the decoding of GeoTIFF responses
    (including WCS multipart responses).

    Attributes set by subclasses: `crs`, `source`, `resolution`,
    `grid_origin` (offset of the native pixel grid), `margin`.
    """

    MAX_PIXELS = 5000
    attribution = None

    def _init_http(self, timeout=30, retries=3, backoff=1.0, user_agent=None):
        from . import __version__
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.user_agent = user_agent or "pypvroof/{}".format(__version__)

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

    def _extent(self, geometry):
        """Requested extent snapped on the native grid, and its size in pixels."""
        minx, miny, maxx, maxy = geometry.bounds
        res = self.resolution
        minx, maxx = _snap(minx - self.margin, maxx + self.margin, res, self.grid_origin)
        miny, maxy = _snap(miny - self.margin, maxy + self.margin, res, self.grid_origin)
        return (minx, miny, maxx, maxy), int(round((maxx - minx) / res)), int(round((maxy - miny) / res))

    @staticmethod
    def _tiff_bytes(ctype, body):
        """GeoTIFF bytes of a response, or None (multipart WCS responses handled)."""
        if body is None:
            return None
        if "multipart" in ctype:
            import email
            msg = email.message_from_bytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
            for part in msg.walk():
                if "tiff" in part.get_content_type():
                    return part.get_payload(decode=True)
            return None
        return body if body[:4] in (b"II*\x00", b"MM\x00*") else None

    def _error(self, meta, code, body):
        meta.update(http_status=code, error=body[:300].decode("utf-8", "replace") if body else None)
        return FetchResult(SERVICE_ERROR, None, meta)


class IGNLidarService(RemoteDSMService):
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
    source = "ign-lidar-hd"
    attribution = "© IGN, LiDAR HD"

    def __init__(self, product="MNS", margin=1.0, resolution=0.5, grid_origin=0.25,
                 timeout=30, retries=3, backoff=1.0, fetch_metadata=True,
                 user_agent=None):
        product = product.upper()
        if product not in ("MNS", "MNT", "MNH"):
            raise ValueError("product must be MNS, MNT or MNH")
        self.product = product
        self.layer = "IGNF_LIDAR-HD_{}_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93".format(product)
        self.margin = margin
        self.resolution = resolution
        self.grid_origin = grid_origin
        self.fetch_metadata = fetch_metadata
        self._init_http(timeout, retries, backoff, user_agent)
        self._metadata_cache = {}
        self._metadata_lock = threading.Lock()

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
        meta = {"source": self.source, "product": self.product}
        (minx, miny, maxx, maxy), width, height = self._extent(geometry)
        res = self.resolution
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
            return self._error(meta, code, body)

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


class IGNCorrelService(IGNLidarService):
    """
    IGN "MNS Correl" surface model (50 cm), served by the Géoplateforme WMS-R.

    It is computed by correlation of the aerial images of the BD ORTHO, i.e.
    from the same images as the orthophotos: it is consistent in time and
    geometry with detections made on the BD ORTHO, and covers areas without
    LiDAR HD. Correlation is less precise than LiDAR, especially on low-texture
    surfaces (dark panels, uniform roofs) and near roof edges.

    Same parameters as IGNLidarService, without `product`; there is no
    acquisition metadata index (`fetch_metadata` is ignored).
    `meta` keys: source ("ign-mns-correl"), product.
    """

    source = "ign-mns-correl"
    attribution = "© IGN, MNS Correl"

    def __init__(self, margin=1.0, resolution=0.5, grid_origin=0.25, timeout=30,
                 retries=3, backoff=1.0, fetch_metadata=False, user_agent=None):
        super().__init__("MNS", margin=margin, resolution=resolution, grid_origin=grid_origin,
                         timeout=timeout, retries=retries, backoff=backoff,
                         fetch_metadata=False, user_agent=user_agent)
        self.product = "MNS-CORREL"
        self.layer = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES.MNS"


class WCSService(RemoteDSMService):
    """
    Elevation model served by an OGC Web Coverage Service (WCS 2.0.1 or
    1.0.0) returning GeoTIFFs. Most national LiDAR surface models in Europe
    are published this way; see `pypvroof.catalog` for ready-made presets.

    Parameters
    ----------
    url : endpoint of the WCS (without query string)
    coverage : coverage identifier (CoverageId in 2.0.1, coverage in 1.0.0)
    crs : CRS of the coverage (EPSG code or any pyproj input); polygons are
        reprojected to it
    resolution : native pixel size (metres)
    grid_origin : offset of the native pixel grid (0 for grids aligned on
        whole metres)
    version : "2.0.1" (subsets in native pixels) or "1.0.0" (bbox + size)
    axis_labels : labels of the (easting, northing) axes for 2.0.1 subsets,
        as given by DescribeCoverage (e.g. ("x", "y"), ("E", "N"))
    format : output format requested
    source : name reported in `meta["source"]`
    attribution : licence / attribution string reported in `meta`
    margin, timeout, retries, backoff, user_agent : as for IGNLidarService

    `meta` keys: source, coverage, attribution. On errors: http_status, error.
    """

    def __init__(self, url, coverage, crs, resolution, grid_origin=0.0, version="2.0.1",
                 axis_labels=("x", "y"), format="image/tiff", source=None, attribution=None,
                 extra_params=None, margin=1.0, timeout=30, retries=3, backoff=1.0,
                 user_agent=None):
        if version not in ("2.0.1", "1.0.0"):
            raise ValueError("version must be 2.0.1 or 1.0.0")
        self.url, self.coverage, self.version = url, coverage, version
        self.crs = CRS.from_user_input(crs)
        self.resolution, self.grid_origin, self.margin = resolution, grid_origin, margin
        self.axis_labels, self.format = axis_labels, format
        self.source = source or "wcs:{}".format(coverage)
        self.attribution = attribution
        self.extra_params = dict(extra_params or {})
        self._init_http(timeout, retries, backoff, user_agent)

    def _url(self, minx, miny, maxx, maxy, width, height):
        if self.version == "1.0.0":
            q = [("SERVICE", "WCS"), ("VERSION", "1.0.0"), ("REQUEST", "GetCoverage"),
                 ("COVERAGE", self.coverage), ("CRS", "EPSG:{}".format(self.crs.to_epsg())),
                 ("BBOX", "{},{},{},{}".format(minx, miny, maxx, maxy)),
                 ("WIDTH", width), ("HEIGHT", height), ("FORMAT", self.format)]
        else:
            ex, ny = self.axis_labels
            q = [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
                 ("COVERAGEID", self.coverage), ("FORMAT", self.format),
                 ("SUBSET", "{}({},{})".format(ex, minx, maxx)),
                 ("SUBSET", "{}({},{})".format(ny, miny, maxy))]
        q += list(self.extra_params.items())
        return self.url + "?" + urllib.parse.urlencode(q)

    def fetch(self, geometry):
        meta = {"source": self.source, "coverage": self.coverage}
        if self.attribution:
            meta["attribution"] = self.attribution
        (minx, miny, maxx, maxy), width, height = self._extent(geometry)
        if width > self.MAX_PIXELS or height > self.MAX_PIXELS:
            return FetchResult(TOO_LARGE, None, meta)
        code, ctype, body = self._get(self._url(minx, miny, maxx, maxy, width, height))
        data = self._tiff_bytes(ctype, body) if code == 200 else None
        if data is None:
            return self._error(meta, code, body)
        with rasterio.MemoryFile(data) as mem, mem.open() as ds:
            z = _clean(ds.read(1), ds.nodata)
            transform = ds.transform
            if transform.is_identity or transform.e > 0:  # no / south-up georeferencing
                transform = Affine((maxx - minx) / z.shape[1], 0, minx, 0, -(maxy - miny) / z.shape[0], maxy)
        if not np.isfinite(z).any():
            return FetchResult(NO_COVERAGE, None, meta)
        return FetchResult(OK, DSMPatch(z, transform, self.crs), meta)


class STACCogService(RemoteDSMService):
    """
    Elevation model published as Cloud-Optimized GeoTIFF tiles indexed by a
    STAC API (e.g. swisstopo swissSURFACE3D Raster). For each polygon, the
    tiles intersecting it are found with a STAC search (cached per tile
    footprint) and only the needed window is read (HTTP range requests).

    Parameters
    ----------
    stac_url : STAC API root (e.g. https://data.geo.admin.ch/api/stac/v0.9)
    collection : collection id
    crs : CRS of the tiles
    asset_filter : function (asset_key, asset_dict) -> bool selecting the
        GeoTIFF asset of an item
    tile_size : size (metres) of the tiles, used to cache the searches
    latest : when several items cover a tile (several acquisitions), keep
        the most recent one
    source, attribution, margin, timeout, retries, backoff, user_agent : as
        for WCSService

    `meta` keys: source, attribution, items (ids of the items used),
    datetime (acquisition date of the items, if given).
    """

    def __init__(self, stac_url, collection, crs, asset_filter, tile_size=1000.0, latest=True,
                 source=None, attribution=None, margin=1.0, timeout=30, retries=3, backoff=1.0,
                 user_agent=None):
        from pyproj import Transformer
        self.stac_url, self.collection = stac_url.rstrip("/"), collection
        self.crs = CRS.from_user_input(crs)
        self.asset_filter, self.tile_size, self.latest = asset_filter, tile_size, latest
        self.source = source or "stac:{}".format(collection)
        self.attribution = attribution
        self.margin = margin
        self._to_lonlat = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self._cache, self._lock = {}, threading.Lock()
        self._init_http(timeout, retries, backoff, user_agent)

    def _items(self, tx, ty):
        """Items covering the tile (tx, ty), from the STAC API (cached)."""
        key = (tx, ty)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        import json
        s = self.tile_size
        # a point well inside the tile, to avoid neighbours touching its edges
        lon, lat = self._to_lonlat.transform((tx + 0.5) * s, (ty + 0.5) * s)
        q = {"bbox": "{},{},{},{}".format(lon, lat, lon, lat), "limit": 50}
        url = "{}/collections/{}/items?{}".format(self.stac_url, self.collection, urllib.parse.urlencode(q))
        code, _, body = self._get(url)
        if code != 200:
            return None  # not cached: may be transient
        items = []
        for f in json.loads(body).get("features", []):
            href = next((a["href"] for k, a in f.get("assets", {}).items() if self.asset_filter(k, a)), None)
            if href:
                items.append((f["properties"].get("datetime") or "", f["id"], href))
        if self.latest and items:
            items = [max(items)]
        with self._lock:
            self._cache[key] = items
        return items

    def fetch(self, geometry):
        meta = {"source": self.source}
        if self.attribution:
            meta["attribution"] = self.attribution
        minx, miny, maxx, maxy = geometry.buffer(self.margin).bounds
        s = self.tile_size
        found = []
        for tx in range(int(np.floor(minx / s)), int(np.floor(maxx / s)) + 1):
            for ty in range(int(np.floor(miny / s)), int(np.floor(maxy / s)) + 1):
                items = self._items(tx, ty)
                if items is None:
                    meta["error"] = "STAC search failed"
                    return FetchResult(SERVICE_ERROR, None, meta)
                found += items
        if not found:
            return FetchResult(NO_COVERAGE, None, meta)
        meta["items"] = [i for _, i, _ in found]
        dates = sorted({d[:10] for d, _, _ in found if d})
        if dates:
            meta["datetime"] = dates[-1] if len(dates) == 1 else dates
        env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY=str(self.retries),
                           GDAL_HTTP_RETRY_DELAY=str(self.backoff), GDAL_HTTP_USERAGENT=self.user_agent)
        try:
            with env:
                datasets = [rasterio.open("/vsicurl/" + h) for _, _, h in found]
                try:
                    t = datasets[0].transform
                    xres, yres = datasets[0].res
                    bx = _snap(minx, maxx, xres, t.c)
                    by = _snap(miny, maxy, yres, t.f)
                    fill = -99999.0
                    arr, out_transform = rasterio.merge.merge(
                        datasets, bounds=(bx[0], by[0], bx[1], by[1]), res=(xres, yres),
                        nodata=fill, dtype="float64")
                    nodata = datasets[0].nodata
                finally:
                    for ds in datasets:
                        ds.close()
        except rasterio.errors.RasterioIOError as e:
            meta["error"] = str(e)[:300]
            return FetchResult(SERVICE_ERROR, None, meta)
        z = _clean(arr[0], fill)
        if nodata is not None:
            z[arr[0] == nodata] = np.nan
        if not np.isfinite(z).any():
            return FetchResult(NO_COVERAGE, None, meta)
        return FetchResult(OK, DSMPatch(z, out_transform, self.crs), meta)


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
      - a name from the catalogue (`pypvroof.catalog`), e.g. "fr-lidar"
        (IGNLidarService), "fr-correl" (IGNCorrelService), "nl-ahn",
        "uk-england", "ch-swisssurface3d"; `**kwargs` override its settings
      - a path to a folder -> LocalRasterService(folder, **kwargs)
    """
    from . import catalog
    if isinstance(spec, DSMService):
        return spec
    if isinstance(spec, str) and spec.lower() in catalog.SERVICES:
        return catalog.service(spec, **kwargs)
    if isinstance(spec, (str, os.PathLike)) and os.path.isdir(spec):
        return LocalRasterService(spec, **kwargs)
    raise ValueError("unknown DSM service: {!r} (see pypvroof.catalog.SERVICES)".format(spec))
