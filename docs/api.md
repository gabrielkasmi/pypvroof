# PyPVRoof API Documentation

## MetadataExtraction

The main class for extracting PV roof characteristics.

### Initialization

```python
from pypvroof import MetadataExtraction

extraction = MetadataExtraction(p=params)
```

#### Parameters

- `p` (dict): Dictionary of parameters for the extraction process
- `lut` (str or dict, optional, **deprecated**): custom look-up table for
  `tilt-method: "lut"`. The look-up table is deprecated (noisy, France only)
  and will be removed; use `"theil-sen"` with a surface model, or `"constant"`.

### Methods

#### extract_all_characteristics

```python
characteristics = extraction.extract_all_characteristics(data, crs=None)
```

Extracts all characteristics (azimuth, tilt, surface, installed capacity).

**Parameters:**
- `data`: a single GeoJSON feature (Polygon / MultiPolygon, lon/lat), or anything accepted by the bulk API (FeatureCollection, GeoDataFrame, list of features, file path...)
- `crs`: CRS of inputs that do not carry one (default EPSG:4326)

**Returns:**
- for a single feature: a tuple `(lon, lat, tilt, azimuth, installed_capacity, surface)`
- otherwise: a DataFrame with these columns, one row per polygon. With the `theil-sen` method, the columns `status`, `inspect` and `meta_*` are added and the polygons are processed concurrently (parameter `workers`, default 8).

#### compute_azimuth

```python
azimuth = extraction.compute_azimuth(polygon)
```

Computes the azimuth angle for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature

**Returns:**
- float: The computed azimuth angle

#### compute_tilt

```python
tilt = extraction.compute_tilt(polygon)
```

Computes the tilt angle for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature

**Returns:**
- float: The computed tilt angle

#### compute_surface

```python
surface = extraction.compute_surface(polygon, tilt=tilt)
```

Computes the surface area for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature
- `tilt` (float): The tilt angle of the surface

**Returns:**
- float: The computed surface area

#### compute_installed_capacity

```python
capacity = extraction.compute_installed_capacity(polygon, tilt=tilt, surface=surface)
```

Computes the installed capacity for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature
- `tilt` (float): The tilt angle of the surface
- `surface` (float): The surface area

**Returns:**
- float: The computed installed capacity 

## Azimuth convention

Azimuths are expressed in degrees, **0 = north, clockwise** (90 = east,
180 = south, 270 = west): the direction the panels face. The `bounding-box`
method returns the normal to the long side of the polygon that is closest to
south (it cannot tell which side of the rectangle the panels face).

## DSM services and Theil-Sen

### characterize_roofs

```python
from pypvroof import characterize_roofs
out = characterize_roofs(polygons, service="fr-lidar", crs=None, params=None, split=False,
                         workers=8, id_column=None, meta_prefix="meta_", progress=None)
```

- `polygons`: GeoDataFrame / GeoSeries, DataFrame with a `geometry` column (shapely or WKB), GeoJSON dict, list of features or shapely geometries, or a path (GeoJSON, GeoPackage, (Geo)Parquet, ...)
- `service`: a `DSMService`, `"fr-lidar"` (LiDAR HD), `"fr-correl"` (MNS Correl), or a folder of GeoTIFFs
- `params`: dict or `TheilSenParams` (see below)
- `split`: split polygons covering several roof planes into homogeneous parts
- `workers`: polygons processed concurrently (threads). Mind the usage limits of remote services.
- `id_column`: input column copied into `parent_id` (default: the input index)

Output: a GeoDataFrame in the CRS of the input, with the input columns
(suffixed `_input` when they clash), `parent_id`, `part`, and:

| column | content |
|---|---|
| `tilt`, `azimuth` | degrees (NaN when not measured; azimuth NaN on flat roofs) |
| `status` | see below |
| `inspect` | True when a human should look at the polygon (multimodal) |
| `surface_proj`, `surface` | projected area (m²), area corrected for the tilt |
| `n_pixels`, `n_modes`, `primary_share`, `inlier_frac`, `rmse`, `roughness`, `offset_suspected`, `modes` | diagnostics |
| `meta_*` | metadata returned by the service, passed through as is |

Statuses:

| status | meaning |
|---|---|
| `ok` | one roof plane, tilt and azimuth measured |
| `flat` | flat roof (tilt < `flat_tilt`), azimuth undefined. A high `roughness` hints at racks |
| `multimodal` | several roof planes under the polygon: no value, `inspect=True` (or split) |
| `too_few_points` | too few valid pixels (very small polygon) |
| `fit_failed` | no plane could be fitted |
| `no_coverage` | the service has no data here |
| `service_error` | the service failed (after retries); retrying later may work |
| `too_large` | polygon too large for a single request |
| `error` | unexpected exception (see `meta_error`) |

pypvroof never applies a fallback: deciding what to do with a status (use a
another service, heuristics, retry, inspect, check the building existed at the LiDAR
acquisition date from `meta_date_fin_acquisition`...) is up to the caller.

### Services

A service retrieves the elevation grid under a polygon and returns a
`FetchResult(status, patch, meta)`. `meta` is a free-form dict whose content
depends on the service; pypvroof does not interpret it.

**`IGNLidarService(product="MNS", margin=1.0, timeout=30, retries=3, backoff=1.0, fetch_metadata=True)`**:
IGN LiDAR HD elevation models through the Géoplateforme WMS-R, as raw float32
GeoTIFFs at the native 50 cm grid. `meta`: `source`, `product`, and the
acquisition metadata of the LiDAR tile from the WFS index (`tile`,
`date_debut_acquisition`, `date_fin_acquisition`, `date_edition`,
`code_mission`), cached per 1 km tile.

**`IGNCorrelService(margin=1.0, timeout=30, retries=3, backoff=1.0)`**:
IGN "MNS Correl" (50 cm), computed by correlation of the BD ORTHO aerial
images, through the same WMS-R. It covers areas without LiDAR HD and is
consistent with detections made on the BD ORTHO (same images, same date), but
it is less precise. On 1,000 random detections where both were available:
tilt within 5° of LiDAR for 56% of them (mean absolute difference 6.6°), azimuth within 15° for 87%. `meta`: `source`, `product`
(no acquisition metadata index).

**`LocalRasterService(folder, pattern="*.tif", margin=1.0, crs=None)`**:
local rasters. Expected format: single-band GeoTIFFs whose pixel values are
elevations in metres (a DSM such as the LiDAR HD MNS, or a thumbnail exported
from the WMS-R), georeferenced in a projected CRS (Lambert-93 / EPSG:2154 for
IGN products). Any file name; the folder is indexed by extent; all rasters
must share the same CRS; polygons spanning several rasters are handled.
`meta`: `source`, `files`, `tags` (GeoTIFF tags of the first raster).

**`WCSService(url, coverage, crs, resolution, grid_origin=0, version="2.0.1", axis_labels=("x", "y"), format="image/tiff", source=None, attribution=None, margin=1.0, ...)`**:
any surface model served by an OGC WCS (2.0.1 or 1.0.0) returning GeoTIFFs
(multipart WCS responses are handled). `meta`: `source`, `coverage`, `attribution`.

**`STACCogService(stac_url, collection, crs, asset_filter, tile_size=1000, latest=True, ...)`**:
surface models published as Cloud-Optimized GeoTIFF tiles indexed by a STAC
API; only the window under the polygon is read. When several acquisitions
cover a tile, the latest is used. `meta`: `source`, `attribution`, `items`, `datetime`.

**Catalogue** (`pypvroof.catalog`): ready-made presets, usable by name
everywhere a service is expected (`characterize_roofs(polys, "nl-ahn")`,
`"dsm-service": "uk-england"`, `plot_polygon(..., services=("de-nw",))`).
`available_services()` returns the table below with CRS and licences. Each
entry was checked on live requests (see `tests/test_catalog.py`, run with
`PYPVROOF_NETWORK_TESTS=1`).

| Name | Region | Product | Resolution | CRS | Protocol |
|---|---|---|---|---|---|
| `fr-lidar` | France | IGN LiDAR HD MNS | 0.5 m | EPSG:2154 | WMS-R |
| `fr-correl` | France | IGN MNS Correl | 0.5 m | EPSG:2154 | WMS-R |
| `nl-ahn` | Netherlands | AHN DSM (PDOK) | 0.5 m | EPSG:28992 | WCS 2.0.1 |
| `be-flanders` | Belgium (Flanders) | DHMV II DSM (2013-2015) | 1 m | EPSG:31370 | WCS 2.0.1 (multipart) |
| `uk-england` | England | Environment Agency LiDAR composite DSM (first return) | 1 m | EPSG:27700 | WCS 2.0.1 |
| `de-nw` | Germany (NRW) | DOM1 | 1 m | EPSG:25832 | WCS 2.0.1 |
| `de-mv` | Germany (M-V) | DOM1 | 1 m | EPSG:25833 | WCS 2.0.1 |
| `pl` | Poland | GUGiK NMPT | 0.5 m | EPSG:2180 | WCS 1.0.0 (~7 s per request) |
| `no` | Norway | Nasjonal høydemodell DOM | 1 m | EPSG:25833 | WCS 1.0.0 |
| `ch-swisssurface3d` | Switzerland | swissSURFACE3D Raster | 0.5 m | EPSG:2056 | STAC + COG |

Not included (checked in November 2026): Denmark (Danmarks Højdemodel needs a
Dataforsyningen token: usable with `WCSService(..., extra_params={"token": ...})`),
Wallonia (rendered maps only, no elevation values), Czech Republic (DMP 1G,
2 m from a TIN), Spain (5 m surface model only), Sachsen-Anhalt (WCS
returning errors).

To add a region, add an entry to `pypvroof/catalog.py` with a test point on a
building. Fully custom services subclass `DSMService`, set `crs` and implement
`fetch(geometry)`.

### TheilSenParams

The estimator fits planes on many random subsets of pixels (exact planes
through 3 points by default), finds the modes of their distribution in
gradient space, assigns each pixel to the closest plane and refits each plane
on its pixels. A secondary plane that is too steep (`max_tilt`), too small
(`min_part_area`), or a band along the polygon's edge (`edge_band`,
`edge_frac`) while the main one dominates (`dominant_share`) is treated as a
misregistration or an obstacle (`offset_suspected=True`); otherwise the
polygon is `multimodal`.

Main parameters (defaults): `n_subsets=2000`, `subset_size=3`,
`subset_radius=3` m (pixels of a subset drawn within this distance, so that
subsets stay on one pan), `min_pixels=12`, `erosion=0.5` m, `bandwidth=0.05`
(gradient units), `inlier_tol=0.15` m, `min_mode_share=0.1` (or
`min_plane_area=8` m² on large polygons), `max_planes=16` (parallel pans
included), `same_tilt=5` and `same_azimuth=15` (planes with the same
orientation are not multimodal), `max_tilt=60`, `flat_tilt=5`,
`min_part_area=3` m², `dominant_share=0.75`, `edge_band=1` m,
`edge_frac=0.8`, `seed=42`. See `pypvroof/theilsen.py`.
