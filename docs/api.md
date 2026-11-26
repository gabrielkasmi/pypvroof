# PyPVRoof API Documentation

## MetadataExtraction

The main class for extracting PV roof characteristics.

### Initialization

```python
from pypvroof import MetadataExtraction

# Basic initialization with default LUT-France
extraction = MetadataExtraction(p=params)

# With custom LUT
extraction = MetadataExtraction(p=params, lut=custom_lut)
```

#### Parameters

- `p` (dict): Dictionary of parameters for the extraction process
- `lut` (str or dict, optional): Path to a custom lookup table JSON file or a dictionary containing the lookup table. If None, the default LUT-France will be used.

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
out = characterize_roofs(polygons, service="ign", crs=None, params=None, split=False,
                         workers=8, id_column=None, meta_prefix="meta_", progress=None)
```

- `polygons`: GeoDataFrame / GeoSeries, DataFrame with a `geometry` column (shapely or WKB), GeoJSON dict, list of features or shapely geometries, or a path (GeoJSON, GeoPackage, (Geo)Parquet, ...)
- `service`: a `DSMService`, `"ign"`, or a folder of GeoTIFFs
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
LUT / model, retry, inspect, check the building existed at the LiDAR
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

**`LocalRasterService(folder, pattern="*.tif", margin=1.0, crs=None)`**:
local rasters. Expected format: single-band GeoTIFFs whose pixel values are
elevations in metres (a DSM such as the LiDAR HD MNS, or a thumbnail exported
from the WMS-R), georeferenced in a projected CRS (Lambert-93 / EPSG:2154 for
IGN products). Any file name; the folder is indexed by extent; all rasters
must share the same CRS; polygons spanning several rasters are handled.
`meta`: `source`, `files`, `tags` (GeoTIFF tags of the first raster).

Custom services subclass `DSMService`, set `crs` and implement `fetch(geometry)`.

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
`min_pixels=12`, `erosion=0.5` m, `bandwidth=0.05` (gradient units),
`inlier_tol=0.15` m, `min_mode_share=0.1`, `max_tilt=60`, `flat_tilt=5`,
`min_part_area=3` m², `dominant_share=0.75`, `edge_band=1` m,
`edge_frac=0.8`, `seed=42`. See `pypvroof/theilsen.py`.
