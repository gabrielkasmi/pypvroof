# Contributing to PyPVRoof

The most useful contribution is **adding a surface model (DSM) for a country
or region** that is not covered yet. This guide walks through it. Bug reports
and fixes are welcome too: open an issue with a minimal example (the polygon
as GeoJSON, the service used, the output).

## Development setup

```bash
git clone https://github.com/gabrielkasmi/pypvroof.git
cd pypvroof
pip install -e ".[dev,viz]"
pytest                                   # offline tests (synthetic data)
PYPVROOF_NETWORK_TESTS=1 pytest          # also query every remote service
```

## Adding a service

### 1. Check that the data fits

pypvroof measures roof planes, so the surface model must:

- be a **surface model** (DSM: first return, with buildings), not a terrain
  model (DTM);
- have a resolution of **1 m or finer** (0.5 m is better: at 1 m, polygons
  under about 12 m² cannot be measured);
- be served as **elevation values** in metres (a float GeoTIFF), not as a
  rendered image (colours, hillshade): a WMS that only returns PNG/JPEG maps
  does not work;
- be **queryable on an arbitrary small extent** (a few tens of metres), with a
  response time of a few seconds at most: one request is made per polygon;
- have a **licence that allows reuse** (open data), and no mandatory API key.
  Services needing a free token can still be used through
  `WCSService(..., extra_params={"token": ...})`, but are not added to the
  catalogue.

National mapping agencies usually publish these as an **OGC WCS** (Web
Coverage Service). Some publish **Cloud-Optimized GeoTIFF tiles indexed by a
STAC API** instead. Look for the dataset on the national geoportal or the
INSPIRE geoportal (theme "Elevation"), and find the "download service" link.

### 2. Find the parameters (WCS)

Ask the service what it serves:

```
<url>?SERVICE=WCS&REQUEST=GetCapabilities&VERSION=2.0.1
```

Note the **coverage identifier** of the DSM (`<wcs:CoverageId>`) and the
supported versions. Then describe that coverage:

```
<url>?SERVICE=WCS&REQUEST=DescribeCoverage&VERSION=2.0.1&COVERAGEID=<id>
```

and note:

| In the response | Parameter |
|---|---|
| `srsName="…/EPSG/0/28992"` | `crs` (here 28992). It must be projected, in metres |
| `axisLabels="x y"` | `axis_labels`, in the order (easting, northing): e.g. `("x", "y")` or `("E", "N")` |
| `offsetVector` (e.g. `0.5 0`) | `resolution` (here 0.5) |
| `lowerCorner` | `grid_origin`: the remainder of the corner coordinates divided by the resolution, usually 0 |

If 2.0.1 does not work (errors, closed connections, axis order problems), try
`version="1.0.0"`, which takes a bbox and a size instead of subsets.

Finally, request a small extent on a building and check the values:

```
<url>?SERVICE=WCS&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID=<id>&FORMAT=image/tiff&SUBSET=x(121400,121430)&SUBSET=y(487400,487430)
```

The response should be a GeoTIFF with plausible elevations in metres. Some
servers wrap the GeoTIFF in a multipart response: this is handled.

### 3. Add the entry

Add an entry to `SERVICES` in [`pypvroof/catalog.py`](pypvroof/catalog.py).
Name it `<country code>-<product>` in lower case (e.g. `nl-ahn`, `de-nw`):

```python
"xx-product": {
    "kind": "wcs",
    "region": "Country (region)",
    "product": "Name of the surface model",
    "resolution": 0.5,
    "crs": 28992,
    "licence": "CC BY 4.0",
    "test_point": (121400, 487400),   # (x, y) in `crs`, on a building roof
    "kwargs": dict(
        url="https://example.org/wcs",
        coverage="dsm_05m",
        axis_labels=("x", "y"),
        attribution="© Producer, licence",   # reported in the outputs' meta
        # version="1.0.0", format="GeoTIFF", timeout=90, grid_origin=0.25 ...
    ),
},
```

For a STAC + COG service, use `"kind": "stac"` with `stac_url`, `collection`
and an `asset_filter` selecting the GeoTIFF asset of each item (see the
`ch-swisssurface3d` entry). If the service fits neither, write a subclass of
`pypvroof.dsm.RemoteDSMService` (it provides retries, grid snapping and
GeoTIFF decoding; see `IGNLidarService`) and use `"kind": "class"`.

Then add a row to the table of covered countries in the README.

### 4. Test it

```bash
PYPVROOF_NETWORK_TESTS=1 pytest tests/test_catalog.py -k xx-product
```

checks that the service returns a valid grid at the test point, at the
declared resolution. Then check, on a few real buildings, that the estimates
make sense and that the grid is aligned with the polygons:

```python
from pypvroof import characterize_roofs
from pypvroof.viz import plot_polygon

out = characterize_roofs("buildings.geojson", "xx-product")   # e.g. OSM footprints
print(out[["tilt", "azimuth", "status"]])
fig = plot_polygon(out.iloc[[0]], services=("xx-product",))   # the roof should sit inside the polygon
```

Most pitched roofs should come out `ok` or `multimodal` (gable roofs), flat
roofs `flat`, with tilts between 15° and 50°. A grid shifted by a few metres,
or tilts close to 60°, usually mean a wrong CRS, axis order or grid origin.

### 5. Open a pull request

Describe in the pull request:

- the producer, the product and its acquisition period;
- the licence (with a link);
- the checks you made (network test, a few buildings, response time).

## Code conventions

- Services know nothing about roofs: they return the elevation grid and a
  `meta` dict, and never apply fallbacks.
- Azimuths are in degrees, 0 = north, clockwise.
- New behaviour comes with a test: offline tests on synthetic data in
  `tests/`, network tests behind `PYPVROOF_NETWORK_TESTS`.
- Keep the dependencies light (numpy, shapely, pyproj, rasterio, geopandas).
