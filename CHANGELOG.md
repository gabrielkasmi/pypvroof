# Changelog

## 0.2.1 (2026-11-27)

- Licence: the LICENSE file now holds the MIT licence, as declared in the
  package metadata (it still held the GPL v3 text in 0.2.0).

## 0.2.0 (2026-11-27)

### Added
- Tilt and azimuth measured on a DSM with a new Theil-Sen-type estimator
  (`pypvroof.theilsen`): plane fits on random pixel subsets, mode detection,
  pixel assignment, robust to misregistered polygons, chimneys and edges;
  statuses (`ok`, `flat`, `multimodal`, ...) and diagnostics.
- DSM services (`pypvroof.dsm`): `IGNLidarService` (IGN LiDAR HD through the
  Géoplateforme WMS-R, acquisition metadata from the WFS index) and
  `IGNCorrelService` (IGN MNS Correl, derived from the BD ORTHO images) and
  `LocalRasterService` (folder of GeoTIFFs). Services return free-form
  metadata that is passed through to the output.
- Bulk API `characterize_roofs`: generic inputs, concurrent processing,
  GeoDataFrame output, optional splitting of multimodal polygons.

- `pypvroof.viz.plot_polygon` (extra `viz`): orthophoto and elevation models
  under a polygon, with the estimates.
- Gradio demo (`app/`), deployable as a Hugging Face Space: draw a polygon on
  a map (computed on the fly, with the split roof planes drawn back on the map)
  or upload a file; the surface model is chosen from the location
  (`catalog.services_at`); national orthophotos as map layers; installed
  capacity optional ("Extract PV characteristics", off for roof pans).
- GeoJSON files with unclosed rings and self-intersecting polygons are
  accepted (repaired with `make_valid`).

- README simplified, with a new pipeline diagram (`docs/assets/pipeline.png`,
  drawn by `docs/make_diagram.py`) and a link to the Hugging Face Space.

- Surface models across Europe: generic `WCSService` (WCS 2.0.1 / 1.0.0) and
  `STACCogService` (STAC + Cloud-Optimized GeoTIFFs), and a catalogue of
  presets usable by name (`pypvroof.catalog`, `available_services()`):
  Netherlands (AHN), Flanders (DHMV II), England (Environment Agency),
  North Rhine-Westphalia and Mecklenburg-Vorpommern (DOM1), Poland (NMPT),
  Norway (NHM DOM), Switzerland (swissSURFACE3D).
- `IGNLidarService` now derives from `RemoteDSMService` (shared HTTP retries,
  grid snapping, GeoTIFF decoding).

- Continuous integration (GitHub Actions): offline tests on Python 3.10 and
  3.12; network tests on manual runs.
- `pyarrow` is now a dependency (parquet inputs).

### Removed
- The hands-on notebook (superseded by the demo app) and the old flowchart.

### Changed
- **Azimuth convention: 0 = north, clockwise** (was 0 = south).
- Theil-Sen: subsets drawn locally (`subset_radius`) and parallel pans
  detected, so that large roofs with many pans (repeated gables, sheds) are
  multimodal and split into each pan, instead of being reported flat. Planes
  sharing an orientation (roof at several heights) are not multimodal.
- The `theil-sen` method of `MetadataExtraction` uses the new estimator and a
  DSM service (`dsm-service`; `raster-folder` still accepted). The old
  implementation (8-bit reload of the DSM, resolution assumed to be 2 m) is
  removed, as are the GDAL / OpenCV / matplotlib / scikit-learn dependencies.
- `MetadataExtraction.extract_all_characteristics` accepts any input of the
  bulk API; installed capacity coefficients are fitted once.

## 0.1.2 (not released on PyPI, included in 0.2.0)

- Production fixes: relative imports, LUT nearest-cell fallback for NaN
  cells, lazy GDAL import, CSV shipped in the package, complete dependencies.
