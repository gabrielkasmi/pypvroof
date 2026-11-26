# Changelog

## 0.2.0 (unreleased)

### Added
- Tilt and azimuth measured on a DSM with a new Theil-Sen-type estimator
  (`pypvroof.theilsen`): plane fits on random pixel subsets, mode detection,
  pixel assignment, robust to misregistered polygons, chimneys and edges;
  statuses (`ok`, `flat`, `multimodal`, ...) and diagnostics.
- DSM services (`pypvroof.dsm`): `IGNLidarService` (IGN LiDAR HD through the
  Géoplateforme WMS-R, acquisition metadata from the WFS index) and
  `LocalRasterService` (folder of GeoTIFFs). Services return free-form
  metadata that is passed through to the output.
- Bulk API `characterize_roofs`: generic inputs, concurrent processing,
  GeoDataFrame output, optional splitting of multimodal polygons.

### Changed
- **Azimuth convention: 0 = north, clockwise** (was 0 = south).
- The `theil-sen` method of `MetadataExtraction` uses the new estimator and a
  DSM service (`dsm-service`; `raster-folder` still accepted). The old
  implementation (8-bit reload of the DSM, resolution assumed to be 2 m) is
  removed, as are the GDAL / OpenCV / matplotlib / scikit-learn dependencies.
- `MetadataExtraction.extract_all_characteristics` accepts any input of the
  bulk API; installed capacity coefficients are fitted once.

## 0.1.2 (unreleased)

- Production fixes: relative imports, LUT nearest-cell fallback for NaN
  cells, lazy GDAL import, CSV shipped in the package, complete dependencies.
