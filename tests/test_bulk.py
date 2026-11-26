"""Bulk API: inputs, outputs, metadata passthrough, MetadataExtraction integration."""
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import mapping

from pypvroof import MetadataExtraction, characterize_roofs, dsm
from pypvroof.geometry import reproject, to_geodataframe
from pypvroof.utils import azimuth_bounding_box
from synthetic import FunctionService, X0, Y0, gable, plane, rect


def lonlat(geom):
    return reproject(geom, "EPSG:2154", "EPSG:4326")


@pytest.fixture
def polys():
    return [lonlat(rect(X0 + 20 * i, Y0, 6, 4)) for i in range(5)]


def test_inputs_are_generic(polys, tmp_path):
    feats = [{"type": "Feature", "geometry": mapping(g), "properties": {"id": i}}
             for i, g in enumerate(polys)]
    fc = {"type": "FeatureCollection", "features": feats}
    path = tmp_path / "in.geojson"
    path.write_text(json.dumps(fc))
    gdf = gpd.GeoDataFrame({"id": range(5)}, geometry=polys, crs="EPSG:4326")
    gdf.to_parquet(tmp_path / "in.parquet")
    wkb = pd.DataFrame({"id": range(5), "geometry": [g.wkb for g in polys]})
    for data in (fc, feats, polys, gdf, str(path), str(tmp_path / "in.parquet"), wkb,
                 gdf.to_crs("EPSG:2154")):
        assert len(to_geodataframe(data)) == 5


def test_output_table(polys):
    svc = FunctionService(plane(30, 180), meta={"source": "synthetic", "acquisition": "2023"})
    gdf = gpd.GeoDataFrame({"name": list("abcde"), "tilt": 99.0}, geometry=polys, crs="EPSG:4326")
    out = characterize_roofs(gdf, svc, workers=3, id_column="name")
    assert len(out) == 5 and out.crs == gdf.crs
    assert list(out.parent_id) == list("abcde") and (out.part == 0).all()
    assert (out.status == "ok").all()
    assert np.allclose(out.tilt, 30, atol=0.5)
    assert (out.tilt_input == 99.0).all()  # clashing input columns are kept, renamed
    assert (out.meta_acquisition == "2023").all()
    assert np.allclose(out.surface_proj, 24, rtol=1e-3)
    assert np.allclose(out.surface, 24 / math.cos(math.radians(30)), rtol=1e-2)


def test_split_rows(polys):
    gs = [lonlat(rect(X0 + 20 * i, Y0, 8, 6)) for i in range(3)]
    svc = FunctionService(gable(35, Y0 + 3))
    out = characterize_roofs(gs, svc, split=True, workers=2)
    assert len(out) == 6
    assert out.groupby("parent_id").part.apply(list).tolist() == [[0, 1]] * 3
    unsplit = characterize_roofs(gs, svc, split=False, workers=2)
    assert len(unsplit) == 3 and (unsplit.status == "multimodal").all() and unsplit.inspect.all()


class Broken(dsm.DSMService):
    crs = FunctionService.crs

    def fetch(self, geometry):
        raise RuntimeError("boom")


class Down(dsm.DSMService):
    crs = FunctionService.crs

    def fetch(self, geometry):
        return dsm.FetchResult(dsm.SERVICE_ERROR, None, {"http_status": 503})


def test_service_failures_do_not_stop_bulk(polys):
    out = characterize_roofs(polys, Broken(), workers=2)
    assert (out.status == "error").all() and out.meta_error.str.contains("boom").all()
    out = characterize_roofs(polys, Down(), workers=2)
    assert (out.status == "service_error").all() and (out.meta_http_status == 503).all()


def test_metadata_extraction_theil_sen(polys):
    params = {"tilt-method": "theil-sen", "azimuth-method": "theil-sen",
              "regression-type": "constant", "dsm-service": FunctionService(plane(25, 135)),
              "workers": 2}
    ext = MetadataExtraction(p=params)
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "geometry": mapping(g), "properties": {}} for g in polys]}
    df = ext.extract_all_characteristics(fc)
    assert len(df) == 5 and (df.status == "ok").all()
    assert np.allclose(df.tilt, 25, atol=0.5) and np.allclose(df.azimuth, 135, atol=1.5)
    assert np.allclose(df.installed_capacity, df.surface / 6.5)
    lon, lat, tilt, az, kwp, surface = ext.extract_all_characteristics(fc["features"][0])
    assert tilt == pytest.approx(25, abs=0.5)


def test_metadata_extraction_theil_sen_needs_service():
    with pytest.raises(ValueError):
        MetadataExtraction(p={"tilt-method": "theil-sen"})


@pytest.mark.parametrize("angle,expected", [(0, [180]), (90, [90, 270]), (45, [135]), (-45, [225])])
def test_bounding_box_azimuth_convention(angle, expected):
    # rectangle whose long side is rotated by `angle` degrees from east-west;
    # the panels face the normal closest to south (east and west are equally
    # close for a north-south rectangle: either may be returned)
    from shapely import affinity
    g = affinity.rotate(rect(X0, Y0, 8, 3), angle, origin="center")
    az = azimuth_bounding_box(mapping(lonlat(g)))
    assert any(abs(az - e) <= 1.0 for e in expected), az
