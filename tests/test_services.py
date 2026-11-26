"""DSM services: local GeoTIFFs (offline) and the IGN WMS-R (opt-in network test)."""
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pypvroof import IGNLidarService, LocalRasterService, characterize_roofs
from pypvroof.dsm import get_service
from synthetic import X0, Y0, plane, rect


def write_tile(path, x0, y0, size=100, res=0.5, func=None, nodata=-9999.0, tags=None):
    n = int(size / res)
    t = from_origin(x0, y0 + size, res, res)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    x = x0 + (cols + 0.5) * res
    y = y0 + size - (rows + 0.5) * res
    z = func(x, y).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs="EPSG:2154", transform=t, nodata=nodata) as ds:
        ds.write(z, 1)
        if tags:
            ds.update_tags(**tags)


@pytest.fixture
def tiles(tmp_path):
    f = plane(30, 180)
    write_tile(tmp_path / "a.tif", X0, Y0, func=f, tags={"acquisition": "2023-03-03"})
    write_tile(tmp_path / "b.tif", X0 + 100, Y0, func=f)
    return tmp_path


def test_local_service_single_tile(tiles):
    svc = LocalRasterService(str(tiles))
    fr = svc.fetch(rect(X0 + 10, Y0 + 10, 6, 4))
    assert fr.status == "ok"
    assert fr.meta["files"] == ["a.tif"]
    assert fr.meta["tags"]["acquisition"] == "2023-03-03"
    assert fr.patch.resolution == (0.5, 0.5)
    assert np.isfinite(fr.patch.z).all()


def test_local_service_mosaic_across_tiles(tiles):
    svc = LocalRasterService(str(tiles))
    geom = rect(X0 + 97, Y0 + 10, 6, 4)  # straddles a.tif / b.tif
    fr = svc.fetch(geom)
    assert fr.status == "ok" and sorted(fr.meta["files"]) == ["a.tif", "b.tif"]
    out = characterize_roofs([geom], svc, crs="EPSG:2154", workers=1)
    assert out.status.iloc[0] == "ok"
    assert out.tilt.iloc[0] == pytest.approx(30, abs=0.5)


def test_local_service_no_coverage(tiles):
    svc = LocalRasterService(str(tiles))
    assert svc.fetch(rect(X0 + 1000, Y0, 6, 4)).status == "no_coverage"
    out = characterize_roofs([rect(X0 + 1000, Y0, 6, 4)], svc, crs="EPSG:2154", workers=1)
    assert out.status.iloc[0] == "no_coverage" and np.isnan(out.tilt.iloc[0])


def test_local_service_nodata(tmp_path):
    write_tile(tmp_path / "void.tif", X0, Y0, func=lambda x, y: np.full_like(x, -9999.0))
    assert LocalRasterService(str(tmp_path)).fetch(rect(X0 + 10, Y0 + 10, 6, 4)).status == "no_coverage"


def test_get_service(tiles):
    assert isinstance(get_service(str(tiles)), LocalRasterService)
    assert isinstance(get_service("ign"), IGNLidarService)
    with pytest.raises(ValueError):
        get_service("nowhere")


network = pytest.mark.skipif(not os.environ.get("PYPVROOF_NETWORK_TESTS"),
                             reason="set PYPVROOF_NETWORK_TESTS=1 to query the IGN services")


@network
def test_ign_service_paris():
    svc = IGNLidarService()
    fr = svc.fetch(rect(651400, 6862400, 10, 10))
    assert fr.status == "ok"
    assert fr.patch.resolution == (0.5, 0.5)
    # native grid: pixel edges at x.25 / x.75
    assert (fr.patch.transform.c - 0.25) % 0.5 == pytest.approx(0)
    assert fr.meta["date_fin_acquisition"].startswith("2023")
    assert fr.meta["tile"] == "0651-6863"


@network
def test_ign_service_no_coverage():
    fr = IGNLidarService().fetch(rect(100000, 6800000, 10, 10))  # Atlantic
    assert fr.status == "no_coverage"
