"""Catalogue of remote surface models: request building (offline) and live checks (opt-in)."""
import io
import os
import urllib.parse

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from pypvroof import WCSService, available_services, get_service
from pypvroof.catalog import SERVICES
from pypvroof.dsm import RemoteDSMService

network = pytest.mark.skipif(not os.environ.get("PYPVROOF_NETWORK_TESTS"),
                             reason="set PYPVROOF_NETWORK_TESTS=1 to query the remote services")


def _geotiff(z, x0, y0, res, crs="EPSG:28992"):
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                      crs=crs, transform=from_origin(x0, y0, res, res), nodata=-9999) as ds:
            ds.write(z.astype("float32"), 1)
        buf.write(mem.read())
    return buf.getvalue()


def test_catalogue_entries_build():
    table = available_services()
    assert set(table.name) == set(SERVICES)
    for name in SERVICES:
        svc = get_service(name)
        assert svc.crs is not None and svc.crs.is_projected


def test_services_at():
    from pypvroof.catalog import services_at
    assert services_at(2.18, 43.25) == ["fr-lidar", "fr-correl"]
    assert services_at(8.53, 47.39)[0] == "ch-swisssurface3d"
    assert services_at(5.10, 52.09)[0] == "nl-ahn"
    assert services_at(10.0, 40.0) == []


def test_wcs_request_urls():
    svc = WCSService("https://example.org/wcs", "dsm", 28992, 0.5, axis_labels=("E", "N"))
    q = urllib.parse.parse_qs(urllib.parse.urlparse(svc._url(0, 10, 5, 20, 10, 20)).query)
    assert q["SUBSET"] == ["E(0,5)", "N(10,20)"] and q["COVERAGEID"] == ["dsm"]
    svc = WCSService("https://example.org/wcs", "dsm", 2180, 0.5, version="1.0.0")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(svc._url(0, 10, 5, 20, 10, 20)).query)
    assert q["BBOX"] == ["0,10,5,20"] and q["CRS"] == ["EPSG:2180"] and q["WIDTH"] == ["10"]


def test_wcs_fetch_plain_and_multipart(monkeypatch):
    tif = _geotiff(np.arange(200.0).reshape(10, 20), 100, 210, 0.5)
    svc = WCSService("https://example.org/wcs", "dsm", 28992, 0.5, margin=0)
    monkeypatch.setattr(svc, "_get", lambda url: (200, "image/tiff", tif))
    fr = svc.fetch(box(100, 205, 110, 210))
    assert fr.status == "ok" and fr.patch.z.shape == (10, 20) and fr.patch.transform.c == 100
    multipart = (b'--wcs\r\nContent-Type: text/xml\r\n\r\n<gml/>\r\n--wcs\r\nContent-Type: image/tiff\r\n'
                 b'Content-Transfer-Encoding: binary\r\n\r\n' + tif + b'\r\n--wcs--\r\n')
    monkeypatch.setattr(svc, "_get", lambda url: (200, 'multipart/related; boundary="wcs"', multipart))
    assert svc.fetch(box(100, 205, 110, 210)).status == "ok"
    monkeypatch.setattr(svc, "_get", lambda url: (200, "text/xml", b"<ExceptionReport/>"))
    assert svc.fetch(box(100, 205, 110, 210)).status == "service_error"


@network
@pytest.mark.parametrize("name", sorted(SERVICES))
def test_live_service(name):
    x, y = SERVICES[name]["test_point"]
    fr = get_service(name).fetch(box(x - 6, y - 4, x + 6, y + 4))
    assert fr.status == "ok", fr.meta
    assert np.isfinite(fr.patch.z).mean() > 0.9
    assert abs(fr.patch.resolution[0] - SERVICES[name]["resolution"]) < 1e-6
