# -*- coding: utf-8 -*-
"""
Catalogue of open surface models (DSM) that can be queried on the fly.

Each entry is a preset of a generic service (`WCSService`, `STACCogService`)
or a dedicated class. Use the names with `characterize_roofs(..., service=name)`
or `get_service(name)`; `available_services()` lists them.

Adding a region is usually one entry here: a WCS (or STAC + COG) endpoint
serving elevations (metres) as GeoTIFF, its CRS, native resolution and
licence, plus a test point on a building (used by `tests/test_catalog.py`).
See CONTRIBUTING.md.
"""

from . import dsm

# Each entry: kind ("wcs", "stac", "class"), human-readable fields, and the
# keyword arguments of the service. "bbox" is the approximate extent of the
# coverage (lon_min, lat_min, lon_max, lat_max), used to pick a service from a
# location. "test_point" is a point (x, y, in `crs`) on buildings, used by the
# network tests.
SERVICES = {
    "fr-lidar": {
        "bbox": (-5.3, 41.3, 9.7, 51.2),
        "kind": "class", "cls": "IGNLidarService",
        "region": "France", "product": "IGN LiDAR HD MNS", "resolution": 0.5,
        "crs": 2154, "licence": "Etalab Open Licence 2.0", "test_point": (842200, 6519700),
    },
    "fr-correl": {
        "bbox": (-5.3, 41.3, 9.7, 51.2),
        "kind": "class", "cls": "IGNCorrelService",
        "region": "France", "product": "IGN MNS Correl (from BD ORTHO images)", "resolution": 0.5,
        "crs": 2154, "licence": "Etalab Open Licence 2.0", "test_point": (842200, 6519700),
    },
    "nl-ahn": {
        "bbox": (3.3, 50.7, 7.3, 53.6),
        "kind": "wcs", "region": "Netherlands", "product": "AHN DSM (PDOK)", "resolution": 0.5,
        "crs": 28992, "licence": "CC0", "test_point": (121400, 487400),
        "kwargs": dict(url="https://service.pdok.nl/rws/ahn/wcs/v1_0", coverage="dsm_05m",
                       axis_labels=("x", "y"), attribution="AHN (Rijkswaterstaat), CC0"),
    },
    "be-flanders": {
        "bbox": (2.5, 50.68, 5.95, 51.51),
        "kind": "wcs", "region": "Belgium (Flanders)", "product": "DHMV II DSM (2013-2015)",
        "resolution": 1.0, "crs": 31370, "licence": "Modellicentie Gratis Hergebruik",
        "test_point": (104515, 194015),
        "kwargs": dict(url="https://geo.api.vlaanderen.be/dhmv/wcs", coverage="DHMVII_DSM_1m",
                       axis_labels=("x", "y"), attribution="Digitaal Vlaanderen, DHMV II"),
    },
    "uk-england": {
        "bbox": (-6.5, 49.85, 1.8, 55.85),
        "kind": "wcs", "region": "United Kingdom (England)",
        "product": "Environment Agency LiDAR Composite DSM, first return", "resolution": 1.0,
        "crs": 27700, "licence": "Open Government Licence v3", "test_point": (530015, 180015),
        "kwargs": dict(url="https://environment.data.gov.uk/spatialdata/"
                           "lidar-composite-digital-surface-model-first-return-dsm-1m/wcs",
                       coverage="df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m",
                       axis_labels=("E", "N"), attribution="© Environment Agency, OGL v3"),
    },
    "de-nw": {
        "bbox": (5.85, 50.3, 9.5, 52.55),
        "kind": "wcs", "region": "Germany (North Rhine-Westphalia)", "product": "DOM1",
        "resolution": 1.0, "crs": 25832, "licence": "dl-de/zero-2-0", "test_point": (344415, 5677415),
        "kwargs": dict(url="https://www.wcs.nrw.de/geobasis/wcs_nw_dom", coverage="nw_dom",
                       axis_labels=("x", "y"), attribution="Geobasis NRW, dl-de/zero-2-0"),
    },
    "de-mv": {
        "bbox": (10.55, 53.1, 14.45, 54.7),
        "kind": "wcs", "region": "Germany (Mecklenburg-Vorpommern)", "product": "DOM1",
        "resolution": 1.0, "crs": 25833, "licence": "CC BY 4.0", "test_point": (312015, 6000015),
        "kwargs": dict(url="https://www.geodaten-mv.de/dienste/dom_wcs", coverage="mv_dom1",
                       axis_labels=("x", "y"), attribution="© GeoBasis-DE/M-V, CC BY 4.0"),
    },
    "pl": {
        "bbox": (14.1, 49.0, 24.2, 54.9),
        "kind": "wcs", "region": "Poland", "product": "NMPT (GUGiK), slow server",
        "resolution": 0.5, "crs": 2180, "licence": "free reuse (GUGiK)", "test_point": (637015, 486015),
        "kwargs": dict(url="https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModel",
                       coverage="DSM_PL-EVRF2007-NH", version="1.0.0", timeout=90,
                       attribution="GUGiK, geoportal.gov.pl"),
    },
    "no": {
        "bbox": (4.5, 57.9, 31.2, 71.2),
        "kind": "wcs", "region": "Norway", "product": "Nasjonal høydemodell DOM", "resolution": 1.0,
        "crs": 25833, "licence": "CC BY 4.0", "test_point": (262015, 6649015),
        "kwargs": dict(url="https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25833",
                       coverage="nhm_dom_topo_25833", version="1.0.0", format="GeoTIFF",
                       attribution="© Kartverket, CC BY 4.0"),
    },
    "ch-swisssurface3d": {
        "bbox": (5.95, 45.8, 10.5, 47.85),
        "kind": "stac", "region": "Switzerland", "product": "swisstopo swissSURFACE3D Raster",
        "resolution": 0.5, "crs": 2056, "licence": "swisstopo OGD (free, attribution)",
        "test_point": (2683515, 1247515),
        "kwargs": dict(stac_url="https://data.geo.admin.ch/api/stac/v0.9",
                       collection="ch.swisstopo.swisssurface3d-raster",
                       asset_filter=lambda k, a: k.endswith(".tif") and "_0.5_" in k,
                       attribution="© swisstopo"),
    },
}


def service(name, **overrides):
    """Builds the service `name` of the catalogue; `overrides` replace its settings."""
    e = SERVICES[name.lower()]
    if e["kind"] == "class":
        return getattr(dsm, e["cls"])(**overrides)
    kw = dict(e["kwargs"], crs=e["crs"], source=name.lower())
    if e["kind"] == "wcs":
        kw["resolution"] = e["resolution"]
        return dsm.WCSService(**dict(kw, **overrides))
    return dsm.STACCogService(**dict(kw, **overrides))


def services_at(lon, lat):
    """
    Names of the services whose approximate extent contains (lon, lat), the
    most specific first (smallest extent; catalogue order on ties, so that
    "fr-lidar" comes before "fr-correl"). The extents are rough: a service may
    still answer "no_coverage", in which case the next one can be tried.
    """
    hits = [(i, k, e["bbox"]) for i, (k, e) in enumerate(SERVICES.items())
            if e["bbox"][0] <= lon <= e["bbox"][2] and e["bbox"][1] <= lat <= e["bbox"][3]]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return [k for i, k, b in sorted(hits, key=lambda h: (area(h[2]), h[0]))]


def available_services():
    """Table of the services of the catalogue (name, region, product, resolution, CRS, licence)."""
    import pandas as pd
    rows = [{"name": k, "region": e["region"], "product": e["product"], "resolution_m": e["resolution"],
             "crs": "EPSG:{}".format(e["crs"]), "licence": e["licence"]} for k, e in SERVICES.items()]
    return pd.DataFrame(rows)
