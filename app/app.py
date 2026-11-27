"""
pypvroof demo: tilt, azimuth and installed capacity of rooftop PV systems,
measured on open surface models (DSM) of several European countries.

    pip install -e ".[viz]" gradio
    python app/app.py

Two ways in: draw a polygon on the map (computed on the fly), or upload /
paste polygons (GeoJSON, GeoPackage, GeoParquet).
"""

import io
import json
import os
import tempfile
import urllib.request

import geopandas as gpd
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from shapely.ops import unary_union

from pypvroof import characterize_roofs
from pypvroof.catalog import SERVICES as CATALOG, services_at
from pypvroof.dsm import get_service
from pypvroof.geometry import reproject, to_geodataframe
from pypvroof.viz import ELEVATION_RAMP, fetch_ortho

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples", "test_polygons.geojson")
REPO = "https://github.com/gabrielkasmi/pypvroof"
MAX_POLYGONS = int(os.environ.get("PYPVROOF_MAX_POLYGONS", 200))  # be gentle with the services
KWP_PER_M2 = 1 / 6.5  # installed capacity coefficient (pypvroof default)
PART_COLORS = ["#f59f00", "#1c7ed6", "#2f9e44", "#ae3ec9", "#e8590c", "#0c8599"]

AUTO = "Auto (from the location)"
SERVICES = {AUTO: None}
SERVICES.update({"{} — {}".format(e["region"], e["product"]): k for k, e in CATALOG.items()})
SHOWN = {"parent_id": "polygon", "part": "plane", "service": "surface model", "status": "status",
         "tilt": "tilt (°)", "azimuth": "azimuth (°)", "surface_proj": "projected area (m²)",
         "surface": "panel surface (m²)", "installed_capacity": "capacity (kWp)"}
STATUS_TEXT = {
    "ok": "Roof plane measured.",
    "flat": "Flat roof: the azimuth is undefined.",
    "multimodal": "Several roof planes under the polygon: tick “split” to measure each one.",
    "no_coverage": "No surface model covers this location (see the covered areas below).",
    "too_few_points": "Polygon too small for the resolution of the surface model.",
    "fit_failed": "No plane could be fitted (vegetation, superstructures...).",
    "service_error": "The surface model service did not answer, try again later.",
}


# -- computation --------------------------------------------------------------

def characterize_auto(gdf, split):
    """
    Each polygon with the most specific service covering its location; when
    a service answers no_coverage, the next candidate is tried.
    """
    pts = gdf.to_crs(4326).geometry.representative_point()
    cands = {pid: services_at(p.x, p.y) for pid, p in zip(gdf["id"], pts)}
    results = []
    while cands:
        groups = {}
        for pid, c in cands.items():
            groups.setdefault(c[0] if c else None, []).append(pid)
        retry = {}
        for svc, pids in groups.items():
            sub = gdf[gdf["id"].isin(pids)]
            if svc is None:  # no service covers this location
                results.append(gpd.GeoDataFrame(
                    {"parent_id": sub["id"].values, "part": 0, "status": "no_coverage", "service": ""},
                    geometry=sub.geometry.values, crs=sub.crs))
                continue
            out = characterize_roofs(sub, svc, split=split, workers=8, id_column="id").assign(service=svc)
            again = {pid for pid in out.loc[out.status == "no_coverage", "parent_id"] if len(cands[pid]) > 1}
            results.append(out[~out.parent_id.isin(again)])
            retry.update({pid: cands[pid][1:] for pid in again})
        cands = retry
    return pd.concat(results, ignore_index=True)


def compute(gdf, service_label, split, pv=True):
    """`pv`: the polygons are PV arrays (installed capacity); otherwise roof pans."""
    if len(gdf) > MAX_POLYGONS:
        raise gr.Error(f"{len(gdf)} polygons: the demo is limited to {MAX_POLYGONS}.")
    if "id" not in gdf.columns:
        gdf["id"] = [str(i + 1) for i in range(len(gdf))]
    gdf["id"] = gdf["id"].astype(str)
    spec = SERVICES[service_label]
    if spec is None:
        out = characterize_auto(gdf, split)
    else:
        out = characterize_roofs(gdf, spec, split=split, workers=8, id_column="id").assign(service=spec)
    out["parent_id"] = out["parent_id"].astype(str)
    if pv:
        out["installed_capacity"] = out.get("surface", np.nan) * KWP_PER_M2
    out = out.drop(columns=["modes"], errors="ignore")

    names = dict(SHOWN, surface="panel surface (m²)" if pv else "surface (m²)")
    table = pd.DataFrame(out[[c for c in SHOWN if c in out.columns]]).round(1).rename(columns=names)
    table["plane"] = table["plane"] + 1
    path = os.path.join(tempfile.mkdtemp(), "pypvroof_results.geojson")
    out.to_file(path, driver="GeoJSON")
    return out, table, path


# -- figure -------------------------------------------------------------------

def aerial(bounds, crs, resolution=0.15):
    """
    Aerial image on `bounds` (in `crs`): IGN BD ORTHO in France, Esri World
    Imagery elsewhere (fetched in Web Mercator and warped to `crs`).
    """
    from pyproj import CRS
    crs = CRS.from_user_input(crs).to_epsg()
    if crs == 2154:
        return fetch_ortho(bounds, resolution=resolution), "IGN BD ORTHO"
    from PIL import Image
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject as warp, transform_bounds

    minx, miny, maxx, maxy = bounds
    w, h = int((maxx - minx) / resolution), int((maxy - miny) / resolution)
    # mosaic of the Esri tiles (zoom 19, ~0.2 m) covering the extent, in Web Mercator
    z, R = 19, 20037508.342789244
    mb = transform_bounds(f"EPSG:{crs}", "EPSG:3857", *bounds, densify_pts=5)
    size = 2 * R / 2 ** z
    c0, c1 = int((mb[0] + R) // size), int((mb[2] + R) // size)
    r0, r1 = int((R - mb[3]) // size), int((R - mb[1]) // size)
    src = np.zeros(((r1 - r0 + 1) * 256, (c1 - c0 + 1) * 256, 3), dtype=np.uint8)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            url = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
                   f"{z}/{r}/{c}")
            req = urllib.request.Request(url, headers={"User-Agent": "pypvroof"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                tile = np.asarray(Image.open(io.BytesIO(resp.read())).convert("RGB"))
            src[(r - r0) * 256:(r - r0 + 1) * 256, (c - c0) * 256:(c - c0 + 1) * 256] = tile
    src_bounds = (c0 * size - R, R - (r1 + 1) * size, (c1 + 1) * size - R, R - r0 * size)
    dst = np.zeros((h, w, 3), dtype=np.uint8)
    for b in range(3):
        warp(src[:, :, b], dst[:, :, b], src_transform=from_bounds(*src_bounds, src.shape[1], src.shape[0]),
             src_crs="EPSG:3857", dst_transform=from_bounds(*bounds, w, h), dst_crs=f"EPSG:{crs}",
             resampling=Resampling.bilinear)
    return dst, "Esri World Imagery"


def part_label(row):
    if row["status"] in ("ok", "flat") and np.isfinite(row.get("tilt", np.nan)):
        az = row.get("azimuth", np.nan)
        return "{:.0f}°".format(row["tilt"]) + ("" if not np.isfinite(az) else " · {:.0f}°".format(az))
    return row["status"]


def figure(parts, service, margin=6.0):
    """Aerial image and surface model around the polygon, with every part outlined."""
    if not service or parts.empty:
        return None
    plt.close("all")
    svc = get_service(service)
    geoms = [reproject(g, parts.crs, svc.crs) for g in parts.geometry]
    fr = svc.fetch(unary_union(geoms).buffer(margin))
    if fr.patch is None:
        return None
    t, (h, w) = fr.patch.transform, fr.patch.z.shape
    ext = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), dpi=110)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)
    try:
        img, credit = aerial((ext[0], ext[2], ext[1], ext[3]), svc.crs)
        axes[0].imshow(img, extent=ext)
    except Exception:  # imagery is a nice-to-have
        credit = "unavailable"
        axes[0].set_facecolor("#eceff3")
    z = fr.patch.z
    vmin, vmax = np.nanpercentile(z, [2, 98])
    axes[1].imshow(z, extent=ext, cmap=LinearSegmentedColormap.from_list("e", ELEVATION_RAMP),
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    entry = CATALOG.get(service, {})
    titles = ["Aerial imagery\n" + credit,
              "Surface model\n{} ({} m)".format(entry.get("product", service).split(",")[0],
                                                entry.get("resolution", "?"))]
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    total = sum(g.area for g in geoms)
    for ax, title in zip(axes, titles):
        for k, (g, (_, row)) in enumerate(zip(geoms, parts.iterrows())):
            color = PART_COLORS[k % len(PART_COLORS)]
            for poly in getattr(g, "geoms", [g]):
                x, y = poly.exterior.xy
                ax.plot(x, y, color="white", lw=3.2)
                ax.plot(x, y, color=color, lw=1.8)
            if len(geoms) > 6 and g.area < 0.08 * total:
                continue  # many parts: label the large ones only
            p = g.representative_point()
            ax.text(p.x, p.y, part_label(row), ha="center", va="center", fontsize=9, weight="bold",
                    color="#1a1a1a", path_effects=halo)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=9.5, color="#3d4451", linespacing=1.4)
    x0, y0 = ext[0] + 1, ext[2] + 1
    axes[0].plot([x0, x0 + 5], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    axes[0].text(x0 + 2.5, y0 + 0.5, "5 m", color="white", ha="center", fontsize=8, weight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a1a")])
    return fig


# -- result card --------------------------------------------------------------

def _compass(az):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((az + 22.5) // 45) % 8]


def card(parts):
    """HTML summary of the parts of one polygon."""
    rows = []
    for k, (_, r) in enumerate(parts.iterrows()):
        color = PART_COLORS[k % len(PART_COLORS)]
        st = r["status"]
        title = "Roof plane {}".format(k + 1) if len(parts) > 1 else "Result"
        metrics = ""
        if st in ("ok", "flat"):
            az = r.get("azimuth", np.nan)
            vals = [("Tilt", "{:.0f}°".format(r["tilt"])),
                    ("Azimuth", "—" if not np.isfinite(az) else "{:.0f}° <small>{}</small>".format(az, _compass(az))),
                    ("Surface", "{:.0f} m²".format(r["surface"]))]
            if "installed_capacity" in parts.columns:
                vals.append(("Capacity", "{:.1f} kWp".format(r["installed_capacity"])))
            metrics = "".join('<div class="pv-metric"><span class="pv-v">{}</span><span class="pv-k">{}</span></div>'
                              .format(v, k_) for k_, v in vals)
        rows.append('<div class="pv-card" style="border-left-color:{}"><div class="pv-head">{}'
                    '<span class="pv-badge pv-{}">{}</span></div>{}<p class="pv-note">{}</p></div>'
                    .format(color, title, "good" if st in ("ok", "flat") else "warn", st,
                            '<div class="pv-metrics">{}</div>'.format(metrics) if metrics else "",
                            STATUS_TEXT.get(st, "")))
    svc = parts["service"].iloc[0] if "service" in parts else ""
    foot = ""
    if svc:
        foot = "Surface model: {}".format(CATALOG.get(svc, {}).get("product", svc))
        acq = parts.get("meta_date_fin_acquisition")
        if acq is not None and pd.notna(acq.iloc[0]):
            foot += ", acquired {}".format(str(acq.iloc[0])[:10])
        foot += " · azimuth 0 = north, clockwise"
        if "installed_capacity" in parts.columns:
            foot += " · capacity = surface × {:.3f} kWp/m²".format(KWP_PER_M2)
        foot = '<p class="pv-foot">{}</p>'.format(foot)
    return "".join(rows) + foot


EMPTY_CARD = ('<div class="pv-empty">Find a roof, then draw a polygon over the panels (or the whole roof) '
              'with the tools on the left of the map.</div>')


# -- map tab ------------------------------------------------------------------

def run_drawn(geojson, split, pv):
    if not geojson or not geojson.strip():
        return EMPTY_CARD, None, None, None, ""
    gdf = to_geodataframe(json.loads(geojson))
    gdf["id"] = ["drawn"]
    out, table, path = compute(gdf, AUTO, split, pv)
    parts = out.to_crs(4326)
    shapes = parts[["part", "status", "tilt", "azimuth"]].copy()
    shapes["label"] = [part_label(r) for _, r in parts.iterrows()]
    shapes = gpd.GeoDataFrame(shapes[["part", "label"]], geometry=parts.geometry.values, crs=4326)
    return card(out), figure(out, out["service"].iloc[0]), table, path, shapes.to_json()


# -- file tab -----------------------------------------------------------------

def load(file, text):
    if file is not None:
        return to_geodataframe(file if isinstance(file, str) else file.name)
    if text and text.strip():
        return to_geodataframe(json.loads(text))
    return to_geodataframe(EXAMPLES)


def run_file(file, text, service_label, split, pv):
    gdf = load(file, text)
    out, table, path = compute(gdf, service_label, split, pv)
    ids = list(dict.fromkeys(out["parent_id"]))
    state = out.to_json()
    first = ids[0] if ids else None
    return (table, path, gr.update(choices=ids, value=first), state) + show(first, state)


def show(polygon_id, state):
    if not polygon_id or not state:
        return "", None
    out = gpd.read_file(io.StringIO(state))
    parts = out[out["parent_id"].astype(str) == str(polygon_id)]
    if parts.empty:
        return "", None
    return card(parts), figure(parts, parts["service"].iloc[0])


# -- page ---------------------------------------------------------------------

def covered_areas():
    by_region = {}
    for k, e in CATALOG.items():
        by_region.setdefault(e["region"], []).append("{} ({} m, `{}`)".format(e["product"], e["resolution"], k))
    items = "\n".join("- **{}**: {}".format(r, "; ".join(p)) for r, p in by_region.items())
    return ("### Covered areas\n" + items + "\n\nYour country is missing? "
            f"[Request it]({REPO}/issues/new) or [add it]({REPO}/blob/master/CONTRIBUTING.md): "
            "most national surface models are a single entry in the catalogue.")


HOW = f"""
### How it works
1. The **surface model** (DSM, elevations of roofs) under the polygon is fetched on the fly
   from the open service covering the location.
2. A robust **Theil-Sen** fit gives the plane of the roof: its **tilt** and **azimuth**
   (0 = north, clockwise). Polygons covering several roof planes can be **split** into one part per plane.
3. The **installed capacity** is the surface of the panels (projected area / cos(tilt))
   times a coefficient (1/6.5 kWp/m²).

Python package, documentation and citation: [{REPO.split('//')[1]}]({REPO}).
"""

HEAD = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
"""
DRAW_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"

MAP_HTML = ('<div class="pv-searchbar"><input id="pvsearch" type="search" autocomplete="off" '
            'placeholder="Search an address or a place, or type “lat, lon”, then press Enter"></div>'
            '<div id="pvmap"></div>')

MAP_JS = """
() => {
  const init = () => {
    const el = document.getElementById('pvmap');
    if (!window.L || !el) { setTimeout(init, 200); return; }
    // Leaflet.draw needs Leaflet: load it only now (head scripts run in any order)
    if (!L.Control.Draw) {
      if (!window._pvdraw) {
        window._pvdraw = document.createElement('script');
        window._pvdraw.src = 'DRAW_JS_URL';
        document.head.appendChild(window._pvdraw);
      }
      setTimeout(init, 200); return;
    }
    if (el._map) return;
    const o = {maxZoom: 21, maxNativeZoom: 19};
    const wms = (url, layers, attribution) =>
      L.tileLayer.wms(url, {layers, format: 'image/jpeg', maxZoom: 21, attribution});
    const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      {...o, attribution: '© OpenStreetMap contributors'});
    const layers = {
      'OpenStreetMap': osm,
      'Aerial · Esri World Imagery (everywhere)': L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        {...o, attribution: '© Esri, Maxar, Earthstar Geographics'}),
      'Aerial · France (IGN)': L.tileLayer(
        'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg',
        {...o, attribution: '© IGN'}),
      'Aerial · Netherlands (PDOK)': L.tileLayer(
        'https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg',
        {...o, attribution: '© PDOK, Beeldmateriaal Nederland'}),
      'Aerial · Flanders (Digitaal Vlaanderen)': wms('https://geo.api.vlaanderen.be/OMWRGBMRVL/wms', 'Ortho', '© Digitaal Vlaanderen'),
      'Aerial · North Rhine-Westphalia (Geobasis NRW)': wms('https://www.wms.nrw.de/geobasis/wms_nw_dop', 'nw_dop_rgb', '© Geobasis NRW'),
      'Aerial · Mecklenburg-Vorpommern (GeoBasis-DE/M-V)': wms('https://www.geodaten-mv.de/dienste/adv_dop', 'mv_dop', '© GeoBasis-DE/M-V'),
      'Aerial · Poland (GUGiK)': wms('https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution', 'Raster', '© GUGiK'),
      'Aerial · Switzerland (swisstopo)': L.tileLayer(
        'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg',
        {...o, attribution: '© swisstopo'}),
    };
    const map = L.map(el, {layers: [osm]}).setView([48.5, 6], 5);
    el._map = map;
    L.control.layers(layers, null, {position: 'topright', collapsed: true}).addTo(map);
    L.control.scale({imperial: false}).addTo(map);
    const drawn = new L.FeatureGroup().addTo(map);
    const style = {color: '#f59f00', weight: 2, fillOpacity: 0.15};
    map.addControl(new L.Control.Draw({
      position: 'topleft',
      draw: {polygon: {showArea: true, metric: true, shapeOptions: style}, rectangle: {shapeOptions: style},
             polyline: false, circle: false, marker: false, circlemarker: false},
      edit: {featureGroup: drawn}
    }));
    const send = (layer) => {
      const box = document.querySelector('#drawn_geojson textarea');
      box.value = JSON.stringify(layer.toGeoJSON().geometry);
      box.dispatchEvent(new Event('input', {bubbles: true}));
    };
    let result = null;
    const clearResult = () => { if (result) { map.removeLayer(result); result = null; } };
    map.on(L.Draw.Event.CREATED, (e) => { clearResult(); drawn.clearLayers(); drawn.addLayer(e.layer); send(e.layer); });
    map.on(L.Draw.Event.EDITED, (e) => { clearResult(); e.layers.eachLayer(send); });
    map.on(L.Draw.Event.DELETED, () => clearResult());
    // the measured parts, drawn back on the map with their tilt and azimuth
    const colors = PART_COLORS;
    window.pvShowResult = (g) => {
      clearResult();
      if (!g) return;
      drawn.eachLayer(l => l.setStyle({color: '#495057', weight: 1.5, dashArray: '4 4', fillOpacity: 0}));
      result = L.geoJSON(JSON.parse(g), {
        style: f => ({color: colors[f.properties.part % colors.length], weight: 2.5, fillOpacity: 0.3}),
        onEachFeature: (f, l) => l.bindTooltip(f.properties.label,
          {permanent: true, direction: 'center', className: 'pv-tip'})
      }).addTo(map);
    };
    // search bar above the map: "lat, lon" or an address (Nominatim)
    const input = document.getElementById('pvsearch');
    input.addEventListener('keydown', async (ev) => {
      if (ev.key !== 'Enter') return;
      const m = input.value.match(/^\\s*(-?\\d+(?:\\.\\d+)?)\\s*[,; ]\\s*(-?\\d+(?:\\.\\d+)?)\\s*$/);
      if (m) { map.setView([+m[1], +m[2]], 19); return; }
      const r = await fetch('https://nominatim.openstreetmap.org/search?format=json&limit=1&q='
                            + encodeURIComponent(input.value));
      const j = await r.json();
      if (j.length) map.setView([+j[0].lat, +j[0].lon], 19);
      else input.setCustomValidity('Not found'), input.reportValidity(), input.setCustomValidity('');
    });
  };
  init();
}
""".replace("DRAW_JS_URL", DRAW_JS).replace("PART_COLORS", json.dumps(PART_COLORS))

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto}
#drawn_geojson, #result_geojson {display: none}
#pvmap {height: 560px; border-radius: 12px; border: 1px solid var(--border-color-primary)}
.pv-hero {padding: 28px 4px 8px}
.pv-hero h1 {font-size: 2.1rem; font-weight: 700; margin: 0; letter-spacing: -0.02em}
.pv-hero h1 span {color: var(--color-accent)}
.pv-hero p {font-size: 1.05rem; color: var(--body-text-color-subdued); margin: 6px 0 12px; max-width: 760px}
.pv-links a {display: inline-block; margin-right: 8px; padding: 3px 10px; border-radius: 999px;
  border: 1px solid var(--border-color-primary); font-size: 0.85rem; text-decoration: none;
  color: var(--body-text-color)}
.pv-links a:hover {border-color: var(--color-accent)}
.pv-empty {padding: 28px 18px; border: 1px dashed var(--border-color-primary); border-radius: 12px;
  color: var(--body-text-color-subdued); text-align: center}
.pv-card {border: 1px solid var(--border-color-primary); border-left: 5px solid; border-radius: 10px;
  padding: 12px 16px; margin-bottom: 10px; background: var(--block-background-fill)}
.pv-head {font-weight: 600; display: flex; justify-content: space-between; align-items: center}
.pv-badge {font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 999px}
.pv-good {background: #d3f9d8; color: #2b8a3e}
.pv-warn {background: #fff3bf; color: #a35d00}
.pv-metrics {display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 8px; margin-top: 10px}
.pv-metric {display: flex; flex-direction: column}
.pv-v {font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; white-space: nowrap}
.pv-v small {font-size: 0.8rem; font-weight: 600; color: var(--body-text-color-subdued)}
.pv-k {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--body-text-color-subdued)}
.pv-note {margin: 8px 0 0; font-size: 0.85rem; color: var(--body-text-color-subdued)}
.pv-foot {font-size: 0.78rem; color: var(--body-text-color-subdued); margin-top: 4px}
.pv-searchbar {margin-bottom: 10px}
#pvsearch {width: 100%; box-sizing: border-box; padding: 11px 14px 11px 38px; border-radius: 10px;
  border: 1px solid var(--border-color-primary); background: var(--input-background-fill)
  url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='none' stroke='%23868e96' stroke-width='2' viewBox='0 0 24 24'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-4-4'/%3E%3C/svg%3E")
  no-repeat 13px center; color: var(--body-text-color); font-size: 0.95rem}
#pvsearch:focus {outline: none; border-color: var(--color-accent); box-shadow: 0 0 0 3px rgba(245,159,0,.2)}
.leaflet-tooltip.pv-tip {background: transparent; border: none; box-shadow: none; padding: 0;
  font-weight: 700; font-size: 12px; color: #1a1a1a;
  text-shadow: 0 0 3px #fff, 0 0 3px #fff, 0 0 2px #fff}
.leaflet-tooltip.pv-tip::before {display: none}
"""

THEME = gr.themes.Soft(primary_hue="amber", secondary_hue="blue", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    button_primary_text_color="*neutral_900", button_primary_text_color_hover="*neutral_950")

HERO = f"""
<div class="pv-hero">
  <h1>PyPV<span>Roof</span></h1>
  <p>Tilt, azimuth and installed capacity of rooftop PV systems, measured on the open
  surface models of European mapping agencies.</p>
  <div class="pv-links">
    <a href="{REPO}" target="_blank">GitHub</a>
    <a href="https://arxiv.org/abs/2309.07143" target="_blank">Paper</a>
    <a href="{REPO}/blob/master/docs/api.md" target="_blank">Documentation</a>
  </div>
</div>
"""

with gr.Blocks(title="PyPVRoof", theme=THEME, head=HEAD, css=CSS) as demo:
    gr.HTML(HERO)
    with gr.Tab("Draw on the map"):
        gr.Markdown("Search a place, zoom on a roof and **draw a polygon** (or a rectangle) with the tools on the "
                    "left of the map: the characteristics are computed right away. The map shows OpenStreetMap; "
                    "switch to **aerial imagery** with the layer button (top right): national orthophotos "
                    "where available, Esri World Imagery elsewhere.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.HTML(MAP_HTML)
                with gr.Row():
                    pv_map = gr.Checkbox(value=True, label="Extract PV characteristics",
                                         info="The polygon is a PV array: installed capacity. "
                                              "Untick for a roof pan.")
                    split_map = gr.Checkbox(value=True, label="Split into roof planes",
                                            info="When the polygon covers several roof planes.")
            with gr.Column(scale=2):
                card_map = gr.HTML(EMPTY_CARD)
                plot_map = gr.Plot(show_label=False)
                with gr.Accordion("Table and download", open=False):
                    table_map = gr.Dataframe(show_label=False)
                    download_map = gr.File(label="Result (GeoJSON)")
        drawn = gr.Textbox(elem_id="drawn_geojson", show_label=False)
        result = gr.Textbox(elem_id="result_geojson", show_label=False)
        outs = [card_map, plot_map, table_map, download_map, result]
        for trigger in (drawn, split_map, pv_map):
            trigger.change(run_drawn, [drawn, split_map, pv_map], outs)
        result.change(None, [result], None, js="(g) => { window.pvShowResult && window.pvShowResult(g); }")
    with gr.Tab("Upload polygons"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                file = gr.File(label="Polygons (GeoJSON, GeoPackage, GeoParquet)",
                               file_types=[".geojson", ".json", ".gpkg", ".parquet"])
                text = gr.Textbox(label="… or paste a GeoJSON", lines=3)
                service = gr.Dropdown(list(SERVICES), value=AUTO, label="Surface model")
                pv = gr.Checkbox(value=True, label="Extract PV characteristics",
                                 info="The polygons are PV arrays: installed capacity. Untick for roof pans.")
                split = gr.Checkbox(value=True, label="Split polygons covering several roof planes")
                go = gr.Button("Characterize", variant="primary")
                gr.Markdown(f"Without a file, the built-in examples are used (France, Switzerland, Netherlands, "
                            f"England). Up to {MAX_POLYGONS} polygons.")
            with gr.Column(scale=2):
                table = gr.Dataframe(label="Results")
                download = gr.File(label="Results (GeoJSON)")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                pick = gr.Dropdown(label="Polygon", choices=[])
                card_file = gr.HTML()
            with gr.Column(scale=2):
                plot = gr.Plot(show_label=False)
        state = gr.State()
        go.click(run_file, [file, text, service, split, pv], [table, download, pick, state, card_file, plot])
        pick.change(show, [pick, state], [card_file, plot])
    with gr.Row():
        gr.Markdown(HOW)
        gr.Markdown(covered_areas())
    demo.load(None, None, None, js=MAP_JS)

# Hugging Face ZeroGPU hardware (the only free Space hardware) refuses to start
# an app without a @spaces.GPU function. The demo needs no GPU: this function
# is never called, it only satisfies the startup check.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        pass
except ImportError:  # not on a Space
    pass

if __name__ == "__main__":
    demo.launch(ssr_mode=False)  # the map is injected by JavaScript on page load
