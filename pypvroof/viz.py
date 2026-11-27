# -*- coding: utf-8 -*-
"""
Visual checks: the orthophoto under a polygon and the elevation models used to
estimate its tilt and azimuth, one under the other.

Needs matplotlib and Pillow (`pip install pypvroof[viz]`). The orthophoto is
the IGN BD ORTHO (France), fetched from the Géoplateforme WMS-R.

    from pypvroof.viz import plot_polygon
    fig = plot_polygon(polygon, services=("ign", "ign-correl"))
"""

import io
import urllib.parse
import urllib.request

import numpy as np

from . import dsm, theilsen
from .bulk import characterize_polygon, _polygonal
from .geometry import reproject, to_geodataframe

TEXT, TEXT2 = "#191919", "#666666"
# sequential single-hue ramp (blue, light -> dark) for elevations
ELEVATION_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

ORTHO_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"


def fetch_ortho(bounds, resolution=0.2, layer=ORTHO_LAYER, timeout=60):
    """
    IGN orthophoto (RGB array) on a Lambert-93 extent (minx, miny, maxx, maxy).
    """
    from PIL import Image
    minx, miny, maxx, maxy = bounds
    w = max(1, int(round((maxx - minx) / resolution)))
    h = max(1, int(round((maxy - miny) / resolution)))
    params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer,
              "FORMAT": "image/jpeg", "STYLES": "", "CRS": "EPSG:2154",
              "BBOX": "{},{},{},{}".format(minx, miny, maxx, maxy), "WIDTH": w, "HEIGHT": h}
    url = dsm.IGNLidarService.WMS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "pypvroof"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))


def _label(service):
    return {"ign-lidar-hd": "MNS LiDAR HD", "ign-mns-correl": "MNS Correl",
            "local": "MNS local"}.get(getattr(service, "source", "local"), type(service).__name__)


def _fmt(r):
    if r["status"] in (theilsen.OK, theilsen.FLAT):
        az = "—" if not np.isfinite(r["azimuth"]) else "{:.0f}°".format(r["azimuth"])
        return "{}: tilt {:.1f}°, azimut {}".format(r["status"], r["tilt"], az)
    return r["status"]


def plot_polygon(polygon, services=("ign", "ign-correl"), crs=None, margin=6.0, params=None,
                 ortho=True, figsize_per_panel=(5.0, 4.2)):
    """
    Figure with the orthophoto (if `ortho`) and, below, the elevation grid of
    each service around the polygon, with the estimated tilt / azimuth.

    polygon  : a GeoJSON feature / geometry, a shapely polygon, or a one-row
               GeoDataFrame (CRS from `crs`, default EPSG:4326)
    services : services or specifications ("ign", "ign-correl", a folder...)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    gdf = to_geodataframe(polygon if not hasattr(polygon, "iloc") else polygon.iloc[:1], crs=crs)
    services = [dsm.get_service(s) for s in services]
    # the panels share the extent of the first service's grid (IGN: Lambert-93)
    target = services[0].crs
    geom = _polygonal(reproject(gdf.geometry.iloc[0], gdf.crs, target))

    panels = []
    for svc in services:
        g = reproject(geom, target, svc.crs)
        fr = svc.fetch(g.buffer(margin))  # bigger window for context
        [(_, r, _)] = characterize_polygon(g, svc, params)
        panels.append((svc, fr, r))

    ref = next((fr.patch for _, fr, _ in panels if fr.patch is not None), None)
    if ref is None:
        minx, miny, maxx, maxy = geom.buffer(margin).bounds
    else:
        t, (h, w) = ref.transform, ref.z.shape
        minx, maxx, miny, maxy = t.c, t.c + w * t.a, t.f + h * t.e, t.f
    extent = (minx, maxx, miny, maxy)

    zs = [fr.patch.z for _, fr, _ in panels if fr.patch is not None]
    vals = np.concatenate([z[np.isfinite(z)] for z in zs]) if zs else np.array([0.0, 1.0])
    vmin, vmax = np.percentile(vals, [2, 98]) if vals.size else (0, 1)

    n = len(panels) + (1 if ortho else 0)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n),
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    cmap = LinearSegmentedColormap.from_list("elevation", ELEVATION_RAMP)
    xs, ys = geom.exterior.xy if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda p: p.area).exterior.xy

    def outline(ax):
        ax.plot(xs, ys, color="#ffffff", lw=2.2)
        ax.plot(xs, ys, color=TEXT, lw=1.0)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    i = 0
    if ortho:
        ax = axes[0]
        try:
            ax.imshow(fetch_ortho((minx, miny, maxx, maxy)), extent=extent)
        except Exception as e:  # orthophoto is a nice-to-have
            ax.text(0.5, 0.5, "orthophoto indisponible\n{}".format(type(e).__name__),
                    ha="center", va="center", transform=ax.transAxes, color=TEXT2)
        ax.set_title("Orthophoto (BD ORTHO) — surface projetée {:.0f} m²".format(geom.area),
                     loc="left", color=TEXT, fontsize=10)
        x0, y0 = minx + 1, miny + 1
        ax.plot([x0, x0 + 5], [y0, y0], color="#ffffff", lw=4, solid_capstyle="butt")
        ax.text(x0 + 2.5, y0 + 0.6, "5 m", color="#ffffff", ha="center", fontsize=9, weight="bold")
        outline(ax)
        i = 1

    for svc, fr, r in panels:
        ax = axes[i]
        i += 1
        title = "{} — {}".format(_label(svc), _fmt(r))
        acq = fr.meta.get("date_fin_acquisition")
        if acq:
            title += "\nacquisition LiDAR {}".format(str(acq)[:10])
        if fr.patch is not None:
            t, (h, w) = fr.patch.transform, fr.patch.z.shape
            im = ax.imshow(fr.patch.z, extent=(t.c, t.c + w * t.a, t.f + h * t.e, t.f), cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("Altitude (m)", color=TEXT2)
            cb.outline.set_visible(False)
        else:
            ax.text(0.5, 0.5, fr.status, ha="center", va="center", transform=ax.transAxes, color=TEXT2)
        ax.set_title(title, loc="left", color=TEXT, fontsize=10)
        outline(ax)
    return fig
