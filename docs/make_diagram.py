"""
Draws docs/assets/pipeline.png: how pypvroof characterizes one polygon.
Needs network (IGN services). Run from the repository root:

    python docs/make_diagram.py
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, FancyArrowPatch
from shapely.geometry import shape

from pypvroof import IGNLidarService, MetadataExtraction
from pypvroof.bulk import _masks, characterize_polygon
from pypvroof.geometry import reproject
from pypvroof.theilsen import TheilSenParams
from pypvroof.viz import fetch_ortho

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/pipeline.png"
INK, MUTED, RULE, ACCENT = "#1a1a1a", "#6b6b6b", "#bdbdbd", "#1f6fd1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})

# a PV array detected by DeepPVMapper (array_id 239353, validated)
POLYGON = {"type": "Polygon", "coordinates": [[
    [0.748136, 47.448596], [0.748126, 47.448586], [0.748105, 47.448579], [0.748105, 47.448571],
    [0.748092, 47.448568], [0.748085, 47.448551], [0.748059, 47.448535], [0.748047, 47.448516],
    [0.748028, 47.448507], [0.748008, 47.448487], [0.748027, 47.448476], [0.74803, 47.448469],
    [0.748041, 47.448469], [0.748071, 47.448497], [0.748082, 47.448499], [0.748089, 47.448512],
    [0.7481, 47.448516], [0.748107, 47.448534], [0.748123, 47.448541], [0.748125, 47.448548],
    [0.748146, 47.448561], [0.748145, 47.448567], [0.748158, 47.448576], [0.748136, 47.448596]]]}


def load_polygon():
    if len(sys.argv) > 2:  # optional: a GeoJSON geometry file
        import json
        return shape(json.load(open(sys.argv[2])))
    return shape(POLYGON)


poly_ll = load_polygon()
svc = IGNLidarService(fetch_metadata=True)
geom = reproject(poly_ll, 4326, svc.crs)
[(_, res, meta)] = characterize_polygon(geom, svc)
ctx = svc.fetch(geom.buffer(5))
patch = ctx.patch
t, (h, w) = patch.transform, patch.z.shape
ext = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)
ortho = fetch_ortho((ext[0], ext[2], ext[1], ext[3]), resolution=0.1)
row = MetadataExtraction(p={"tilt-method": "theil-sen", "azimuth-method": "theil-sen",
                            "dsm-service": svc, "regression-type": "linear"}
                         ).extract_all_characteristics({"type": "Feature", "properties": {},
                                                        "geometry": poly_ll.__geo_interface__})
lon, lat, tilt, azimuth, kwp, surface = row

# pixels used by the fit, projected on the downslope direction
fp = svc.fetch(geom)
full, mask, _ = _masks(geom, fp.patch, TheilSenParams())
rows, cols = np.nonzero(mask & np.isfinite(fp.patch.z))
px, py = fp.patch.pixel_centers(rows, cols)
pz = fp.patch.z[rows, cols]
a = np.radians(azimuth)
d = (px - px.mean()) * np.sin(a) + (py - py.mean()) * np.cos(a)  # metres downslope
slope = np.tan(np.radians(tilt))
z0 = np.median(pz + slope * d)

fig = plt.figure(figsize=(12, 3.8), dpi=160)
W, B, H = 0.19, 0.25, 0.57
xs = [0.02, 0.27, 0.52, 0.77]
axes = [fig.add_axes([x, B, W, H]) for x in xs]
axes[2].set_position([xs[2] + 0.04, B + 0.09, W - 0.04, H - 0.09])  # room for the axis labels
outline = np.array(geom.exterior.coords)


def frame(ax):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(RULE); s.set_linewidth(0.8)


def caption(i, title, text):
    fig.text(xs[i], B + H + 0.07, "{}  {}".format(i + 1, title), fontsize=10.5, weight="bold", va="bottom")
    fig.text(xs[i], B - 0.05, text, fontsize=8.5, color=MUTED, va="top", linespacing=1.4)


# 1. polygon
ax = axes[0]
ax.imshow(ortho, extent=ext)
ax.plot(outline[:, 0], outline[:, 1], color="white", lw=2.2)
ax.plot(outline[:, 0], outline[:, 1], color=INK, lw=0.9)
ax.plot([ext[0] + 1, ext[0] + 6], [ext[2] + 1] * 2, color="white", lw=2.5, solid_capstyle="butt")
ax.text(ext[0] + 3.5, ext[2] + 1.5, "5 m", color="white", ha="center", fontsize=7.5, weight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground=INK)])
frame(ax)
caption(0, "Input polygon", "a detected PV array\nGeoJSON, GeoDataFrame, parquet… any CRS")

# 2. surface model
ax = axes[1]
grey = LinearSegmentedColormap.from_list("grey", ["#3a3a3a", "#f4f4f4"])
ax.imshow(patch.z, extent=ext, cmap=grey, interpolation="nearest",
          vmin=np.nanpercentile(patch.z, 2), vmax=np.nanpercentile(patch.z, 99))
ax.plot(outline[:, 0], outline[:, 1], color=ACCENT, lw=1.3)
frame(ax)
caption(1, "Surface model (DSM)", "elevations under the polygon, fetched on the fly\n"
        "here IGN LiDAR HD, 0.5 m")

# 3. roof plane
ax = axes[2]
resid = pz - z0 + slope * d
inl = np.abs(resid) < TheilSenParams().inlier_tol * 2
ax.scatter(d[inl], (pz - z0)[inl], s=5, color=INK, alpha=0.6, linewidths=0)
ax.scatter(d[~inl], (pz - z0)[~inl], s=5, color=RULE, linewidths=0)
if (~inl).any():
    i = np.argmin((pz - z0)[~inl])
    ax.annotate("ignored by the fit", xy=(d[~inl][i], (pz - z0)[~inl][i]), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", fontsize=7.5, color=MUTED)
dd = np.array([d.min(), d.max()])
ax.plot(dd, -slope * dd, color=ACCENT, lw=1.8)
x0 = dd[0]; y0 = -slope * x0
L = (dd[1] - dd[0]) * 0.45
ax.plot([x0, x0 + L], [y0, y0], color=INK, lw=0.7, ls=(0, (3, 2)))
ax.add_patch(Arc((x0, y0), L * 1.2, L * 1.2, theta1=-tilt, theta2=0, color=INK, lw=0.8))
ax.text(x0 + L * 0.68, y0 - L * 0.05, "{:.0f}°".format(tilt), fontsize=9, va="top")
ax.set_xlabel("distance downslope (m)", fontsize=8, color=MUTED)
ax.set_ylabel("relative elevation (m)", fontsize=8, color=MUTED)
ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
ax.locator_params(nbins=4)
ax.set_aspect("equal", adjustable="datalim")  # true angle
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(RULE)
caption(2, "Tilt and azimuth", "Theil-Sen estimation of the roof plane\non the DSM pixels")

# compass: azimuth of the plane
cx = fig.add_axes([xs[2] + W - 0.045, B + H - 0.13, 0.05, 0.14], polar=True)
cx.set_theta_zero_location("N"); cx.set_theta_direction(-1)
cx.set_rticks([]); cx.set_xticks([]); cx.spines["polar"].set_color(RULE)
cx.annotate("", xy=(a, 1), xytext=(a, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4))
cx.text(0, 1.45, "N", ha="center", va="center", fontsize=7, color=MUTED)
cx.set_ylim(0, 1)

# 4. characteristics
ax = axes[3]
ax.axis("off")
compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((azimuth + 22.5) // 45) % 8]
items = [("tilt", "{:.0f}°".format(tilt)), ("azimuth (0 = north)", "{:.0f}°  ({})".format(azimuth, compass)),
         ("surface", "{:.0f} m²".format(surface)), ("installed capacity", "{:.1f} kWp".format(kwp)),
         ("location", "{:.4f}, {:.4f}".format(lat, lon))]
y = 0.97
for k, v in items:
    ax.text(0.0, y, k, fontsize=8.5, color=MUTED, va="top", transform=ax.transAxes)
    ax.text(1.0, y, v, fontsize=9.5, va="top", ha="right", transform=ax.transAxes,
            weight="bold" if k in ("tilt", "azimuth (0 = north)", "installed capacity") else "normal")
    ax.plot([0, 1], [y - 0.115, y - 0.115], color=RULE, lw=0.5, transform=ax.transAxes)
    y -= 0.145
caption(3, "Characteristics", "surface = projected area / cos(tilt)\n"
        "capacity = surface × coefficient (kWp/m²)")

# arrows between panels
for x in xs[1:]:
    fig.add_artist(FancyArrowPatch((x - 0.052, B + H / 2), (x - 0.012, B + H / 2), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=0.9))
fig.text(0.98, 0.015, "Orthophoto and LiDAR HD: © IGN", fontsize=7, color=MUTED, ha="right")
fig.savefig(OUT, facecolor="white")
print("ok", OUT, round(tilt, 1), round(azimuth), res["status"], round(kwp, 1))
