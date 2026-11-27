# -*- coding: utf-8 -*-
"""
Theil-Sen-type estimation of the tilt and azimuth of a roof plane from an
elevation grid.

The method only sees an elevation grid and a mask (see `dsm.DSMPatch`):

1. Many planes are fitted on random subsets of the pixels under the polygon
   (exact planes through 3 points by default). Subsets drawn on a single roof
   pan give that pan's plane; subsets mixing pans, chimneys, ground or facade
   pixels give scattered planes.
2. The plane estimates are expressed as gradients (dz/dx, dz/dy). The modes of
   their distribution are the candidate roof planes, refined by the median of
   the estimates around each peak (robust to the scattered ones). Working on
   gradients avoids the circularity of the azimuth and is well defined for
   flat roofs.
3. Each pixel is assigned to the closest plane (in elevation); each plane is
   then refitted on its pixels. Planes covering too small a share of the
   polygon are dropped.
4. Decision:
   - one significant plane -> "ok" (or "flat" below `flat_tilt`, azimuth NaN)
   - several planes, but the secondary ones are implausible (too steep), too
     small, or a band along the polygon's edge while the main plane dominates
     -> treated as a misregistration of the polygon (orthophoto vs LiDAR) or
     an obstacle: "ok", with `offset_suspected=True`
   - otherwise -> "multimodal": tilt / azimuth NaN, `inspect=True`. With
     `split=True` (bulk API), the polygon is split into homogeneous parts.

Angles: tilt in degrees from horizontal; azimuth in degrees, 0 = north,
clockwise (90 = east, 180 = south, 270 = west), direction the plane faces.
"""

from dataclasses import dataclass, replace

import numpy as np
import shapely
from scipy import ndimage

OK = "ok"
FLAT = "flat"
MULTIMODAL = "multimodal"
TOO_FEW_POINTS = "too_few_points"
FIT_FAILED = "fit_failed"


@dataclass
class TheilSenParams:
    """Parameters of the estimator. Distances in metres, angles in degrees."""
    n_subsets: int = 2000          # number of random subsets
    subset_size: int = 3           # pixels per subset (3 = exact planes)
    min_subset_spread: float = 0.5  # min area (m2) covered by a subset (degenerate subsets dropped);
                                    # capped at 5% of the area covered by the pixels

    seed: int = 42
    min_pixels: int = 12           # min valid pixels under the (eroded) polygon
    min_area: float = 2.0          # min area (m2) of these pixels, whatever the resolution
    erosion: float = 0.5           # polygon erosion before estimation (edges, gutters)
    bandwidth: float = 0.05        # mode width, in gradient units (tan(tilt))
    peak_min_ratio: float = 0.2    # min peak height relative to the highest peak
    max_modes: int = 4
    search_max_tilt: float = 75.0  # gradients steeper than this are ignored
    inlier_tol: float = 0.15       # max elevation residual (m) to belong to a plane
    min_mode_share: float = 0.10   # min share of the pixels for a plane to count
    max_tilt: float = 60.0         # steeper secondary planes are implausible (facades)
    flat_tilt: float = 5.0         # below: flat roof, azimuth undefined
    min_part_area: float = 3.0     # min area (m2) of a secondary plane / of a split part
    dominant_share: float = 0.75   # share above which the main plane dominates
    edge_band: float = 1.0         # width (m) of the band along the polygon's edge
    edge_frac: float = 0.8         # share of a secondary plane in the edge band to call it an edge effect

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls()
        if isinstance(d, cls):
            return d
        names = cls.__dataclass_fields__.keys()
        unknown = set(d) - set(names)
        if unknown:
            raise ValueError("unknown Theil-Sen parameters: {}".format(sorted(unknown)))
        return cls(**d)


def gradient_to_angles(gx, gy):
    """(dz/dx, dz/dy) with x = east, y = north -> (tilt, azimuth 0 = N clockwise)."""
    tilt = np.degrees(np.arctan(np.hypot(gx, gy)))
    # the plane faces downslope, i.e. towards -gradient
    azimuth = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    return float(tilt), float(azimuth)


def _subset_gradients(x, y, z, p, rng):
    n = len(z)
    idx = rng.integers(0, n, size=(p.n_subsets, max(3, p.subset_size)))
    P = np.stack([x, y, z], axis=1)[idx]  # (M, k, 3)
    if P.shape[1] == 3:
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        spread = np.abs(nrm[:, 2]) / 2.0  # projected triangle area
        ok = spread >= p.min_subset_spread
        nz = np.where(ok, nrm[:, 2], 1.0)
        gx, gy = -nrm[:, 0] / nz, -nrm[:, 1] / nz
    else:
        C = P - P.mean(axis=1, keepdims=True)
        sxx, syy, sxy = (C[..., 0] ** 2).sum(1), (C[..., 1] ** 2).sum(1), (C[..., 0] * C[..., 1]).sum(1)
        sxz, syz = (C[..., 0] * C[..., 2]).sum(1), (C[..., 1] * C[..., 2]).sum(1)
        det = sxx * syy - sxy ** 2
        k = P.shape[1]
        ok = np.sqrt(np.maximum(det, 0)) / k >= p.min_subset_spread / 2
        d = np.where(ok, det, 1.0)
        gx, gy = (syy * sxz - sxy * syz) / d, (sxx * syz - sxy * sxz) / d
    gmax = np.tan(np.radians(p.search_max_tilt))
    ok &= np.isfinite(gx) & np.isfinite(gy) & (np.hypot(gx, gy) <= gmax)
    return gx[ok], gy[ok]


def _find_modes(gx, gy, p):
    """Peaks of the smoothed 2D histogram of the gradients, refined by medians."""
    gmax = np.tan(np.radians(p.search_max_tilt))
    step = p.bandwidth / 2.0
    nb = int(np.ceil(2 * gmax / step))
    H, xe, ye = np.histogram2d(gx, gy, bins=nb, range=[[-gmax, gmax], [-gmax, gmax]])
    Hs = ndimage.gaussian_filter(H, sigma=p.bandwidth / step)
    if Hs.max() <= 0:
        return []
    size = 2 * int(np.ceil(2 * p.bandwidth / step)) + 1
    peaks = np.argwhere((Hs == ndimage.maximum_filter(Hs, size=size)) &
                        (Hs >= p.peak_min_ratio * Hs.max()))
    peaks = sorted(peaks, key=lambda ij: -Hs[ij[0], ij[1]])[:p.max_modes]
    centers = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    modes = []
    for i, j in peaks:
        cx, cy = centers[0][i], centers[1][j]
        near = np.hypot(gx - cx, gy - cy) <= 1.5 * p.bandwidth
        if near.sum() >= 3:
            cx, cy = np.median(gx[near]), np.median(gy[near])
        modes.append((cx, cy))
    return modes


def _intercept(r, tol):
    """Densest value of the residuals r = z - gx x - gy y (robust intercept)."""
    lo, hi = np.min(r), np.max(r)
    nb = int(min(2000, max(1, np.ceil((hi - lo) / (tol / 3)))))
    h, e = np.histogram(r, bins=nb)
    h = np.convolve(h, np.ones(3), mode="same")
    c0 = (e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2
    near = np.abs(r - c0) <= tol
    return float(np.median(r[near])) if near.any() else float(c0)


def _assign(x, y, z, planes, tol):
    """Label of the closest plane for each pixel (-1 if farther than tol)."""
    d = np.stack([np.abs(z - (gx * x + gy * y + c)) for gx, gy, c in planes])
    lab = np.argmin(d, axis=0)
    lab[d[lab, np.arange(len(z))] > tol] = -1
    return lab


def _lstsq(x, y, z):
    A = np.column_stack([x, y, np.ones_like(x)])
    (gx, gy, c), *_ = np.linalg.lstsq(A, z, rcond=None)
    return gx, gy, c


def fit_planes(x, y, z, params=None):
    """
    Robust multi-plane fit on scattered points (x, y in metres, z elevations).

    Returns (planes, labels): planes is a list of dicts sorted by decreasing
    share (gx, gy, c, tilt, azimuth, share, n_pixels, rmse); labels gives the
    plane index of each point (-1 for outliers).
    """
    p = TheilSenParams.from_dict(params)
    rng = np.random.default_rng(p.seed)
    # scale the degeneracy threshold to the extent of the points (small
    # polygons / fine rasters)
    hull = shapely.MultiPoint(np.column_stack([x, y])).convex_hull.area
    p = replace(p, min_subset_spread=min(p.min_subset_spread, 0.05 * hull))
    gx, gy = _subset_gradients(x, y, z, p, rng)
    if len(gx) < 3:
        return [], np.full(len(z), -1)
    modes = _find_modes(gx, gy, p)
    planes = [(mx, my, _intercept(z - mx * x - my * y, p.inlier_tol)) for mx, my in modes]
    if not planes:
        return [], np.full(len(z), -1)

    for _ in range(3):
        lab = _assign(x, y, z, planes, p.inlier_tol)
        n = len(z)
        # drop planes with too small a share, refit the others on their pixels
        keep = [k for k in range(len(planes)) if (lab == k).sum() >= max(3, p.min_mode_share * n)]
        if not keep:
            keep = [int(np.bincount(lab[lab >= 0]).argmax())] if (lab >= 0).any() else []
        if not keep:
            return [], np.full(n, -1)
        planes = [_lstsq(x[lab == k], y[lab == k], z[lab == k]) if (lab == k).sum() >= 3 else planes[k]
                  for k in keep]
    lab = _assign(x, y, z, planes, p.inlier_tol)

    out = []
    for k, (pgx, pgy, c) in enumerate(planes):
        sel = lab == k
        if not sel.any():
            continue
        res = z[sel] - (pgx * x[sel] + pgy * y[sel] + c)
        tilt, az = gradient_to_angles(pgx, pgy)
        out.append(dict(gx=float(pgx), gy=float(pgy), c=float(c), tilt=tilt, azimuth=az,
                        share=float(sel.sum() / len(z)),
                        n_pixels=int(sel.sum()), rmse=float(np.sqrt(np.mean(res ** 2))), _k=k))
    order = sorted(range(len(out)), key=lambda i: -out[i]["share"])
    remap = np.full(len(planes), -1)
    for new, i in enumerate(order):
        remap[out[i]["_k"]] = new
    labels = np.where(lab >= 0, remap[np.maximum(lab, 0)], -1)
    planes = [{k: v for k, v in out[i].items() if k != "_k"} for i in order]
    return planes, labels


def _nan_result(status, n_pixels=0, **extra):
    r = dict(status=status, inspect=False, tilt=np.nan, azimuth=np.nan, n_pixels=n_pixels,
             n_modes=0, primary_share=np.nan, inlier_frac=np.nan, rmse=np.nan,
             roughness=np.nan, offset_suspected=False, modes=[])
    r.update(extra)
    return r


def estimate(z, transform, mask, edge_distance=None, params=None, return_labels=False):
    """
    Estimates tilt and azimuth from an elevation grid.

    z             : 2D elevations (NaN = no data)
    transform     : affine transform of the grid (metres)
    mask          : boolean grid, pixels to use (polygon, usually eroded)
    edge_distance : optional grid, distance (m) of each pixel to the polygon's
                    edge; enables the edge-band test for secondary planes
    return_labels : also return (rows, cols, labels, planes) for splitting

    Returns a dict: status, inspect, tilt, azimuth, n_pixels, n_modes,
    primary_share, inlier_frac, rmse, roughness, offset_suspected, modes
    (list of {tilt, azimuth, share}).
    """
    p = TheilSenParams.from_dict(params)
    valid = mask & np.isfinite(z)
    rows, cols = np.nonzero(valid)
    n = len(rows)
    if n < p.min_pixels or n * abs(transform.a * transform.e) < p.min_area:
        r = _nan_result(TOO_FEW_POINTS, n)
        return (r, None) if return_labels else r

    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    x0, y0 = x.mean(), y.mean()  # centre for numerical stability
    x, y, zz = x - x0, y - y0, z[rows, cols]

    planes, labels = fit_planes(x, y, zz, p)
    if not planes:
        r = _nan_result(FIT_FAILED, n)
        return (r, None) if return_labels else r

    pixel_area = abs(transform.a * transform.e)
    primary = planes[0]
    resid = zz - (primary["gx"] * x + primary["gy"] * y + primary["c"])
    roughness = float(1.4826 * np.median(np.abs(resid - np.median(resid))))

    secondaries = planes[1:]
    offset_suspected = False
    multimodal = False
    real_planes = [0]  # planes considered as genuine roof planes (used to split)
    for k, s in enumerate(secondaries, start=1):
        implausible = s["tilt"] > p.max_tilt
        small = s["n_pixels"] * pixel_area < p.min_part_area
        on_edge = False
        if edge_distance is not None:
            d = edge_distance[rows[labels == k], cols[labels == k]]
            on_edge = (d <= p.edge_band).mean() >= p.edge_frac
        dominated = primary["share"] >= p.dominant_share
        if implausible or small or (on_edge and dominated):
            offset_suspected = True
        else:
            multimodal = True
            real_planes.append(k)

    modes = [dict(tilt=m["tilt"], azimuth=m["azimuth"], share=m["share"]) for m in planes]
    common = dict(n_pixels=n, n_modes=len(planes), primary_share=primary["share"],
                  inlier_frac=float((labels >= 0).mean()), roughness=roughness, modes=modes)
    if multimodal:
        r = _nan_result(MULTIMODAL, inspect=True, offset_suspected=offset_suspected, **common)
    else:
        tilt, az = primary["tilt"], primary["azimuth"]
        status = OK
        if tilt < p.flat_tilt:
            status, az = FLAT, np.nan
        r = dict(status=status, inspect=False, tilt=tilt, azimuth=az, rmse=primary["rmse"],
                 offset_suspected=offset_suspected, **common)
    if return_labels:
        return r, dict(rows=rows, cols=cols, labels=labels, planes=planes,
                       real_planes=real_planes)
    return r
