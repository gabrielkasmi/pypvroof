# -*- coding: utf-8 -*-
"""
Splitting of a multimodal polygon (covering several roof planes) into parts
that are homogeneous in tilt and azimuth.

Each pixel was assigned to a plane by the estimator. Pixels assigned to no
real plane (outliers, or planes dismissed as edge effects / obstacles) are
given the label of the nearest assigned pixel, so that the parts tile the
polygon. Connected components smaller than `min_part_area` are merged into
their neighbours. The label raster is then vectorized and intersected with
the original polygon; internal boundaries follow the pixel grid.
"""

import numpy as np
import shapely
from scipy import ndimage

from . import theilsen


def _fill_nearest(L, domain):
    """Gives unlabeled pixels of `domain` the label of the nearest labeled pixel."""
    unlabeled = L < 0
    if unlabeled.all():
        return L
    _, (ii, jj) = ndimage.distance_transform_edt(unlabeled, return_indices=True)
    out = L[ii, jj]
    return np.where(domain, out, -1)


def split_polygon(geometry, patch, full_mask, mask, fit, params=None):
    """
    Returns a list of (part geometry, result dict) in the CRS of `patch`, or
    an empty list if the polygon cannot be split into at least two parts.
    """
    p = theilsen.TheilSenParams.from_dict(params)
    rows, cols, labels, planes = fit["rows"], fit["cols"], fit["labels"], fit["planes"]
    real = fit.get("real_planes", list(range(len(planes))))
    if len(real) < 2:
        return []

    # domain: the polygon's pixels plus a 1-pixel ring, so that the parts
    # still cover the polygon once intersected with it
    domain = ndimage.binary_dilation(full_mask, structure=np.ones((3, 3), bool))
    L = np.full(patch.z.shape, -1, dtype=np.int32)
    keep = np.isin(labels, real)
    L[rows[keep], cols[keep]] = labels[keep]
    L = _fill_nearest(L, domain)

    xres, yres = patch.resolution
    min_px = p.min_part_area / (xres * yres)
    for _ in range(3):
        changed = False
        for k in np.unique(L[L >= 0]):
            comp, n = ndimage.label(L == k)
            sizes = ndimage.sum(np.ones_like(comp), comp, index=np.arange(1, n + 1))
            for c, size in enumerate(sizes, start=1):
                if size < min_px and n > 0:
                    L[comp == c] = -1
                    changed = True
        if not changed:
            break
        if (L < 0).all():
            return []
        L = _fill_nearest(L, domain)

    present = [int(k) for k in np.unique(L[L >= 0])]
    if len(present) < 2:
        return []

    # vectorize: union of the pixel squares of each label (pure shapely, thread safe)
    t = patch.transform
    by_label = {}
    for k in present:
        rr, cc = np.nonzero(L == k)
        x0 = t.c + cc * t.a
        y0 = t.f + rr * t.e
        cells = shapely.box(x0, y0 + t.e, x0 + t.a, y0)
        by_label[k] = cells

    n_total = len(rows)
    parts = []
    for k in present:
        with np.errstate(invalid="ignore"):
            part = shapely.union_all(by_label[k]).intersection(geometry)
        if part.is_empty or part.area < p.min_part_area:
            continue
        pl = planes[k]
        tilt, az, status = pl["tilt"], pl["azimuth"], theilsen.OK
        if tilt < p.flat_tilt:
            status, az = theilsen.FLAT, np.nan
        parts.append((part, dict(
            status=status, inspect=False, tilt=tilt, azimuth=az, n_pixels=pl["n_pixels"],
            n_modes=1, primary_share=pl["n_pixels"] / n_total if n_total else np.nan,
            inlier_frac=np.nan, rmse=pl["rmse"], roughness=np.nan, offset_suspected=False,
            modes=[dict(tilt=pl["tilt"], azimuth=pl["azimuth"], share=pl["share"])])))
    return parts if len(parts) >= 2 else []
