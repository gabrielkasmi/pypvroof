"""Synthetic DSM service for the tests: elevations given by a function z(x, y)."""
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from pypvroof import dsm

X0, Y0 = 651000.0, 6862000.0  # somewhere in Lambert-93


class FunctionService(dsm.DSMService):
    crs = CRS.from_epsg(2154)

    def __init__(self, func, res=0.5, margin=1.0, noise=0.0, seed=0, meta=None):
        self.func, self.res, self.margin, self.noise = func, res, margin, noise
        self.rng = np.random.default_rng(seed)
        self.meta = meta or {"source": "synthetic"}

    def fetch(self, geometry):
        minx, miny, maxx, maxy = geometry.bounds
        minx, miny = np.floor((minx - self.margin) / self.res) * self.res, np.floor((miny - self.margin) / self.res) * self.res
        maxx, maxy = np.ceil((maxx + self.margin) / self.res) * self.res, np.ceil((maxy + self.margin) / self.res) * self.res
        w, h = int(round((maxx - minx) / self.res)), int(round((maxy - miny) / self.res))
        t = Affine(self.res, 0, minx, 0, -self.res, maxy)
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        x = minx + (cols + 0.5) * self.res
        y = maxy - (rows + 0.5) * self.res
        z = self.func(x, y).astype(float)
        if self.noise:
            z += self.rng.normal(0, self.noise, z.shape)
        return dsm.FetchResult(dsm.OK, dsm.DSMPatch(z, t, self.crs), dict(self.meta))


def plane(tilt, azimuth, h=10.0):
    """Roof plane facing `azimuth` (0 = N, clockwise) with the given tilt."""
    s = np.tan(np.radians(tilt))
    # downslope direction (east, north) components
    de, dn = np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))
    return lambda x, y: h - s * ((x - X0) * de + (y - Y0) * dn)


def gable(tilt, ridge_y, h=12.0):
    """East-west ridge at y = ridge_y: south pan faces 180, north pan faces 0."""
    s = np.tan(np.radians(tilt))
    return lambda x, y: h - s * np.abs(y - ridge_y)


def rect(x, y, w, h):
    from shapely.geometry import box
    return box(x, y, x + w, y + h)
