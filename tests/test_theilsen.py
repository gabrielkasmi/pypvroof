"""Theil-Sen estimator on synthetic roofs (no network)."""
import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from pypvroof.bulk import characterize_polygon
from pypvroof.theilsen import TheilSenParams, gradient_to_angles
from synthetic import FunctionService, X0, Y0, plane, gable, rect


def run(func, geom, noise=0.0, **kw):
    return characterize_polygon(geom, FunctionService(func, noise=noise), **kw)


@pytest.mark.parametrize("tilt,azimuth", [(30, 180), (35, 90), (20, 225), (45, 270), (15, 0)])
def test_single_plane(tilt, azimuth):
    [(_, r, _)] = run(plane(tilt, azimuth), rect(X0, Y0, 6, 4), noise=0.03)
    assert r["status"] == "ok"
    assert r["tilt"] == pytest.approx(tilt, abs=1.0)
    d = (r["azimuth"] - azimuth + 180) % 360 - 180
    assert abs(d) < 2.0


def test_angle_convention():
    # z increases northwards -> faces south (180); increases westwards -> faces east (90)
    assert gradient_to_angles(0.0, 1.0)[1] == pytest.approx(180)
    assert gradient_to_angles(-1.0, 0.0)[1] == pytest.approx(90)
    assert gradient_to_angles(0.0, -1.0)[1] == pytest.approx(0)
    assert gradient_to_angles(1.0, 0.0)[0] == pytest.approx(45)


def test_flat_roof():
    [(_, r, _)] = run(plane(1, 180), rect(X0, Y0, 10, 8), noise=0.02)
    assert r["status"] == "flat"
    assert np.isnan(r["azimuth"])
    assert r["tilt"] < 5


def test_too_few_points():
    [(_, r, _)] = run(plane(30, 180), rect(X0, Y0, 1.5, 1.5))
    assert r["status"] == "too_few_points"
    assert np.isnan(r["tilt"])


def test_robust_to_overhang_and_chimney():
    roof = plane(30, 180)

    def scene(x, y):
        z = np.where(x < X0 + 5, roof(x, y), 2.0)  # polygon overhangs the roof edge by 1 m
        chimney = (np.abs(x - (X0 + 3)) < 0.5) & (np.abs(y - (Y0 + 2)) < 0.5)
        return np.where(chimney, z + 1.5, z)

    [(_, r, _)] = run(scene, rect(X0, Y0, 6, 4), noise=0.03)
    assert r["status"] == "ok"
    assert r["tilt"] == pytest.approx(30, abs=1.5)


def test_gable_is_multimodal_and_splits():
    func, geom = gable(35, Y0 + 3), rect(X0, Y0, 8, 6)
    [(_, r, _)] = run(func, geom)
    assert r["status"] == "multimodal" and r["inspect"]
    assert np.isnan(r["tilt"])
    real = [m for m in r["modes"] if m["tilt"] < 60]
    assert sorted(round(m["azimuth"]) % 360 for m in real) == [0, 180]

    parts = run(func, geom, split=True)
    assert len(parts) == 2
    assert sorted(round(p[1]["azimuth"]) for p in parts) == [0, 180]
    assert all(p[1]["tilt"] == pytest.approx(35, abs=0.5) for p in parts)
    # the parts tile the polygon
    union = shapely.union_all([p[0] for p in parts])
    assert union.symmetric_difference(geom).area < 1e-6


def test_split_tiles_skewed_polygon():
    geom = Polygon([(X0, Y0), (X0 + 7.3, Y0 + 0.4), (X0 + 6.8, Y0 + 6.1), (X0 + 0.2, Y0 + 5.7)])
    parts = run(gable(35, Y0 + 3), geom, split=True)
    assert len(parts) == 2
    assert sum(p[0].area for p in parts) == pytest.approx(geom.area, rel=1e-6)


def test_ridge_band_on_edge_is_not_split():
    # the polygon crosses the ridge by 1.2 m: misregistration, not two installations
    [(_, r, _)] = run(gable(35, Y0 + 4.8), rect(X0, Y0, 8, 6), noise=0.03)
    assert r["status"] == "ok"
    assert r["azimuth"] == pytest.approx(180, abs=2)


def test_racks_on_flat_roof_are_flat_and_rough():
    def racks(x, y):
        u = (y - Y0) % 2.5
        return 8.0 + np.where(u < 1.5, u * np.tan(np.radians(12)), 0.0)

    [(_, r, _)] = run(racks, rect(X0, Y0, 10, 8), noise=0.02)
    assert r["status"] == "flat"
    assert r["roughness"] > 0.05


def test_params_validation():
    with pytest.raises(ValueError):
        TheilSenParams.from_dict({"nope": 1})
    assert TheilSenParams.from_dict({"n_subsets": 10}).n_subsets == 10
