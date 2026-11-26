"""
Basic tests for pypvroof
"""
import math
import unittest

from pypvroof import MetadataExtraction


def _feature(lon, lat, d=0.0001):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lon, lat], [lon + d, lat], [lon + d, lat + d / 2],
                             [lon, lat + d / 2], [lon, lat]]],
        },
    }


class TestMetadataExtraction(unittest.TestCase):
    def setUp(self):
        self.params = {
            "azimuth-method": "bounding-box",
            "tilt-method": "constant",
            "regression-type": "constant",
            "constant-tilt": 30,
            "default-coefficient": 1 / 6.5,
        }
        self.polygon = _feature(2.35, 48.85)

    def test_initialization(self):
        self.assertIsNotNone(MetadataExtraction(p=self.params))

    def test_extract_all_characteristics(self):
        lon, lat, tilt, az, kwp, surface = MetadataExtraction(p=self.params) \
            .extract_all_characteristics(self.polygon)
        self.assertEqual(tilt, 30)
        self.assertTrue(surface > 0 and kwp > 0)

    def test_feature_collection(self):
        fc = {"type": "FeatureCollection", "features": [self.polygon, _feature(5.0, 45.0)]}
        df = MetadataExtraction(p=self.params).extract_all_characteristics(fc)
        self.assertEqual(len(df), 2)

    def test_clustered_regression_uses_packaged_csv(self):
        p = dict(self.params, **{"regression-type": "clustered", "regression-clusters": 4})
        out = MetadataExtraction(p=p).extract_all_characteristics(self.polygon)
        self.assertFalse(math.isnan(out[4]))

    def test_lut_never_nan(self):
        # points on the edges of / outside the LUT grid used to give NaN tilts
        p = dict(self.params, **{"tilt-method": "lut"})
        ext = MetadataExtraction(p=p)
        for lon, lat in [(-5.0, 48.4), (8.2, 49.0), (3.0, 42.4), (9.5, 42.0), (-10, 60)]:
            tilt = ext.compute_tilt(_feature(lon, lat))
            self.assertFalse(math.isnan(tilt), (lon, lat))


if __name__ == '__main__':
    unittest.main()
