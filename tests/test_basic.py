"""
Basic tests for pypvroof
"""
import unittest
import geojson
from pypvroof import MetadataExtraction

class TestMetadataExtraction(unittest.TestCase):
    def setUp(self):
        # Basic parameters
        self.params = {
            "azimuth-method": "bounding-box",
            "tilt-method": "constant",
            "regression-type": "constant",
            "has-data": False,
            "has-dem": False,
            "constant-tilt": 30,
            "default-coefficient": 1/(6.5)
        }

        # Sample polygon
        self.polygon = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            }
        }

    def test_initialization(self):
        """Test that the extractor initializes correctly"""
        extraction = MetadataExtraction(p=self.params)
        self.assertIsNotNone(extraction)

    def test_extract_all_characteristics(self):
        """Test that characteristics can be extracted"""
        extraction = MetadataExtraction(p=self.params)
        characteristics = extraction.extract_all_characteristics(self.polygon, save_ext=False)
        self.assertIsNotNone(characteristics)

if __name__ == '__main__':
    unittest.main() 