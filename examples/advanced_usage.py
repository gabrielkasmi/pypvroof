"""
Advanced usage example for pypvroof showing custom LUT usage
"""
import geojson
import json
from pypvroof import MetadataExtraction

def main():
    # Example parameters dictionary
    params = {
        "azimuth-method": "bounding-box",
        "tilt-method": "lut",  # Using LUT method
        "regression-type": "constant",
        "has-data": False,
        "has-dem": False
    }

    # Use the lookup table from data directory
    # set the path to your lookup table file if necessary
    # replace none by your table. Otherwise it will use
    # the default lookup table computed for France.
    custom_lut = None

    # Initialize the extractor with custom LUT
    extraction = MetadataExtraction(
        p=params,
        lut=custom_lut
    )

    # Load your geojson file
    with open('path/to/your/input.geojson', 'r') as f:
        polygons = geojson.load(f)

    # Process a single polygon
    polygon = polygons['features'][0]
    
    # Extract characteristics
    characteristics = extraction.extract_all_characteristics(polygon, save_ext=False)
    print("Extracted characteristics:", characteristics)

    # example on a look up table in France

    custom_lut = None

    # Initialize with LUT from file
    extraction_from_file = MetadataExtraction(
        p=params,
        lut=custom_lut
    )

    # Extract characteristics using LUT from file
    characteristics_from_file = extraction_from_file.extract_all_characteristics(polygon, save_ext=False)
    print("Characteristics using LUT from file:", characteristics_from_file)

if __name__ == "__main__":
    main() 