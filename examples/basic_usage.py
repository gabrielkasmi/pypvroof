"""
Basic usage example for pypvroof
"""
import geojson
from pypvroof import MetadataExtraction

def main():
    # Example parameters dictionary
    params = {
        "azimuth-method": "bounding-box",
        "tilt-method": "constant",
        "regression-type": "constant",
        "has-data": False,
        "has-dem": False,
        "constant-tilt": 30,
        "default-coefficient": 1/(6.5)
    }

    # Initialize the extractor (will use default LUT-France)
    extraction = MetadataExtraction(p=params)

    # Load your geojson file
    with open('path/to/your/input.geojson', 'r') as f:
        polygons = geojson.load(f)

    # Process a single polygon
    polygon = polygons['features'][0]
    
    # Extract characteristics
    characteristics = extraction.extract_all_characteristics(polygon, save_ext=False)
    print("Extracted characteristics:", characteristics)

if __name__ == "__main__":
    main() 