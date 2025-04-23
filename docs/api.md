# PyPVRoof API Documentation

## MetadataExtraction

The main class for extracting PV roof characteristics.

### Initialization

```python
from pypvroof import MetadataExtraction

# Basic initialization with default LUT-France
extraction = MetadataExtraction(p=params)

# With custom LUT
extraction = MetadataExtraction(p=params, lut=custom_lut)
```

#### Parameters

- `p` (dict): Dictionary of parameters for the extraction process
- `lut` (str or dict, optional): Path to a custom lookup table JSON file or a dictionary containing the lookup table. If None, the default LUT-France will be used.

### Methods

#### extract_all_characteristics

```python
characteristics = extraction.extract_all_characteristics(polygon, save_ext=False)
```

Extracts all characteristics (azimuth, tilt, surface, installed capacity) for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature
- `save_ext` (bool): Whether to save the results to a file

**Returns:**
- dict: Dictionary containing all extracted characteristics

#### compute_azimuth

```python
azimuth = extraction.compute_azimuth(polygon)
```

Computes the azimuth angle for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature

**Returns:**
- float: The computed azimuth angle

#### compute_tilt

```python
tilt = extraction.compute_tilt(polygon)
```

Computes the tilt angle for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature

**Returns:**
- float: The computed tilt angle

#### compute_surface

```python
surface = extraction.compute_surface(polygon, tilt=tilt)
```

Computes the surface area for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature
- `tilt` (float): The tilt angle of the surface

**Returns:**
- float: The computed surface area

#### compute_installed_capacity

```python
capacity = extraction.compute_installed_capacity(polygon, tilt=tilt, surface=surface)
```

Computes the installed capacity for a given polygon.

**Parameters:**
- `polygon` (dict): A GeoJSON polygon feature
- `tilt` (float): The tilt angle of the surface
- `surface` (float): The surface area

**Returns:**
- float: The computed installed capacity 