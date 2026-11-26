# PyPVRoof

A Python package for extracting characteristics of rooftop PV installations.

## Installation

### Basic Installation

You can install the package using pip:

```bash
pip install pypvroof
```

GDAL is no longer required: rasters are read with `rasterio`, whose wheels ship their own GDAL.

### Installation from Source

```bash
git clone https://github.com/gabrielkasmi/pypvroof.git
cd pypvroof
pip install -e ".[dev]"
```

## System Requirements

- Python >= 3.9
- Core dependencies are automatically installed

## Documentation

- [API Documentation](docs/api.md)
- [Tutorials](docs/tutorials/)
- [Examples](examples/)
- [Hands-on Notebook](notebooks/hands-on.ipynb)

## Quick Start

The supplementary data is accessible on our Zenodo repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7586879.svg)](https://doi.org/10.5281/zenodo.7586879)

```python
import geojson
from pypvroof import MetadataExtraction


# load the data 
index =1356
arrays=geojson.load(open("path/to/geojson/with/arrays"))
case=arrays['features'][index]


# Example parameters dictionary
# choose your own methods
params = {
    "azimuth-method": "bounding-box", # choose between bounding-box and theil-sen
    "tilt-method": "lut", # choose between constant, lut and theil-sen
#    "dsm-service": "ign", # DSM used by theil-sen: "ign" (IGN LiDAR HD), a folder of GeoTIFFs, or a DSMService
    "regression-type": "linear", # choose between constant, linear or clustered
#    "constant-tilt": 30, # default parameters if no lookup table of DEM
#    "default-coefficient": 1/(6.5) # default parameter if no data to calibrate the linear regression coefficients
}

# initialize the object for extracting the metadata
extractor = MetadataExtraction(p=params)

# extract all characteristics at once for a single polygon
characteristics = extractor.extract_all_characteristics(case)


# or if you want a dataframe with all the characteristics for multiple polygons

dataframe = extractor.extract_all_characteristics(arrays)
```

### Tilt and azimuth from LiDAR (Theil-Sen)

The tilt and azimuth of the roof plane under each polygon can be measured on a
digital surface model (DSM). In France, the IGN LiDAR HD surface model (50 cm)
is queried on the fly, one small request per polygon:

```python
from pypvroof import characterize_roofs, IGNLidarService, LocalRasterService

out = characterize_roofs("detections.parquet", IGNLidarService(), workers=8)
# or, on local rasters:
out = characterize_roofs(arrays, LocalRasterService("path/to/dsm_tiles"), split=True)
```

The input can be a GeoDataFrame, a GeoJSON FeatureCollection, a list of
features or shapely polygons, or a file path. The output is a GeoDataFrame
with one row per polygon (several if `split=True` and the polygon covers
several roof planes), the columns `tilt`, `azimuth`, `status`, `inspect`,
`surface_proj`, `surface`, diagnostics, and the service metadata (`meta_*`,
e.g. the LiDAR acquisition dates). No fallback is applied: when no value can
be measured, `status` says why. See [docs/api.md](docs/api.md#dsm-services-and-theil-sen).

Azimuths are in degrees, **0 = north, clockwise** (90 = east, 180 = south).

## Repository Structure

```
pypvroof/
├── src/                    # Package source code
├── examples/              # Example scripts
│   ├── basic_usage.py
│   └── advanced_usage.py
├── notebooks/            # Jupyter notebooks
│   ├── hands-on.ipynb
│   └── advanced_features.ipynb
├── docs/                # Additional documentation
│   ├── api.md
│   └── tutorials/
├── tests/              # Test suite
├── setup.py
├── README.md
└── LICENSE
```

## Features

- Extract PV roof characteristics from GeoJSON polygons (or any geo table)
- Tilt and azimuth measured on LiDAR / DSM data (IGN LiDAR HD or local GeoTIFFs), in bulk
- Support for multiple computation methods
- Custom lookup table support
- Flexible parameter configuration


The package is shipped with the lookup table for France used in [3] (to be replaced by a model trained on LiDAR measurements) and array metadata from BDPV database and with a dataset of PV systems characteristics coming from BDPV if you want to use the linear regression for deriving the installed capacity of the system. 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Overview and motivation

PyPVRoof is an all-in-one approach for extracting metadata of rooftop PV installations. The approach is modular, depending on the data available, we use different methods to extract these characteristics. The user only has to set his preferred parameters depending on the data available and the module will automatically proceed a single polygon or a complete `.geojson` file. We extract the following characteristics:

* Localization (latitude, longitude)
* Tilt angle
* Azimuth angle
* Surface
* Installed capacity

These characteristics can be deduced from overhead imagery and some additional data (a PV registry or surface models). Besides, these characteristics are sufficient for a broad range of application, e.g. surveying [1](https://arxiv.org/abs/2009.05738) or regional PV estimation [2](https://www.sciencedirect.com/science/article/abs/pii/S0038092X18308211)

This software offers practitionners a fast and efficient way to extract installations metadata to generate consistent registries. This work expands and completes the characteristics extraction module of [3](https://arxiv.org/abs/2207.07466). 

The flowchart of the package is summarized below:

<p align="center">
<img src="assets/flowchart.png" width=700px>
</p>

## Citation

If you wish to use this work, please cite us as: 

```
@article{tremenbert2023pypvroof,
  title={PyPVRoof: a Python package for extracting the characteristics of rooftop PV installations using remote sensing data},
  author={Tr{\'e}menbert, Yann and Kasmi, Gabriel and Dubus, Laurent and Saint-Drenan, Yves-Marie and Blanc, Philippe},
  journal={arXiv preprint arXiv:2309.07143},
  year={2023}
}
```

Like this work ? Do not hesitate to <a class="github-button" href="https://github.com/gabrielkasmi/pypvroof" data-icon="octicon-star" aria-label="Star gabrielkasmi/pypvroof on GitHub">star</a> us !

## References

[1] De Jong, T., Bromuri, S., Chang, X., Debusschere, M., Rosenski, N., Schartner, C., ... & Curier, L. (2020). Monitoring spatial sustainable development: semi-automated analysis of satellite and aerial images for energy transition and sustainability indicators. arXiv preprint arXiv:2009.05738.

[2] Killinger, S., Lingfors, D., Saint-Drenan, Y. M., Moraitis, P., van Sark, W., Taylor, J., ... & Bright, J. M. (2018). On the search for representative characteristics of PV systems: Data collection and analysis of PV system azimuth, tilt, capacity, yield and shading. Solar Energy, 173, 1087-1106.

[3] Kasmi, G., Dubus, L., Blanc, P., & Saint-Drenan, Y. M. (2022, September). Towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping. In Workshop on Machine Learning for Earth Observation (MACLEAN), in Conjunction with the ECML/PKDD 2022.
