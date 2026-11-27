from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypvroof",
    version="0.2.1",
    author="Gabriel Kasmi",
    author_email="gabriel.kasmi@minesparis.psl.eu",
    description="Characteristics (tilt, azimuth, installed capacity) of rooftop PV systems from their polygons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielkasmi/pypvroof",
    packages=["pypvroof"],
    package_data={
        "pypvroof": ["data/*.json", "data/*.csv"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "shapely>=2.0.0",
        "pyproj>=3.0.0",
        "rasterio>=1.3.0",
        "geopandas>=0.14.0",
        "pyarrow>=10.0",
        "area>=1.1.0",
        "geojson>=2.5.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4.0", "pillow"],
        "dev": ["pytest>=6.0", "matplotlib>=3.4.0", "pillow"],
    },
) 