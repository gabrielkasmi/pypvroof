from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypvroof",
    version="0.1.0",
    author="Gabriel Kasmi",
    author_email="gabriel.kasmi@minesparis.psl.eu",
    description="A Python package for PV roof mapping and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielkasmi/pypvroof",
    packages=find_packages(),
    package_data={
        "pypvroof": ["data/*.json"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "geojson",
        "tqdm",
        "GDAL>=3.6.2",
        "shapely",
        "rasterio",
    ],
) 