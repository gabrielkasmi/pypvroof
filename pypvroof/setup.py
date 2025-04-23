from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypvroof",
    version="0.1.0",
    author="Gabriel",
    author_email="your.email@example.com",
    description="A Python package for PV roof mapping and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pypvroof",
    packages=find_packages(exclude=["examples", "notebooks", "docs", "tests"]),
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