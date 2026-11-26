from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypvroof",
    version="0.2.0",
    author="Gabriel Kasmi",
    author_email="gabriel.kasmi@minesparis.psl.eu",
    description="A Python package for PV roof mapping and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielkasmi/pypvroof",
    packages=find_packages(),
    package_data={
        "pypvroof": ["data/*.json", "data/*.csv"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "shapely>=2.0.0",
        "pyproj>=3.0.0",
        "rasterio>=1.3.0",
        "geopandas>=0.14.0",
        "area>=1.1.0",
        "geojson>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
) 