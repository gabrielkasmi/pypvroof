from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypvroof",
    version="0.1.1",
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
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "geojson>=2.5.0",
        "tqdm>=4.62.0",
        "shapely>=1.8.0",
        "rasterio>=1.2.0",
    ],
    extras_require={
        "gdal": ["GDAL>=3.6.2"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
) 