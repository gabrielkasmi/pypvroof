# -*- coding: utf-8 -*-

# libraries
import os
import json
from importlib import resources

import pandas as pd
import numpy as np

from . import methods


def _package_data_path(name):
    """Absolute path of a file shipped in pypvroof/data."""
    return str(resources.files('pypvroof').joinpath('data', name))


class MetadataExtraction():
    """
    Class wrapping all the functionnalities of the module. 

    These functionnalities include:

    - running the computation of all the characteristics,
    - running the computation of a single characteristic
    """

    def __init__(self, p=None, lut=None):
        """
        Initialize the metadata extraction with parameters and optional lookup table.
        
        Parameters:
        -----------
        p : dict
            Dictionary of parameters for the extraction process
        lut : str or dict, optional
            Path to a custom lookup table JSON file or a dictionary containing the lookup table.
            If None, the default LUT-France will be used.
        """
        self.params = p or {}
        
        # Load lookup table
        if lut is None:
            # Use default LUT-France
            lut_file = _package_data_path('lut_france.json')
            with open(lut_file, 'r') as f:
                self.lut = json.load(f)
        elif isinstance(lut, str):
            # Load custom LUT from file
            with open(lut, 'r') as f:
                self.lut = json.load(f)
        else:
            # Use provided LUT dictionary
            self.lut = lut

        # Initialize other attributes from params
        self._lut_from_constructor = lut is not None
        self._initialize_from_params()

        # Fill self.params with default values if p is None, or add missing keys if p exists
        if p is None:
            self.params = {}
        
        # Ensure all required keys exist with default values
        default_params = {
            "has-data": False,
            "has-dem": False,
            "tilt-method": "constant",
            "azimuth-method": "bounding-box", 
            "regression-type": "constant",
            "default-coefficient": 1/6.5,
            "constant-tilt": 30,
            "regression-clusters": 4,
            "data-directory": None,
            "data-name": None,
            "lut-table": None
        }

        # Update params with defaults for any missing keys
        for key, default_val in default_params.items():
            if key not in self.params:
                self.params[key] = default_val

    def _initialize_from_params(self):
        """Initialize attributes from the parameters dictionary."""


        self.azimuth_method = self.params.get('azimuth-method', 'bounding-box')
        self.tilt_method = self.params.get('tilt-method', 'constant')
        self.regression_type = self.params.get('regression-type', 'constant')
        self.has_data = self.params.get('has-data', False)
        self.has_dem = self.params.get('has-dem', False)

        # used only if regression type is clustered
        self.regression_clusters = self.params.get('regression-clusters', 4)


        # used only if tilt method is constant or 
        # installec capacity method is constant coefficient
        self.data=None
        self.constant_tilt = self.params.get('constant-tilt', 30)
        self.default_coefficient = self.params.get('default-coefficient', 1/6.5)



        # if tilt or azimuth method is "theil-sen", a DSM service is needed
        # (params "dsm-service": a DSMService, a service name ("fr-lidar"...), or a folder of GeoTIFFs)

        self._theil_sen = None
        if self._uses_theil_sen():
            # validates the service specification ("dsm-service" or "raster-folder")
            self._theil_sen = methods.TheilSen(self.params)
        self.workers = self.params.get('workers', 8)
        self._regression = None

        # methods that require auxiliary data: linear or clustered regression
        # and look up table

        if self.regression_type == 'linear' or self.regression_type == 'clustered':
            # load the data frame that contains the data
            # either from the default data/bdappv-metadata.csv
            # or from the user provided data

            if self.has_data:
                self.data_dir = self.params.get('data-directory')
                self.data_name = self.params.get('data-name')
            
            else:
                self.data_dir = _package_data_path('')
                self.data_name = 'bdappv-metadata.csv'


        if self.tilt_method == 'lut':
            import warnings
            warnings.warn("tilt-method 'lut' is deprecated and will be removed: the look-up table "
                          "is noisy. Use 'theil-sen' with a surface model (see "
                          "pypvroof.available_services()) or 'constant'.",
                          DeprecationWarning, stacklevel=3)
            self.lut_table = self.params.get('lut-table', None)
            # a LUT passed to the constructor (dict or path) takes precedence
            # over the default one
            if self.lut_table is None and self._lut_from_constructor:
                self.lut_table = self.lut


        self._load_data()
                

    def _load_data(self):
        """Load and process data if needed."""

        if self.regression_type == 'linear' or self.regression_type == 'clustered':
            # load the data file on which the regression coefficients are fitted 
            self.data = pd.read_csv(os.path.join(self.data_dir, self.data_name))

        # if tilt_method is lut, load the lookup table
        if self.tilt_method == 'lut':

            # either the default LUT, a dict, or a path (absolute, or relative
            # to the data directory)
            if self.lut_table is None:
                with open(_package_data_path('lut_france.json')) as f:
                    self.lut_table = json.load(f)
            elif isinstance(self.lut_table, str):
                path = self.lut_table
                if not os.path.isfile(path) and getattr(self, 'data_dir', None):
                    path = os.path.join(self.data_dir, path)
                with open(path) as f:
                    self.lut_table = json.load(f)

    def _uses_theil_sen(self):
        return self.tilt_method == 'theil-sen' or self.azimuth_method == 'theil-sen'

    @staticmethod
    def _is_single_polygon(data):
        return (isinstance(data, dict) and data.get('type', 'Feature') == 'Feature'
                and isinstance(data.get('geometry'), dict)
                and data['geometry'].get('type') in ('Polygon', 'MultiPolygon'))

    def extract_all_characteristics(self, data, crs=None):
        """
        Extracts all the characteristics.

        - a single GeoJSON feature (Polygon / MultiPolygon, lon/lat) returns a
          tuple (lon, lat, tilt, azimuth, installed_capacity, surface)
        - anything else accepted by pypvroof.geometry.to_geodataframe (a
          FeatureCollection, a GeoDataFrame, a list of features, a path...)
          returns a DataFrame with the columns lon, lat, tilt, azimuth,
          installed_capacity, surface, one row per input polygon. When the
          theil-sen method is used, the columns status, inspect and the
          service metadata (meta_*) are added; polygons are processed
          concurrently (param "workers", default 8).

        `crs` is the CRS of inputs that do not carry one (default EPSG:4326).
        Tilt / azimuth / capacity are NaN when the DSM estimation fails: no
        fallback is applied, see the status column.
        """
        if self._is_single_polygon(data):
            if self._uses_theil_sen():
                r = self._theil_sen.characterize(data)
                tilt = r['tilt'] if self.tilt_method == 'theil-sen' else self.compute_tilt(data)
                azimuth = r['azimuth'] if self.azimuth_method == 'theil-sen' else self.compute_azimuth(data)
            else:
                tilt = self.compute_tilt(data)
                azimuth = self.compute_azimuth(data)
            surface = self.compute_surface(data, tilt)
            installed_capacity = self.compute_installed_capacity(data, tilt, surface)
            lon, lat = self.return_coordinates(data)
            return (lon, lat, tilt, azimuth, installed_capacity, surface)

        from shapely.geometry import mapping
        from .geometry import to_geodataframe

        gdf = to_geodataframe(data, crs=crs or self.params.get('crs'))
        lonlat = gdf.to_crs('EPSG:4326')
        features = [{'type': 'Feature', 'geometry': mapping(g), 'properties': {}}
                    for g in lonlat.geometry]

        ts = None
        if self._uses_theil_sen():
            from .bulk import characterize_roofs
            ts = characterize_roofs(gdf[[gdf.geometry.name]], self._theil_sen.service,
                                    params=self._theil_sen.estimator_params,
                                    workers=self.workers)

        lines = []
        for i, feature in enumerate(features):
            lon, lat = self.return_coordinates(feature)
            tilt = ts['tilt'].iloc[i] if self.tilt_method == 'theil-sen' else self.compute_tilt(feature)
            azimuth = (ts['azimuth'].iloc[i] if self.azimuth_method == 'theil-sen'
                       else self.compute_azimuth(feature))
            surface = self.compute_surface(feature, tilt)
            installed_capacity = self.compute_installed_capacity(feature, tilt, surface)
            lines.append((lon, lat, tilt, azimuth, installed_capacity, surface))

        df = pd.DataFrame(lines, columns=['lon', 'lat', 'tilt', 'azimuth', 'installed_capacity', 'surface'])
        if ts is not None:
            df['status'] = ts['status'].values
            df['inspect'] = ts['inspect'].values
            for c in ts.columns:
                if c.startswith('meta_'):
                    df[c] = ts[c].values
        return df


    def return_coordinates(self, polygon):
        """Centroid (lon, lat) of a GeoJSON feature."""
        from shapely.geometry import shape
        c = shape(polygon['geometry']).centroid
        return c.x, c.y

    def compute_tilt(self, polygon):
        """
        Computes the tilt of a polygon passed as input using the method
        specified during class initialization.

        Args:
            polygon (dict): The geojson dictionary containing the information and coordinates

        Returns:
            tilt (float): The tilt in degrees of the installation
        """
        # Use tilt method from initialization
        if self.tilt_method == "constant":
            tilt = self.constant_tilt

        elif self.tilt_method == 'lut':
            # compute the surface
            projected_surface = methods.compute_surface(polygon)

            # initialize the LUT
            lut = methods.LUT(lut=self.lut_table)

            # compute the tilt using the look up table
            tilt = lut.return_tilt(polygon, projected_surface)

        elif self.tilt_method == "theil-sen":
            # return only the tilt
            theil_sen = methods.TheilSen(self.params)
            _, tilt = theil_sen.estimate_tilt_and_azimuth(polygon)

        return tilt

    def compute_surface(self, polygon, tilt = None):
        """
        Computes the actual surface area of a solar installation by correcting the projected surface area 
        for tilt angle.

        Args:
            polygon (dict): A GeoJSON dictionary containing the polygon geometry
            tilt (float, optional): The tilt angle in degrees. If None, will be computed using the 
                                  method specified during class initialization.

        Returns:
            float: The actual surface area in square meters, corrected for tilt angle

        The actual surface area is calculated by dividing the projected surface area (as viewed from above) 
        by the cosine of the tilt angle. This accounts for the fact that tilted panels have a larger 
        actual surface area than their projection onto the ground.
        """

        # compute the projected surface
        if tilt is None:

            tilt = self.compute_tilt(polygon)

        projected_surface = methods.compute_surface(polygon)
        surface = projected_surface / np.cos(tilt * np.pi / 180)

        return surface


    def compute_azimuth(self, polygon):
        """
        computes the azimuth of the installation using the method specified at initialization
        """
        if self.azimuth_method == "theil-sen":
            theil_sen = methods.TheilSen(self.params)
            azimuth, _ = theil_sen.estimate_tilt_and_azimuth(polygon)

        if self.azimuth_method == "bounding-box":
            bounding_box = methods.BoudingBox()
            azimuth = bounding_box.compute_azimuth(polygon)

        return azimuth


    def compute_installed_capacity(self, polygon, tilt = None, surface = None):
        """
        Computes the installed capacity using the surface area.

        Args:
            polygon (dict): A GeoJSON dictionary containing the polygon geometry
            tilt (float, optional): The tilt angle in degrees. If None, will be computed
            surface (float, optional): The actual surface area. If None, will be computed

        Returns:
            float: The estimated installed capacity in kWp

        If surface is None:
            - If tilt is also None, computes both tilt and surface
            - If tilt is provided, uses it to compute surface
        If surface is provided, uses it directly to estimate capacity
        """
        if surface is None:
            if tilt is None:
                # Need to compute both tilt and surface
                tilt = self.compute_tilt(polygon)
            # Compute surface using the tilt (whether provided or computed)
            surface = self.compute_surface(polygon, tilt)

        # Compute installed capacity using the surface (coefficients fitted once)
        if self._regression is None:
            self._regression = methods.LinearRegression(params=self.params)
            self._regression.initialize_coefficients(self.data)
        installed_capacity = self._regression.return_installed_capacity(surface)

        return installed_capacity






