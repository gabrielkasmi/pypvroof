# -*- coding: utf-8 -*-

# libraries
import sys
import os 
import json
import pkg_resources
from pathlib import Path

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
import methods

import warnings
warnings.filterwarnings("ignore")


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
            lut_file = pkg_resources.resource_filename('pypvroof', 'data/lut_france.json')
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



        # if tilt or azimuth method is "theil-sen", check that the user specified a path 
        # to the DEM data in the parameters, otherwise throw an error

        if self.tilt_method=="theil-sen" or self.azimuth_method=="theil-sen":
            if 'raster-folder' not in self.params:
                raise ValueError("must input a path to the DEM data")

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
                self.data_dir = pkg_resources.resource_filename('pypvroof', 'data')
                self.data_name = 'bdappv-metadata.csv'

        
        if self.tilt_method == 'lut':
            self.lut_table = self.params.get('lut-table', None)


        self._load_data()
                

    def _load_data(self):
        """Load and process data if needed."""

        if self.regression_type == 'linear' or self.regression_type == 'clustered':
            # load the data file on which the regression coefficients are fitted 
            self.data = pd.read_csv(os.path.join(self.data_dir, self.data_name))

        # if tilt_method is lut, load the lookup table
        if self.tilt_method == 'lut':

            # go the the paths that is either the default one or the one provided by the user
            if self.lut_table is None:
                self.lut_table = json.load(open(pkg_resources.resource_filename('pypvroof', 'data/lut_france.json')))
            else:
                self.lut_table = json.load(open(os.path.join(self.data_dir, self.lut_table)))

    def extract_all_characteristics(self, data):
        """

        Extracts all the characteristics from data
        if data is a single array, then returns a tuple 
            (lon, lat, tilt, azimuth, installed_capacity, surface)

        if data is a geojson file, returns a dataframe with all the characteristics.

        """

        if isinstance(data, dict) and data.get('geometry', {}).get('type') == 'Polygon':

            tilt=self.compute_tilt(data)
            azimuth=self.compute_azimuth(data)
            surface=self.compute_surface(data, tilt)
            installed_capacity=self.compute_installed_capacity(data, tilt, surface)

            lon, lat=self.return_coordinates(data)

            return (lon, lat, tilt, azimuth, installed_capacity, surface)
        
        if isinstance(data, dict) and data.get('type') == 'FeatureCollection':

            lines = []
            for feature in data['features']:
                lon, lat=self.return_coordinates(feature)
                tilt=self.compute_tilt(feature)
                azimuth=self.compute_azimuth(feature)
                surface=self.compute_surface(feature, tilt)
                installed_capacity=self.compute_installed_capacity(feature, tilt, surface)

                lines.append((lon, lat, tilt, azimuth, installed_capacity, surface))

            return pd.DataFrame(lines, columns=['lon', 'lat', 'tilt', 'azimuth', 'installed_capacity', 'surface'])


    def return_coordinates(self, polygon):

        # get the localization
        coordinates = polygon["geometry"]['coordinates'][0]
        center = np.mean(coordinates, axis = 0) 
        lon, lat = center

        return lon, lat
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

        # Compute installed capacity using the surface
        linear_regression = methods.LinearRegression(params=self.params)
        linear_regression.initialize_coefficients(self.data)
        installed_capacity = linear_regression.return_installed_capacity(surface)

        return installed_capacity






