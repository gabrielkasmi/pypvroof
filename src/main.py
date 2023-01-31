# -*- coding: utf-8 -*-

# libraries
import sys
import os 

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

import pandas as pd
import geojson
import numpy as np
import tqdm
import methods, builders

import warnings
warnings.filterwarnings("ignore")


class MetadataExtraction():
    """
    Class wrapping all the functionnalities of the module. 

    These functionnalities include:

    - running the computation of all the characteristics,
    - running the computation of a single characteristic
    """

    def __init__(self, cf = None, p = None, lut = None) -> None:
        """
        initialization. Sets up the parameters based on either the configuration file or
        p, a dictionnary of parameters.

        The keys of p should match the variable names of the configuration file.

        parameters:
        : cf (dict) the configuration file (loaded). 
        : p (dict) the dictionnary that contains the parameters
        : lookup (dict) the dictionnary containing the lookup table and its marginals

        """

        # Load the lookup table if passed as input.
        self.lookup_table = lut

        # Case where initialization is done with
        # the configuration file
        if cf is not None:

            # add two attributes that will be passed as inputs for 
            # intermediary initalizations.

            self.cf = cf 
            self.params = None

            # get the accessible data from the configuration file
            self.source_data = cf.get('has-data')
            self.dem_data = cf.get('has-dem')

            # get the directory of the auxiliary dataframe
            # it should be a csv file
            self.data_dir = cf.get('data-directory')
            self.data_name = cf.get('data-name')

            # input the raw lookup table


            # format the dataset dataframe for subsequent use

            if self.source_data:

                self.dataset = pd.read_csv(os.path.join(self.data_dir, self.data_name))
                self.data = builders.format_dataset(self.dataset, cf = cf)
            else: # if no source data, then the data will be a None
                self.data = None

            # set up the methods for tilt and azimuth estimation
            self.tilt_method = cf.get('tilt-method')
            self.azimuth_method = cf.get('azimuth-method')

            # set up the method for the regression
            self.regression_method = cf.get('regression-type')

            # get the name of the main data file
            self.source_name = cf.get('source-data')

            # name of the output file.
            self.output_name = cf.get('output-name')

            # get constants if relevant
            if self.tilt_method == "constant":
                self.tilt_value = cf.get('constant-tilt')

            if self.regression_method == 'constant':
                self.surface_coeff = cf.get('default-coefficient')

        else:

            self.cf = None
            self.params = p

            if p is None: # case where the user will directly input the dictionnary of parameters when calling the methods
                pass

            else:
                # all values are initialized as None
                
                # auxliary data and dem 
                self.source_data = p['has-data']
                self.dem_data = p['has-dem']
                
                # directory and name of the auxiliary file
                # if relevant

                # instantiace the source data using the dictionnary of parameters 
                # passed as input. 

                if self.source_data:
                    
                    # directory and name of the aux file
                    # if relevant
                    self.data_dir = p['data-directory']
                    self.data_name = p['data-name']

                    self.dataset = pd.read_csv(os.path.join(self.data_dir, self.data_name))
                    self.data = builders.format_dataset(self.dataset, params = p)

                else: # if no source data, then the data will be a None
                    self.data = None

                # methods
                self.tilt_method = p['tilt-method']
                self.azimuth_method = p['azimuth-method']
                self.regression_method = p['regression-type']

                self.output_name = p['output-name']

                # get constants if relevant
                if p['tilt-method'] == "constant":
                    self.tilt_value = p['constant-tilt']

                if p['regression-type'] == 'constant':
                    self.surface_coeff = p['default-coefficient']

    def extract_all_characteristics(self, input_data = None, save_ext = True):
        """
        Runs all the characteristics according to the parameters specified in the input

        If the input_data is None, it will directly open the file provided 
        in the initialization. 

        Otherwise, the user should provide a geojson file or a single polygon

        save_ext specify whether the outputs are returned as a dataframe (for multiple
        polygons) or a tuple (single polygon) or directly explorted 

        """

        # open the file that contains all
        # polygons to proceed
        if input_data is None:
            self.input = geojson.load(open(os.path.join(self.data_dir, self.source_name)))
            polygons = self.input['features']
            return_tuple = False

        if isinstance(input_data, dict):
            polygons = [input_data]
            return_tuple = True

        if isinstance(input_data, geojson.feature.FeatureCollection):
            self.input = input_data
            polygons = self.input['features']
            return_tuple = False

        # initialize the methods that need to be initialized
        if self.tilt_method == "lut":

                lut = methods.LUT(self.data, cf = self.cf, params = self.params, lut = self.lookup_table) # instantiate the class
                lut.generate_lut() # generate the look up table
                self.lookup_table = lut.lut

        if self.tilt_method == "theil-sen" and self.azimuth_method == "theil-sen":
            theil_sen = methods.TheilSen(cf = self.cf, params = self.params)

        if self.tilt_method == "constant":
            tilt_value = self.tilt_value

        if self.azimuth_method == 'bounding-box':
            bounding_box = methods.BoudingBox()

        # compute the installed capacity 
        # initialize the instance
        linear_regression = methods.LinearRegression(cf = self.cf, params = self.params)
        linear_regression.initialize_coefficients(self.data)


        print("Initialization completed, starts the extraction of the characteristics")

        characteristics = []

        for polygon in tqdm.tqdm(polygons):

            # get the installation's ID

            # get the localization
            coordinates = polygon["geometry"]['coordinates'][0]
            center = np.mean(coordinates, axis = 0) 
            lon, lat = center

            # compute the surface
            projected_surface = methods.compute_surface(polygon)

            # compute the tilt and azmimuth, the method depends 
            # on the parameters inputed by the user
            tilt, azimuth = None, None

            if self.tilt_method == 'theil-sen' and self.azimuth_method == 'theil-sen':

                azimuth, tilt = theil_sen.estimate_tilt_and_azimuth(polygon)

            if self.tilt_method == 'lut':

                # compute the tilt using the look up table
                tilt = lut.return_tilt(polygon, projected_surface)
                self.lut = lut.lut

            if self.tilt_method == "constant":

                tilt = tilt_value

            if self.azimuth_method == "bounding-box":

                azimuth = bounding_box.compute_azimuth(polygon)

            # at the end of the process, tilt and azimuth should not be None
            # if so, it means that the methods were ill specified

            if (tilt == None) or (azimuth == None):
                raise ValueError("""
                Verify the methods inputed for tilt and azimuth estimation.
                Values should be :

                * "theil-sen" for tilt and azimuth

                * "lut" or "constant" for tilt
                * "bounding-box" for azimuth
                """)

            # compute the real surface
            surface = projected_surface / np.cos(tilt * np.pi / 180)

            # compute the installed capacity
            installed_capacity = linear_regression.return_installed_capacity(surface)

            # append all the characteristics
            characteristics.append([
                lat, lon, surface, installed_capacity, tilt, azimuth
            ])

        if save_ext:

            print("Main loop completed. Exporting the file in the {} directory.".format(self.data_dir))
            # once the loop is over, save it as a .csv file
            out = pd.DataFrame(characteristics, columns = ["lat", "lon", "surface", "installed_capacity", 'tilt', "azimuth"])
            out.to_csv(os.path.join(self.data_dir, '{}.csv'.format(self.output_name)), index=False)

            print('Extraction complete.')

        else:
            if return_tuple:

                return lat, lon, surface, installed_capacity, tilt, azimuth
            else:
                out = pd.DataFrame(characteristics, columns = ["lat", "lon", "surface", "installed_capacity", 'tilt', "azimuth"])
                return out


    def return_coordinates(self, polygon):

        # get the localization
        coordinates = polygon["geometry"]['coordinates'][0]
        center = np.mean(coordinates, axis = 0) 
        lon, lat = center

        return lon, lat
    def compute_tilt(self, polygon, method = None, params = None):
        """
        computes the tilt of a polygon passed as input, using either the method 
        or the method specified in the class

        args:
        : self
        : polygon (dict) : the geojson dictionnary that contains the information and the coordinates
        : method  (str) : the name of the desired method. If None, will look 
        : params (dict) : the parameters to compute the tilt

        returns 
        : tilt (float) the tilt in degrees of the installation
        """

        if method is None: # in this case, retrieve the parameters from the initialization

            tilt_method = self.tilt_method

        else:

            tilt_method = method
            # also update the dictionnary with the params supplied as input
            self.params = params


        # computation of the tilt using one of the possible methods
        if tilt_method == "constant":

            tilt = self.tilt_value
            self.tilt_value = params['constant-tilt']

        if tilt_method == 'lut':

            # compute the surface
            projected_surface = methods.compute_surface(polygon)

            # initialize the LUT
            lut = methods.LUT(data = self.data, cf = self.cf, params = self.params, lut = self.lookup_table) # instantiate the class
            lut.generate_lut() # generate the look up table

            # compute the tilt using the look up table
            tilt = lut.return_tilt(polygon, projected_surface)
            self.lut = lut.lut

        if tilt_method == "theil-sen":

            # return only the tilt

            theil_sen = methods.TheilSen(cf = self.cf, params = self.params)
            _, tilt = theil_sen.estimate_tilt_and_azimuth(polygon)


        return tilt

    def compute_surface(self, polygon, params = None, tilt = None):
        """
        returns the *real* surface, i.e. corrected for the tilt
        inclination. The tilt can be passed as input or directly computed.
        in this case, the params dictionnary should provide the parameters necessary 
        to estimate the tilt. 
        """

        # compute the projected surface
        if tilt is None:
            # compute the tilt. The dictionnary supplied by the 
            # user should contain the parameters to compute the tilt as well;
            if params is None:

                tilt = self.compute_tilt(polygon, method = params, params = params)
            else: 
                tilt = self.compute_tilt(polygon, method = params['tilt-method'], params = params)


        projected_surface = methods.compute_surface(polygon)
        surface = projected_surface / np.cos(tilt * np.pi / 180)

        return surface


    def compute_azimuth(self, polygon, method = None, params = None):
        """
        computes the azimuth of the installation using the method specified as input
        if method is None, computes using the parameters specified at initialization
        """

        if method is None: # in this case, retrieve the parameters from the initialization

            azimuth_method = self.azimuth_method

        else:

            azimuth_method = method
            # also update the dictionnary with the params supplied as input
            self.params = params

        if azimuth_method == "theil-sen":

            theil_sen = methods.TheilSen(cf = self.cf, params = self.params)
            azimuth, _ = theil_sen.estimate_tilt_and_azimuth(polygon)

        if azimuth_method == "bounding-box":

            bounding_box = methods.BoudingBox()
            azimuth = bounding_box.compute_azimuth(polygon)

        return azimuth


    def compute_installed_capacity(self, polygon, method = None, params = None, tilt = None, surface = None):
        """
        computes the installed capacity. 
        Requires the tilt and the surface.

        tilt and surface can be supplied by the user
        otherwise, they are computed using the methods specified in the params dictionnary. 
        This dictionnary should contain the parameters for the tilt estimation.
        """

        # manually define the method and the parameters if the user specified a method.
        if method is not None:

            self.regression_method = method
            # also update the dictionnary with the params supplied as input
            self.params = params

        # compute the tilt and surface if necessary
            
        # compute the projected surface
        if tilt is None:
            # compute the tilt. The dictionnary supplied by the 
            # user should contain the parameters to compute the tilt as well;
            if params is None:

                tilt = self.compute_tilt(polygon, method = params, params = params)
            else: 
                tilt = self.compute_tilt(polygon, method = params['tilt-method'], params = params)

        if surface is None:

            projected_surface = methods.compute_surface(polygon)
            surface = projected_surface / np.cos(tilt * np.pi / 180)

        # compute the installed capacity
        linear_regression = methods.LinearRegression(cf = self.cf, params = self.params)
        linear_regression.initialize_coefficients(self.data)
        installed_capacity = linear_regression.return_installed_capacity(surface)

        return installed_capacity






