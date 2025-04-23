# -*- coding: utf-8 -*-

# libraries
import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

import numpy as np
from area import area
import utils

# contains the functions necessary to implement 
# the various methods

def compute_surface(polygon):
    """
    function that computes the surface from a polygon
    the polygon an element of a geojson fiile
    it is therefore a dictionnary

    the area is expressed in sq meters
    """

    # compute the projected area
    return area(polygon['geometry'])

class LinearRegression():

    """
    instantiates the linear regression according to 3 modes 
    to compute the installed capacity as a function of the surface 
    """

    def __init__(self, params = None) -> None:

        self.type = params['regression-type']

        if self.type == 'clustered':
            self.clusters = params['regression-clusters']
        if self.type == "constant":
            self.default = params['default-coefficient']


    def initialize_coefficients(self, data = None):
        """
        computes the coefficients based on the distribution of 
        installed capacities and surfaces found in the data
        dataframe

        returns two dictionnaries, surface_categories and regression_coefficients
        """

        if self.type == "constant": 

            # compute the regression coefficient 
            # the coefficient is inputed by the user
            self.surface_cat = {0 : (0, np.inf)}
            self.regression_coefficients = {0 : self.default}

        if self.type == "linear":

            if data is None:
                raise ValueError('The data dataframe needs to be a dataframe object.')

            # compute the regression coefficient 
            # the data dataframe uses cleaned variable names
            self.surface_cat = {0 : (- np.inf, np.inf)}

            coeffs = (data['kWp'] / data['surface']).values
            coeffs = np.nan_to_num(coeffs, posinf=np.nan, neginf=np.nan)

            self.regression_coefficients = {0 : np.nanmean(coeffs)}

        if self.type == "clustered":

            if data is None:
                raise ValueError('The data dataframe needs to be a dataframe object.')

            # define the clusters and the corresponding regression coefficients
            n_quantiles = self.clusters - 1

            # check that there is at least one quantile
            if not n_quantiles >= 1:
                raise ValueError("The number of quantiles should be greater or equal to 2")
            # compute the surface categories and the regression coefficients

            surface_categories, regression_coefficients = utils.create_categories_and_regression_coefficients(n_quantiles, data)

            # add two new attributes
            self.surface_cat = surface_categories
            self.regression_coefficients = regression_coefficients

        return None
  
    def return_installed_capacity(self, surface):
        """
        computes the installed capacity 
        returns a scalar
        """

        surface_id = utils.return_surface_id(surface, self.surface_cat) #compute the surface id

        # return the installed capacity as a linear regression of the corresponding coefficient
        # on the surface of the installation

        return surface * self.regression_coefficients[surface_id] 

class LUT():
    """
    class that computes the look-up table using the 
    data file and the configuration file
    it then uses it to return the estimated tilt of the installation.
    """

    def __init__(self,  lut) -> None:
        """
        should input a looktup table with the same format as the one provided
        in the data/lut.json file.

        the constructor for custom look up tables is not implemented yet.
        """


        self.surface_categories = lut['surface_categories']
        self.longitude_categories = lut['longitude_categories']
        self.latitude_categories = lut['latitude_categories']

        # convert the lookup table as an array
        self.lookup_table = {cat : np.array(lut['lut'][cat]) for cat in self.surface_categories.keys()}



#        if params is not None:
#            self.steps = params['lut-steps']
#            self.clusters = params['regression-clusters']

#    def initialize(self):
#
#        # define the geographical boundaries of the look up table
#
#        if self.lookup is None:
#
#            if self.data is None:
#                raise ValueError('The data dataframe needs to be a dataframe object.')
#
#            south, north = np.floor(np.min(self.data["lat"])), np.ceil(np.max(self.data['lat']))
#            west, east = np.floor(np.min(self.data["lon"])), np.ceil(np.max(self.data['lon']))

            # compute the latitude and longitude categories dictionnaries
            # these dictionnary associate a key (i.e. an int, corresponding to a coordinate)
            # to a localization 
#            latitudes_categories, longitude_categories = utils.create_geo_categories(west, east, south, north, self.steps)

            # define the number of clusters to compute the surface categories
            # surface categories are based on the distribution of projected surfaces
            # as we do not know the tilt yet.

 #           n_quantiles = self.clusters - 1
 #           surface_categories = utils.create_surface_categories(n_quantiles, self.data)

 #           self.surface_categories = surface_categories
 #           self.longitude_categories = longitude_categories
 #           self.latitude_categories = latitudes_categories

  #      else: 
  #          pass
        
#        return None


    #def generate_lut(self):
    #    """
    #    creates the look up table
    #    """

    #    # initialize the "marginal" dictionnaries (surface, latitude and longitude categories)
    #    self.initialize()

    #    if self.lookup is None:

            # associate each installation of the dataframe to a category
#        utils.assign_categories(self.data, self.surface_categories, self.latitude_categories, self.longitude_categories)

            # create the lut
#        lut = utils.create_LUT(self.data, self.surface_categories, self.latitude_categories, self.longitude_categories)

            # store the look up table
#            self.lut = lut

#        else:
#            print('Importing a lookup table. The file should contain the LUT and the categories.')
            
            #self.surface_categories = self.lookup['surface_categories']
            #self.longitude_categories = self.lookup['longitude_categories']
            #self.latitude_categories = self.lookup['latitude_categories']

            # convert the lookup table as an array
            #self.lut = {cat : np.array(self.lookup['lut'][cat]) for cat in self.surface_categories.keys()}


    def return_tilt(self, polygon, surface):
        """
        returns the tilt of the polygon 
        surface corresponds to the projected surface
        """

        # compute the center of the polygon
        coordinates = np.array(polygon['geometry']["coordinates"][0])
        center = np.mean(coordinates, axis = 0)

        lat_id, lon_id = utils.return_latitude_and_longitude_ids(center, self.latitude_categories, self.longitude_categories)
        surface_id = utils.return_surface_id(surface, self.surface_categories)

        if surface_id is None: # return the largest value
            surface_id = list(self.surface_categories.keys())[-1]

        # first key is the surface id
        # then we look for the corresponding latitude and longitude 
        # in the array stored under the surface_id key

        if not isinstance(lat_id, int):
            lat_id = int(lat_id)

        if not isinstance(lon_id, int):
            lon_id = int(lon_id)

        return self.lookup_table[surface_id][lat_id, lon_id]

class TheilSen():
    """
    implements the Theil Sen regressor on a DEM to compute the tilt and azimuth
    """

    def __init__(self, params) -> None:
        """
        conversion: handles a potential 
        """

        # optional parameters
        self.seed = int(params['seed']) if 'seed' in params else 42
        self.N = int(params['N']) if 'N' in params else 10 
        self.M = int(params['M']) if 'M' in params else 1000
        self.offset = int(params['offset']) if 'offset' in params else 25

        # conversion between the coordinates systems
        self.conversion = params['conversion'] if "conversion" in params else "epsg:4326,epsg:2154"

        print('conversion file')
        print(self.conversion)


        if 'raster-folder' not in params:
            raise ValueError("must input a path to the DEM data")
        self.folder = params['raster-folder']
    
    def estimate_tilt_and_azimuth(self, polygon):
        
        # retrieve the DSM and the pixel coords of the polygon
        dsm, pixel_coords = utils.extract_dem_from_raster(polygon, self.offset, self.folder, self.conversion)

        # convert the polygon as a mask, in the 
        # coordinate space of the dsm
        mask = utils.compute_mask_from_polygon(dsm.shape, pixel_coords)

        # compute the tilt and azimuth
        return utils.theil_sen_estimator(mask, dsm, M=self.M, N=self.N, random_state=self.seed)

class BoudingBox():
    """
    Implements the bounding box method to estimate the azmimuth of the polygon
    """

    def __init__(self) -> None:
        pass

    def compute_azimuth(self, polygon):
        return utils.azimuth_bounding_box(polygon)