# -*- coding: utf-8 -*-

# libraries
import numpy as np
from area import area
from shapely.geometry import shape

from . import utils

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
        c = shape(polygon['geometry']).centroid
        center = (c.x, c.y)

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

        grid = self.lookup_table[surface_id]
        tilt = grid[lat_id, lon_id]
        if not np.isnan(tilt):
            return tilt

        # ~45% of the cells are NaN (outside the convex hull of the points the
        # LUT was interpolated from). A NaN tilt propagates to the surface and
        # the installed capacity, so fall back to the nearest non-NaN cell of
        # the same surface category.
        valid = np.argwhere(~np.isnan(grid))
        if valid.size == 0:
            return tilt
        dists = np.sum((valid - np.array([lat_id, lon_id])) ** 2, axis=1)
        nearest = valid[np.argmin(dists)]
        return grid[nearest[0], nearest[1]]

class TheilSen():
    """
    Tilt and azimuth of a single polygon from a DSM, with the Theil-Sen
    estimator (see pypvroof.theilsen). For many polygons, use
    pypvroof.characterize_roofs, which processes them concurrently.

    params:
      - "dsm-service": a DSMService, a service name ("fr-lidar", "nl-ahn"...), or a folder of GeoTIFFs
        ("raster-folder" is accepted for backward compatibility)
      - "theil-sen-params": dict of estimator parameters (TheilSenParams)
      - "crs": CRS of the input polygons (default EPSG:4326)
    """

    def __init__(self, params) -> None:
        from . import dsm
        spec = params.get('dsm-service') or params.get('raster-folder')
        if spec is None:
            raise ValueError("the theil-sen method needs a 'dsm-service' (or 'raster-folder')")
        self.service = dsm.get_service(spec)
        self.estimator_params = params.get('theil-sen-params')
        self.crs = params.get('crs', 'EPSG:4326')

    def characterize(self, polygon):
        """Full result (dict) for a GeoJSON feature / geometry / shapely polygon."""
        from .bulk import characterize_roofs
        out = characterize_roofs([polygon], self.service, crs=self.crs,
                                 params=self.estimator_params, workers=1)
        return out.iloc[0].to_dict()

    def estimate_tilt_and_azimuth(self, polygon):
        """Returns (azimuth, tilt); NaN when the estimation fails."""
        r = self.characterize(polygon)
        return r['azimuth'], r['tilt']

class BoudingBox():
    """
    Implements the bounding box method to estimate the azmimuth of the polygon
    """

    def __init__(self) -> None:
        pass

    def compute_azimuth(self, polygon):
        return utils.azimuth_bounding_box(polygon)