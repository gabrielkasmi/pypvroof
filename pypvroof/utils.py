# -*- coding: utf-8 -*-
# set of useful functions

# libraries
import numpy as np
import pandas as pd
from scipy import interpolate
import os
from sklearn.linear_model import TheilSenRegressor
from shapely.geometry import Polygon, Point
from osgeo import gdal
import itertools
from scipy.spatial import distance
from pyproj import Transformer
import matplotlib.pyplot as plt
import cv2 as cv

def compute_mask_from_polygon(shape, pixel_coords):
    """
    returns a binary mask with the same shape as 
    the dsm
    """

    # initialize the mask
    mask = np.zeros(shape)

    # set up for the intersection between the 
    # coordinates and the polygon
    X, Y = shape

    polygon = Polygon(pixel_coords)

    for x, y in itertools.product(range(X), range(Y)):

        pixel = Point(x,y)
        if polygon.intersects(pixel):

            mask[y,x] = 1

    return mask

def coord_to_pixel(point, geotransform):
    """
    converts a pixel as a point
    expressed in the reference frame of the raster
    from which the geotransform is extracted.
    
    returns xNN, yNN the (x,y) coordinates 
    in pixel of the point.
    """ 
    
    x, y = point
    
    ulx, xres, _, uly, _, yres  = geotransform 
    
    xNN = (x - ulx) / xres
    yNN = (y - uly) / yres

    return int(xNN), int(yNN)

def extract_thumbnail(array, offset, raster):
    """
    extracts a thumbnail containing the polygon from the 
    raster
    """

    # get the geotransform
    ulx, xres, _, uly, _, yres = raster.GetGeoTransform()

    # convert the coordinates as a shapely polygon
    #coords = polygon['geometry']['coordinates'][0]
    #coordinates = Polygon(polygon['geometry']['coordinates'][0])

    # get the boundaries (in geo coordinates)
    minx, miny, maxx, maxy = array.bounds

    # express the boundaries in terms of pixels
    minX = (minx - ulx) / xres
    maxX = (maxx - ulx) / xres

    minY = (miny - uly) / yres
    maxY = (maxy - uly) / yres

    # define the dimensions of the thumbnail, based on the
    # boundaries of the polygon and the offset parameter

    left = minX - offset
    right = maxX + offset
    bottom = minY + offset
    top = maxY - offset


    width = np.abs(right - left)
    height = np.abs(bottom - top)

    # extract the thumbnail based on the upper left corner
    # and the width and height of the thumbnail

    band = raster.GetRasterBand(1) # there is only one band to extract
    dsm = band.ReadAsArray(int(left), int(top), int(width), int(height))

    # express the coordinates of the array polygon relative to the 
    # dsm thumbnail
    xx, yy = array.exterior.coords.xy
    coords = [[x,y] for x,y in zip([x for x in xx], [y for y in yy])]
    pixel_coord = np.array([coord_to_pixel(c, raster.GetGeoTransform()) for c in coords])

    pixel_coord[:,0] -= int(left)
    pixel_coord[:,1] -= int(top)

    return dsm, pixel_coord

def transform_to_polygon(raster):
    """
    transforms a gdal raster as a polygon
    """

    # get the shape (in pixels) and the coordinates
    # of the raster
    width, height = raster.RasterXSize, raster.RasterYSize
    left, xres, _, top, _, yres = raster.GetGeoTransform()

    right = left + width / xres
    bottom = top + height / yres

    coordinates = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
        [left, top]
    ])

    polygon = Polygon(coordinates)

    return polygon

def convert(coordinates, conversion):
    """
    converts the coordinates

    conversion is a list with the in_proj, out_proj
    specified in epsg format

    coordinates is a np.ndarray
    """
    
    in_epsg, out_epsg = conversion.split(',')

    transformer = Transformer.from_crs(in_epsg, out_epsg)

    if isinstance(coordinates, np.ndarray):
        coordinates = np.atleast_2d(coordinates)
        converted = np.array(transformer.transform(coordinates[:, 1], coordinates[:, 0]))
        return converted.transpose()
    else:
        x, y = transformer.transform(coordinates[1], coordinates[0])
        return (x, y)

def find_raster(polygon, folder, conversion = None):
    """
    finds and returns the raster that contains the polygon

    input : 
    - polygon : a dictionnary from a geojson file
    - folder : a str corresponding to the directory where the DEM rasters are located

    Remark: the rasters should be geotiff images that can be opened by gdal.

    returns :
    - raster : a gdal raster 
    - array : the array as a polygon, under the correct coordinates system
    """

    # should be a polygon, if it does not work then
    # the data format is ill specified
    #try:
    if conversion is not None:

        coordinates = np.array(polygon['geometry']['coordinates'][0])
        converted = convert(coordinates, conversion)
        array = Polygon(converted)

        print('converti')

    else:

        array = Polygon(polygon['geometry']['coordinates'][0])

    #except:

        #print('Characteristics extraction can only be done on polygons. Check the data format.')
    
    # get the list of rasters
    rasters = os.listdir(folder)
    rasters = [r for r in rasters if r[-3:] == 'tif']

    print(rasters)

    # initialize the raster by setting it as a Non
    # if after the loop the raster remains a None, it means
    # that no matching raster has been found
    # in this case, an error will be raised.
    raster = None

    for r in rasters:

        # open the raster
        candidate = gdal.Open(os.path.join(folder, r))

        polygon = transform_to_polygon(candidate)

        if polygon.contains(array):
            raster = candidate
            break

    return raster, array
 
def extract_dem_from_raster(polygon, offset, folder, conversion):
        """
        extracts a thumbnail (returned as np.ndarray) from 
        the dem raster centered around the polygon

        the offset corresponds to the width (in pixels) between the boundaries of 
        the polygon and the boundaries of the thumbnail.

        returns the transform as well, which are needed to generate the
        mask of the polygon (which is also a np.ndarray)

        args:

        - polygon: a geojson dictionnary
        - offset : an int, corresponding to the desired offset between 
                   the edges of the polygon and the borders of the thumbnail
        - folder : the folder where the raster is located. The function automatically
                   looks for the raster that contains the polygon in the folder.

        WARINING
        To work, the coordinates sytem of the polygon and of the raster 
        should be identical
        """

        # find the raster that contains the polygon
        raster, array = find_raster(polygon, folder, conversion = conversion)

        if raster is None:
                raise ValueError('3D DEM corresponding to the polygon not found. Please input the raster or choose another method.')

        # extract the thumbnail with the desired offset
        # returns as a by product the pixel coords of the polygon
        # relative to the dsm
        dsm, pixel_coord = extract_thumbnail(array, offset, raster)

        return dsm, pixel_coord

def create_geo_categories(west, east, south, north, steps):
    """
    creates the latitude and longitude dictionnaries given the 
    bounding box and the specified number of steps
    
    returns:
    latitudes_categories : a dictionnary with the categories correpsonding
                           to slices for the latitudes
    longitude_categories: same but for longitudes
    """

    longitudes = np.linspace(west,east, steps)
    latitudes = np.linspace(south,north, steps)

    latitudes_categories, longitude_categories = {}, {}

    for j,i in enumerate(range(len(latitudes) - 1)):

        lb, ub = latitudes[i], latitudes[i+1]
        latitudes_categories[j] = [lb, ub]
        
    for j,i in enumerate(range(len(longitudes) - 1)):
        lb, ub = longitudes[i], longitudes[i+1]
        longitude_categories[j] = [lb, ub]

    return latitudes_categories, longitude_categories

def create_surface_categories(n_quantiles, data):
    """
    creates the surface categories based on the number of quantiles
    these surfaces are based on the projected_surface
    """

    data['projected_surface'] = data.apply(lambda x : x['surface'] * np.cos(x['tilt'] * (np.pi / 180)), axis = 1)

    X_train_ord = data.sort_values('projected_surface', ascending=True)

    # Define the upper and lower bounds of n_quantiles categories of surfaces
    surfaces = [0]
    for i in range(1, n_quantiles):
        quantile = i * (1 / n_quantiles)
        surfaces.append(np.quantile(X_train_ord["projected_surface"].values, quantile))
    surfaces.append(np.ceil(np.max(X_train_ord['projected_surface']))) # the upper bound is the ceil of the max of the surface

    # Create a dictionary that maps category index to upper and lower bounds of surfaces
    surfaces_categories = {i: (surfaces[i], surfaces[i + 1]) for i in range(len(surfaces) - 1)}

    return surfaces_categories


def create_categories_and_regression_coefficients(n_quantiles, df_train):
    """
    Create surface categories and regression coefficients based on the number of quantiles provided.
    :param n_quantiles: Number of quantiles to use to create the surface categories.
    :param df_train: Dataframe to extract surface column, and create new columns.
    :return: tuple of surfaces_categories, regression_coefficients
    """
    # Sort the dataframe by ascending values of the surface column
    X_train_ord = df_train.sort_values('surface', ascending=True)

    # Reset the index of the dataframe and drop the old index
    X_train_ord = X_train_ord.reset_index(drop=True)

    # Add a new column named 'reg_coef' which is the ratio of the 'kWp' and 'surface' columns
    X_train_ord['reg_coef'] = X_train_ord['kWp'] / X_train_ord['surface']

    # Define the upper and lower bounds of n_quantiles categories of surfaces
    surfaces = [0]
    for i in range(1, n_quantiles):
        quantile = i * (1 / n_quantiles)
        surfaces.append(np.quantile(df_train["surface"].values, quantile))

    surfaces.append(np.ceil(np.max(df_train['surface']))) # the upper bound is the ceil of the max of the surface

    # Create a dictionary that maps category index to upper and lower bounds of surfaces
    surfaces_categories = {i: (surfaces[i], surfaces[i + 1]) for i in range(len(surfaces) - 1)}

    # Get the mean of the 'reg_coef' column for each category of surfaces
    rows_per_category = len(X_train_ord) // n_quantiles
    regression_coefficients = {
        i: X_train_ord.iloc[i * rows_per_category:(i + 1) * rows_per_category]['reg_coef'].mean()
        for i in range(n_quantiles)
    }

    return surfaces_categories, regression_coefficients

def assign_categories(df, surface_categories, latitude_categories, longitude_categories):
    """
    Assign each installation in the dataframe to a projected surface category.
    :param df: Dataframe containing the installations
    :param surface_categories: Dictionary of surface categories with upper and lower bounds
    """

    # surface
    for i in range(df.shape[0]):

        proj = df.loc[i,"projected_surface"]

        for category in surface_categories.keys():
            bounds = surface_categories[category]

            status = (proj > bounds[0]) & (proj <= bounds[1])

            if status :
                df.loc[i, "surface_category"] = int(category)


    # Get the array of upper bounds of the categories
    upper_lat = [bounds[1] for bounds in latitude_categories.values()]

    # Assign a category to each installation based on the projected_surface column
    df.sort_values("lat", ascending = True)
    df["latitude_category"] = pd.cut(df["lat"], bins=upper_lat, labels=False, include_lowest=True)
    latitude_keys = list(latitude_categories.keys())
    df['latitude_category'].fillna(latitude_keys[0], inplace = True)

    # Get the array of upper bounds of the categories
    upper_lon = [bounds[1] for bounds in longitude_categories.values()]

    # Assign a category to each installation based on the projected_surface column
    df.sort_values("lon", ascending = True)
    df["longitude_category"] = pd.cut(df["lon"], bins=upper_lon, labels=False, include_lowest=True)
    longitude_keys = list(longitude_categories.keys())
    df['longitude_category'].fillna(longitude_keys[0], inplace = True)

    # drop nans
    # df.dropna(subset=["longitude_category", 'latitude_category'], inplace = True)

    return None

def return_surface_id(area, surface_categories):
    """
    returns the surface id of the installation given
    its (projected) area (in sq meters)
    
    returns an int, O to k corresponding
    to the suface cluster
    """

    surface_id = None

    for surface_key in surface_categories.keys():
        lb, ub = surface_categories[surface_key]
        if area <= ub and area > lb:
            surface_id = surface_key
            break


    # if exceeds the last quantile, considers the 
    # largest possible category
    if surface_id is None:
        surface_id = list(surface_categories.keys())[-1]
    
    return surface_id

def return_latitude_and_longitude_ids(center, latitude_categories, longitude_categories):
    """
    returns the latitude and longitude groups (both in [0, 48])
    given a location. 
    """

    lon, lat = center
        
    for latitude_key in latitude_categories.keys():
        lb, ub = latitude_categories[latitude_key]
        if lat <= ub and lat > lb:  
            lat_id = latitude_key
            break
            
    
    for longitude_key in longitude_categories.keys():
        lb, ub = longitude_categories[longitude_key]
        if lon <= ub and lon > lb:
            lon_id = longitude_key
            break
            
    return lat_id, lon_id



def create_LUT(df, surfaces_categories, latitudes_categories, longitude_categories):
    """
    Generates a look-up table (LUT) of interpolated values for different surface categories using the given dataframe.
    
    Parameters:
    - df: Dataframe containing data to be used for generating the LUT
    - surfaces_categories: Dictionary containing the different surface categories
    - latitudes_categories: Dictionary containing the different latitude categories
    - longitude_categories: Dictionary containing the different longitude categories
    
    Returns:
    - LUT: A dictionary with the generated interpolated values for each surface category
    """

    print('Initializing the look-up-table... This may take some time.')

    # categories for which there are observations : 
    entries = np.unique(df[['longitude_category','latitude_category', 'surface_category']].values, axis = 0)

    df['cluster'] = 0

    for k in range(entries.shape[0]):
        corresponding_indices = df[(df['longitude_category'] == entries[k][0]) & (df['latitude_category'] == entries[k][1]) & (df['surface_category'] == entries[k][2])].index
        for index in corresponding_indices: # quick and dirty
            df.loc[index, 'cluster'] = k

    means = df.groupby(['cluster'])["tilt"].mean()
    for k in range(entries.shape[0]):
        corresponding_indices = df[df['cluster'] == k].index
        for index in corresponding_indices:
            df.loc[index, 'mean_tilt'] = means[k]

    
    LUT = {}
    for i in surfaces_categories.keys():
                
        grid = np.zeros((len(latitudes_categories.keys()),len(longitude_categories.keys())))
        
        subset = df[df['surface_category'] == i]
        contains = np.unique(subset[['longitude_category','latitude_category']].values, axis = 0)


        for contain in contains:
            corresponding_indices = subset[(subset['longitude_category'] == contain[0]) & (subset['latitude_category'] == contain[1])].index
            grid[int(contain[0]),int(contain[1])] += min(60, subset.loc[corresponding_indices[0], 'tilt'])
            
        # interpolate
        x, y = np.linspace(0,grid.shape[0], grid.shape[0]), np.linspace(0,grid.shape[1], grid.shape[1]) 
        X, Y = np.meshgrid(x,y)

        points = [np.array([x, y]) for x, y in zip(np.where(grid > 0)[0],np.where(grid > 0)[1])]
        values = grid[grid>0] 
        out = interpolate.griddata(points, values, (X,Y))
                     
        LUT[i] = out
    
    print("LUT initialization complete.")

    return LUT

def check_phi(phi, offset = 45):
    """
    rescale phi to make it point northwards
    and rotating eastwards:

    before : 
    0 : south 
    -90 : west
    90 : east
    (-)180 : north

    after:
    0 : north
    -90 : west
    90 : east
    (-)180 : south

    also applies an offset that reprojects the angle 
    to the lower half of the circle 
    """

    #print(phi)
    # rescale
    if phi > 0:
        phi  = -phi + 180

    else:
        phi = np.abs(phi) - 180

    return phi


def reload(dem):
    """
    save the dem and reloads it with open cv.imread
    Erases the file on the fly
    """

    plt.imsave("temp.png", dem)
    dsm = cv.imread("temp.png", 0)
    os.remove("temp.png") 

    return dsm

def theil_sen_estimator(mask, dem, M=1000, N=10, random_state=42):

    """
    mask : a np.array of a binary raster
    dem : a np.array of a rasterized DEM. 

    mask and dem should be geographically identical.
    """

    # workaround to make the algorithm work
    dsm = reload(dem)
    masked_mns = dsm * mask

    # Get position of mask's pixels
    X = np.argwhere(masked_mns>0)  # coord y,x in numpy arrays
    # Get associated altitude
    y = masked_mns[np.where(masked_mns>0)]
    # Fit TheilSen regressor to data
    reg = TheilSenRegressor(random_state=random_state, n_subsamples=N, max_subpopulation=M, verbose=False).fit(X, y)
    # Get regressor coefficients
    a, b = reg.coef_
    c = reg.intercept_
    # Derive orientation phi
    phi = np.arctan2(-b, a) * (180/np.pi)
    # Derive tilt theta
    d = np.sqrt(a**2 + b**2) 
    h = 0.5*(reg.predict([reg.coef_])[0]-c)
    theta = np.arctan(h/d) * (180/np.pi)
    # Return results

    return check_phi(phi + 180), theta.item() # shift the orientation to the south


def azimuth_bounding_box(polygon):
    """
    computes the probable azimuth of the installation using 
    the polygon and its minimum bounding rectangle

    returns the angle relative to south :

    0 = south
    -90 = west
    90 = east
    180 = north

    applies a heuristic that "reprojects" panels facing north (i.e. if the azimuth 
    is larger than 135 degrees in absolute value) towards south
    """

    poly = Polygon(polygon['geometry']["coordinates"][0])
    x,y = poly.minimum_rotated_rectangle.exterior.coords.xy # split the tuple into the x and y coordinates. returned starting from the UR corner
                                                            # and rotating counter clockwise
    # get the points from the bounding rectangle
    ulx, uly = x[1], y[1]
    llx, lly = x[2], y[2]
    lrx, lry = x[3], y[3]

    # compute lenghts of the two sides of the rectangle
    long = distance.euclidean((lrx, lry), (llx, lly))
    short = distance.euclidean((ulx, uly), (llx, lly))

    # compute the two associated angles
    angle_short = 90 - (np.arctan2(
        (lry- lly), (lrx - llx)
    ) * 180 / np.pi)

    angle_long = - 90 + (np.arctan2(
        (uly - lly), (ulx - llx)
    ) * 180 / np.pi)

    # aggregate in a dictionnary
    angles = {
        long : angle_long,
        short : angle_short
    }

    # return the angle attached to the longest side of the rectangle
    raw_phi = angles[np.max([long, short])]

    return check_phi(raw_phi).item()