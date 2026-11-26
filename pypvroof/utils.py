# -*- coding: utf-8 -*-
# set of useful functions

# libraries
import numpy as np
import pandas as pd
from scipy import interpolate
from shapely.geometry import shape


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

    def _find(value, categories):
        # category containing the value; outside the grid, the closest
        # category (instead of an UnboundLocalError)
        keys = list(categories.keys())
        for key in keys:
            lb, ub = categories[key]
            if lb < value <= ub:
                return key
        if value <= categories[keys[0]][0]:
            return keys[0]
        return keys[-1]

    return _find(lat, latitude_categories), _find(lon, longitude_categories)



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

def _largest_polygon(geometry):
    geom = shape(geometry) if isinstance(geometry, dict) else geometry
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def azimuth_bounding_box(polygon):
    """
    Probable azimuth of the installation from the minimum rotated rectangle
    of the polygon (coordinates in lon/lat, EPSG:4326).

    The installation is assumed to face one of the two directions orthogonal
    to the long side of the rectangle; the one facing the equator (closest to
    south) is returned, as panels facing north are rare. The 180 degrees
    ambiguity cannot be resolved from the footprint alone (use the Theil-Sen
    method on a DSM for that).

    Returns degrees, 0 = north, clockwise (90 = east, 180 = south, 270 = west),
    in [90, 270].
    """
    if isinstance(polygon, dict):
        polygon = polygon.get("geometry", polygon)
    geom = _largest_polygon(polygon)
    lon0, lat0 = geom.centroid.x, geom.centroid.y
    # local metric frame (x east, y north)
    k = np.cos(np.radians(lat0))
    x, y = geom.minimum_rotated_rectangle.exterior.coords.xy
    x = (np.asarray(x) - lon0) * k
    y = np.asarray(y) - lat0
    e1 = np.array([x[1] - x[0], y[1] - y[0]])
    e2 = np.array([x[2] - x[1], y[2] - y[1]])
    dx, dy = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    if dx == 0 and dy == 0:
        return 180.0
    # normals to the long side, as azimuths (0 = north, clockwise)
    candidates = [np.degrees(np.arctan2(dy, -dx)) % 360, np.degrees(np.arctan2(-dy, dx)) % 360]
    return float(min(candidates, key=lambda a: abs(a - 180.0)))
