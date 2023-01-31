# -*- coding: utf-8 -*-

# set of functions necessary to define the constructor

def format_dataset(dataset, cf = None, params = None):
    """
    takes a .csv file and the names of the columns
    """

    if cf is not None:

        # get the names of the columns in the input 
        lat_name = cf.get("latitude_var")
        lon_name = cf.get('longitude_var')
        installed_capacity_name = cf.get('ic_var')
        surface_name = cf.get('surface_var')
        tilt_name = cf.get('tilt_var')

    if params is not None:

        # get the names of the columns in the input 
        lat_name = params["latitude_var"]
        lon_name = params['longitude_var']
        installed_capacity_name = params['ic_var']
        surface_name = params['surface_var']
        tilt_name = params['tilt_var']

    # edit the dataset accordingly

    target_columns = [
        lat_name,
        lon_name,
        installed_capacity_name,
        surface_name,
        tilt_name
    ]

    data = dataset[target_columns]

    data.rename(columns = {
        lat_name : "lat",
        lon_name : "lon",
        installed_capacity_name : 'kWp',
        surface_name : 'surface',
        tilt_name : "tilt"
    }, inplace = True)
    
    return data

