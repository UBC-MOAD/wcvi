import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import os,sys
import numpy as np
from salishsea_tools import viz_tools, geo_tools,nc_tools
from scipy.interpolate import griddata, interp1d
import matplotlib.cm as cm


def calculate_initial_compass_bearing(pointA, pointB):
    ''' The bearing between two points is calculated.
       
    θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    φ1,λ1 is the start point
    φ2,λ2 the end point 
    
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    '''
    
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = np.radians(pointA[0])
    lat2 = np.radians(pointB[0])
    
    diffLong = np.radians(pointB[1] - pointA[1])

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
            * np.cos(lat2) * np.cos(diffLong))

    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing



def rotate_vectors_to_align_with_grid(path_BC,fname_mask,fname_coord_BC,fname_coord_WCVI, fname_T_BC, fname_U_BC, fname_V_BC): 
    
    gridT = xr.open_dataset(path+ fname_T_BC);
    gridU = xr.open_dataset(path+ fname_U_BC);
    gridV = xr.open_dataset(path+ fname_V_BC);
    mask = xr.open_dataset(path+ fname_mask);
    grid_JP = xr.open_dataset(path+ fname_coord_BC);
    
    glamt = grid_JP.variables['glamt'][55:400,446:701]; gphit = grid_JP.variables['gphit'][55:400,446:701];
    glamu = grid_JP.variables['glamu'][55:400,446:701]; gphiu = grid_JP.variables['gphiu'][55:400,446:701];
    glamv = grid_JP.variables['glamv'][55:400,446:701]; gphiv = grid_JP.variables['gphiv'][55:400,446:701];
    
    
    nav_lon_T = np.array(gridT['nav_lon'][55:400,446:701]);
    nav_lat_T = np.array(gridT['nav_lat'][55:400,446:701]);

    nav_lon_U = np.array(gridU['nav_lon'][55:400,446:701]);
    nav_lat_U = np.array(gridU['nav_lat'][55:400,446:701]);
    
    nav_lon_V = np.array(gridV['nav_lon'][55:400,446:701]);
    nav_lat_V = np.array(gridV['nav_lat'][55:400,446:701]); 
    
    U_vel_BC = np.array(gridU['vozocrtx'][:,:,55:400,446:701]);
    V_vel_BC = np.array(gridV['vomecrty'][:,:,55:400,446:701]);
    
    t_mask = mask['tmask'][0,:,55:400,446:701];

    for i,j in enumerate (U_vel_BC[:,...]):
        U_vel_BC[i,...] = ma.masked_array(U_vel_BC[i,...], mask = 1- t_mask[...]);
        V_vel_BC[i,...] = ma.masked_array(V_vel_BC[i, ...], mask = 1- t_mask[...]);
        
        
    # "Unstagger" the velocity values by interpolating them to the T-grid points
    u_vel_BC_tzyx, v_vel_BC_tzyx = viz_tools.unstagger(U_vel_BC, V_vel_BC);
    
    mag_vel_BC = np.sqrt(np.multiply(u_vel_BC_tzyx,u_vel_BC_tzyx), np.multiply(v_vel_BC_tzyx,v_vel_BC_tzyx));
    ang_vel_BC = np.degrees(np.arctan2(v_vel_BC_tzyx, u_vel_BC_tzyx));
    
    angle_unrotated = np.zeros((ang_vel_BC.shape[0],ang_vel_BC.shape[1],ang_vel_BC.shape[2], ang_vel_BC.shape[3]));


    for p,q in enumerate(ang_vel_BC[:,...]):
        for i,j in enumerate(ang_vel_BC[p,:,...]):
            angle_unrotated[p,i,...] = ang_vel_BC[p,i,...] + angle_needed[1:,:];
    
    u_unrotated = mag_vel_BC*np.cos(np.radians(angle_unrotated));
    v_unrotated = mag_vel_BC*np.sin(np.radians(angle_unrotated));
    
    
    with nc.Dataset(path+fname_coord_WCVI, 'r') as grid_WCVI:
        glamt_wcvi = grid_WCVI.variables['glamt'][0,:]; gphit_wcvi = grid_WCVI.variables['gphit'][0,:];
        glamu_wcvi = grid_WCVI.variables['glamu'][0,:]; gphiu_wcvi = grid_WCVI.variables['gphiu'][0,:];
        glamv_wcvi = grid_WCVI.variables['glamv'][0,:]; gphiv_wcvi = grid_WCVI.variables['gphiv'][0,:];
    
    
    # First point
    lonA_wcvi = glamu_wcvi[:,0:-1]
    latA_wcvi = gphiu_wcvi[:,0:-1]
    # Second point
    lonB_wcvi = glamu_wcvi[:,1:]
    latB_wcvi = gphiu_wcvi[:,1:]


    bearing_wcvi = cibc((latA_wcvi,lonA_wcvi),(latB_wcvi,lonB_wcvi));
    angle_needed_wcvi = 90 - bearing_wcvi;
    
    
    

    

    
    
    
    
    
    
    