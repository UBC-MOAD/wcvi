import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import os,cmocean
import scipy.io as sio
from scipy import interpolate, signal
from pyproj import Proj,transform
from bathy_common import *
from matplotlib import path
from salishsea_tools import viz_tools
from netCDF4 import Dataset
import xarray as xr
from salishsea_tools import nc_tools
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import cmocean.cm as cm
import matplotlib.gridspec as gridspec




cascadiafile = '/home/ssahu/saurav/NEMO_run/bathy_casacadia/cascadia.bil'
def getcascadia(filename):
    # Adapted from: https://pymorton.wordpress.com/2016/02/26/plotting-prism-bil-arrays-without-using-gdal/
    def read_prism_hdr(hdr_path):
        """Read an ESRI BIL HDR file"""
        with open(hdr_path, 'r') as input_f:
            header_list = input_f.readlines()
        return dict(item.strip().split() for item in header_list)
    def read_prism_bil(bil_path):
        """Read an array from ESRI BIL raster file"""
        hdr_dict = read_prism_hdr(bil_path.replace('.bil', '.hdr'))
        data = np.fromfile(bil_path, dtype=np.int16).byteswap()
        data = data.reshape(int(hdr_dict['NROWS']), int(hdr_dict['NCOLS']))
        return data
    z = np.flipud(read_prism_bil(filename))    # load data
    mask = (z == 0) | (z >= 10000)             # mask for nonexistant points and land points
    z -= 10000                                 # remove offset
    z *= -1                                    # make depths positive
    z[mask] = 0                                # set masked values to zero
    zm = ma.masked_array(z, mask=mask)
    # Construct Cascadia coordinates
    xmin, xmax, dx = -738044.062, 749705.938, 250
    ymin, ymax, dy = 101590.289, 1710340.289, 250
    x=xmin + dx*np.arange(0, z.shape[1]) + dx/2
    y=ymin + dy*np.arange(0, z.shape[0]) + dy/2
    p = Proj(r'+proj=lcc +lat_1=41.5 +lat_2=50.5 +lat_0=38 +lon_0=-124.5 +x_0=0 +y_0=0 +ellps=clrk66 +no_defs')
    lat_min, lat_max=47, 50
    lon_min, lon_max=123, 130
    lat=np.linspace(lat_min,lat_max,660)
    lon=np.linspace(lon_min,lon_max,1144)
    grid=np.meshgrid(lat,lon)
    return x,y,z,p





x,y,z,p = getcascadia(cascadiafile)



coord = nc.Dataset('coordinates_westcoast_seagrid_high_resolution_truncated.nc')

T_grid_lon = coord.variables['glamt'][0,:]
T_grid_lat = coord.variables['gphit'][0,:]

e1t = coord.variables['e1t'][:,:]
e2t = coord.variables['e2t'][:,:]



Xt, Yt = p(T_grid_lon, T_grid_lat)


X,Y = np.meshgrid(x, y, sparse=False, indexing='xy')

X,Y,z = X.flatten(), Y.flatten(), z.flatten()


points = (X,Y)
xi = (Xt.flatten(), Yt.flatten())
casnearest = np.reshape(interpolate.griddata(points, z, xi, method='cubic'), Xt.shape)


def lakefill(bathy):
    # Reimplementation of JP's fill_in_lakes.m
    # The strategy is to diffuse a tracer from the open boundary
    # through the whole domain in 2D. Any non-land points that the tracer
    # doesn't reach are lakes and we fill them.
    idxland = bathy == 0           # Record initial land points
    ocean = np.zeros(bathy.shape)   
#    ocean[0,:] = 1                 # Put tracer on southern boundary, except for (Salish Sea)
    ocean[:,0] =1                   # Put tracer on western bloundary, except for (in WCVI)
    ocean[idxland]=0               # land points, meaning southern open bdy

    flag, it = True, 0
    stencil = np.array([[0,1,0],[1,0,1],[0,1,0]])  # diffusion schedule
    while flag:
        nocean = np.sum(ocean)
        it += 1
        ocean = signal.convolve2d(ocean, stencil, mode='same')  # Diffusion step
        ocean[idxland]=0   # Reset land to zero
        ocean[ocean>0]=1   # Anywhere that has tracer is now wet
        flag = np.sum(ocean) > nocean
    
    idxwater = ocean == 1  # Define ocean as connected wet points
    idxlakes = (~idxwater) & (~idxland)  # Lakes are not ocean and not land

    bathyout = np.copy(bathy)
    bathyout[idxlakes] = 0     # Fill the lakes

    print ("Lakes filled in {} iterations".format(it))
    return bathyout


casnearest = lakefill(casnearest)


def writebathy(filename,glamt,gphit,bathy):

    bnc = nc.Dataset(filename, 'w', clobber=True)
    NY,NX = glamt.shape

    # Create the dimensions
    bnc.createDimension('x', NX)
    bnc.createDimension('y', NY)

    bnc.createVariable('nav_lon', 'f', ('y', 'x'), zlib=True, complevel=4)
    bnc.variables['nav_lon'].setncattr('units', 'degrees_east')

    bnc.createVariable('nav_lat', 'f', ('y', 'x'), zlib=True, complevel=4)
    bnc.variables['nav_lat'].setncattr('units', 'degrees_north')

    bnc.createVariable('Bathymetry', 'd', ('y', 'x'), zlib=True, complevel=4, fill_value=0)
    bnc.variables['Bathymetry'].setncattr('units', 'metres')

    bnc.variables['nav_lon'][:] = glamt
    bnc.variables['nav_lat'][:] = gphit
    bnc.variables['Bathymetry'][:] = bathy

    bnc.close()




writebathy('bathy_meter_high_res_cubic_truncated.nc',T_grid_lon,T_grid_lat,casnearest)






























