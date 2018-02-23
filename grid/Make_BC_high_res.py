import numpy as np
import numpy.ma as ma
import netCDF4 as nc
from   salishsea_tools import (nc_tools, gsw_calls, viz_tools, geo_tools)
from   scipy.interpolate import griddata, interp1d
import xarray as xr
from   grid_alignment import calculate_initial_compass_bearing as cibc

#mention path to create bdy files

path_bdy = '/ocean/ssahu/CANYONS/bdy_files/high_res/'


gridT = xr.open_dataset('/ocean/ssahu/JP_BC/cat_42_days_T.nc')

mask = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/grid/meshmask_files/JP_mesh_mask.nc')

tmask_JP = mask.variables['tmask'][0,:,55:420,446:730]
umask_JP = mask.variables['umask'][0,:,55:420,446:730]
vmask_JP = mask.variables['vmask'][0,:,55:420,446:730]


ssh_unfiltered = np.array(gridT['sossheig'][:,55:420,446:730])
votemper_unfiltered = np.array(gridT['votemper'][:,:,55:420,446:730])
vosaline_unfiltered = np.array(gridT['vosaline'][:,:,55:420,446:730])


grid_JP = xr.open_dataset('/ocean/ssahu/JP_BC/coordinates_JP.nc')

glamt_bc_JP = np.array(grid_JP.variables['glamt'][55:420,446:730]); gphit_bc_JP = np.array(grid_JP.variables['gphit'][55:420,446:730])

glamu = np.array(grid_JP.variables['glamu'][55:420,446:730]); gphiu = np.array(grid_JP.variables['gphiu'][55:420,446:730])
glamv = np.array(grid_JP.variables['glamv'][55:420,446:730]); gphiv = np.array(grid_JP.variables['gphiv'][55:420,446:730])



for i in np.arange(vosaline_unfiltered.shape[0]):
    vosaline_unfiltered[i,...] = np.ma.masked_array(vosaline_unfiltered[i,...], mask = tmask_JP[...]);
    vosaline_unfiltered[vosaline_unfiltered == 0] = ['Nan'];
    
for i in np.arange(votemper_unfiltered.shape[0]):
    votemper_unfiltered[i,...] = np.ma.masked_array(votemper_unfiltered[i,...], mask = tmask_JP[...]);
    votemper_unfiltered[votemper_unfiltered == 0] = ['Nan'];


print("Starting to use the tidal filter on ssh, temperature and salinity data of JP")

ssh = np.empty(ssh_unfiltered.shape)
votemper = np.empty(votemper_unfiltered.shape)
vosaline = np.empty(vosaline_unfiltered.shape)


for idx in np.arange(ssh_unfiltered.shape[0]):
    ssh[idx-1,...] = ssh_unfiltered[idx-2,...]*0.25 + ssh_unfiltered[idx-1,...]*0.5 + \
                                ssh_unfiltered[idx,...]*0.25
    votemper[idx-1,...] = votemper_unfiltered[idx-2,...]*0.25 + votemper_unfiltered[idx-1,...]*0.5 + \
                                votemper_unfiltered[idx,...]*0.25
    vosaline[idx-1,...] = vosaline_unfiltered[idx-2,...]*0.25 + vosaline_unfiltered[idx-1,...]*0.5 + \
                                vosaline_unfiltered[idx,...]*0.25

ssh[0,...] = ssh_unfiltered[0,...]
votemper[0,...] = votemper_unfiltered[0,...]
vosaline[0,...] = vosaline_unfiltered[0,...]

print("The ssh, temperature and salinity data is filtered successfully")

print("Removing Nan values from JP's data, since we replaced the zero masks with Nans for temperature and salinity")

for t in np.arange(votemper.shape[0]):
    for i in np.arange(1,votemper.shape[1]):
        for p in np.arange(votemper.shape[2]):
            for l in np.arange(votemper.shape[3]):
                if np.isnan(votemper[t,i,p,l]):
                    votemper[t,i,p,l] = votemper[t,i-1,p,l]
                else:
                    continue


for t in np.arange(vosaline.shape[0]):
    for i in np.arange(1,vosaline.shape[1]):
        for p in np.arange(vosaline.shape[2]):
            for l in np.arange(vosaline.shape[3]):
                if np.isnan(vosaline[t,i,p,l]):
                    vosaline[t,i,p,l] = vosaline[t,i-1,p,l]
                else:
                    continue
                
                
for t in np.arange(votemper.shape[0]):
    for i in np.arange(1,votemper.shape[1]):
        for p in np.arange(votemper.shape[2]):
            for l in np.arange(votemper.shape[3]):
                if np.isnan(votemper[t,i,p,l]):
                    votemper[t,i,p,l] = votemper[t,i,p,l-1]
                else:
                    continue


for t in np.arange(vosaline.shape[0]):
    for i in np.arange(1,vosaline.shape[1]):
        for p in np.arange(vosaline.shape[2]):
            for l in np.arange(vosaline.shape[3]):
                if np.isnan(vosaline[t,i,p,l]):
                    vosaline[t,i,p,l] = vosaline[t,i,p,l-1]
                else:
                    continue
                
for t in np.arange(votemper.shape[0]):
    for p in np.arange(votemper.shape[2]):
        for l in np.arange(votemper.shape[3]):
            if np.isnan(votemper[t,0,p,l]):
                votemper[t,0,p,l] = votemper[t,1,p,l]
            else:
                continue
            
            
for t in np.arange(vosaline.shape[0]):
    for p in np.arange(vosaline.shape[2]):
        for l in np.arange(vosaline.shape[3]):
            if np.isnan(vosaline[t,0,p,l]):
                vosaline[t,0,p,l] = vosaline[t,1,p,l]
            else:
                continue

print("Nan Values removed from JP's data and now preparing for horizontal interpolation to WCVI strips across each boundaries")


fname_wcvi = '/ocean/ssahu/CANYONS/wcvi/grid/coordinates.nc'

with nc.Dataset(fname_wcvi, 'r') as coord:
    gphit_wcvi = coord.variables['gphit'][0,...]
    glamt_wcvi = coord.variables['glamt'][0,...]
    glamu_wcvi = coord.variables['glamu'][0,...]
    gphiu_wcvi = coord.variables['gphiu'][0,...]
    glamv_wcvi = coord.variables['glamv'][0,...]
    gphiv_wcvi = coord.variables['gphiv'][0,...]

X = glamt_bc_JP.flatten()

Y = gphit_bc_JP.flatten()

points = (X[:],Y[:])

xi = (glamt_wcvi.flatten(), gphit_wcvi.flatten())

#Number of points to trim off the ends, set equal to the would be FRS rimwidth

N = 11

glamt_wcvi_bc_left = glamt_wcvi[:,0:N-1]; gphit_wcvi_bc_left = gphit_wcvi[:,0:N-1];
glamt_wcvi_bc_right = glamt_wcvi[:,-N:-1]; gphit_wcvi_bc_right = gphit_wcvi[:,-N:-1];

glamt_wcvi_bc_bottom = glamt_wcvi[0:N-1,:]; gphit_wcvi_bc_bottom = gphit_wcvi[0:N-1,:];
glamt_wcvi_bc_top = glamt_wcvi[-N:-1,:]; gphit_wcvi_bc_top = gphit_wcvi[-N:-1,:];

votemper_wcvi_top = np.empty((votemper_unfiltered.shape[0],votemper_unfiltered.shape[1],glamt_wcvi_bc_top.shape[0], glamt_wcvi_bc_top.shape[1]))
votemper_wcvi_bottom = np.empty((votemper_unfiltered.shape[0],votemper_unfiltered.shape[1],glamt_wcvi_bc_bottom.shape[0], glamt_wcvi_bc_bottom.shape[1]))

vosaline_wcvi_top = np.empty((vosaline_unfiltered.shape[0],vosaline_unfiltered.shape[1],glamt_wcvi_bc_top.shape[0], glamt_wcvi_bc_top.shape[1]))
vosaline_wcvi_bottom = np.empty((vosaline_unfiltered.shape[0],vosaline_unfiltered.shape[1],glamt_wcvi_bc_bottom.shape[0], glamt_wcvi_bc_bottom.shape[1]))


votemper_wcvi_left = np.empty((votemper_unfiltered.shape[0],votemper_unfiltered.shape[1],glamt_wcvi_bc_left.shape[0], glamt_wcvi_bc_left.shape[1]))
votemper_wcvi_right = np.empty((votemper_unfiltered.shape[0],votemper_unfiltered.shape[1],glamt_wcvi_bc_right.shape[0], glamt_wcvi_bc_right.shape[1]))

vosaline_wcvi_left = np.empty((vosaline_unfiltered.shape[0],vosaline_unfiltered.shape[1],glamt_wcvi_bc_left.shape[0], glamt_wcvi_bc_left.shape[1]))
vosaline_wcvi_right = np.empty((vosaline_unfiltered.shape[0],vosaline_unfiltered.shape[1],glamt_wcvi_bc_right.shape[0], glamt_wcvi_bc_right.shape[1]))


ssh_wcvi_left = np.empty((ssh_unfiltered.shape[0],glamt_wcvi_bc_left.shape[0], glamt_wcvi_bc_left.shape[1]))
ssh_wcvi_right = np.empty((ssh_unfiltered.shape[0],glamt_wcvi_bc_right.shape[0], glamt_wcvi_bc_right.shape[1]))


ssh_wcvi_top = np.empty((ssh_unfiltered.shape[0],glamt_wcvi_bc_top.shape[0], glamt_wcvi_bc_top.shape[1]))
ssh_wcvi_bottom = np.empty((ssh_unfiltered.shape[0],glamt_wcvi_bc_bottom.shape[0], glamt_wcvi_bc_bottom.shape[1]))

xi_left = (glamt_wcvi_bc_left.flatten(), gphit_wcvi_bc_left.flatten())
xi_right = (glamt_wcvi_bc_right.flatten(), gphit_wcvi_bc_right.flatten())

xi_top = (glamt_wcvi_bc_top.flatten(), gphit_wcvi_bc_top.flatten())
xi_bottom = (glamt_wcvi_bc_bottom.flatten(), gphit_wcvi_bc_bottom.flatten())


for p in np.arange(votemper.shape[0]):
    ssh_wcvi_left[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_left, method= 'linear'), glamt_wcvi_bc_left.shape)
    ssh_wcvi_right[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_right, method= 'linear'), glamt_wcvi_bc_right.shape)
    ssh_wcvi_top[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_top, method= 'linear'), glamt_wcvi_bc_top.shape)
    ssh_wcvi_bottom[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_bottom, method= 'linear'), glamt_wcvi_bc_bottom.shape)
    for i in np.arange(votemper.shape[1]):
        votemper_wcvi_left[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_left, method= 'linear'), glamt_wcvi_bc_left.shape)
        votemper_wcvi_right[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_right, method= 'linear'), glamt_wcvi_bc_right.shape)
        vosaline_wcvi_left[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_left, method= 'linear'), glamt_wcvi_bc_left.shape)
        vosaline_wcvi_right[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_right, method= 'linear'), glamt_wcvi_bc_right.shape)
        votemper_wcvi_top[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_top, method= 'linear'), glamt_wcvi_bc_top.shape)
        votemper_wcvi_bottom[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_bottom, method= 'linear'), glamt_wcvi_bc_bottom.shape)
        vosaline_wcvi_top[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_top, method= 'linear'), glamt_wcvi_bc_top.shape)
        vosaline_wcvi_bottom[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_bottom, method= 'linear'), glamt_wcvi_bc_bottom.shape)

print("The ssh, temperature and Salinity data has been interpolated to WCVI horizontal grid strips successfully")

print("Beginning to take out JP's velocities for transoformation of the vector fields")


gridU = xr.open_dataset('/home/ssahu/saurav/JP_BC/cat_43_U.nc')
gridV = xr.open_dataset('/home/ssahu/saurav/JP_BC/cat_43_V.nc')


nav_lon_U = np.array(gridU['nav_lon'][55:420,446:730])
nav_lat_U = np.array(gridU['nav_lat'][55:420,446:730])

nav_lon_V = np.array(gridV['nav_lon'][55:420,446:730])
nav_lat_V = np.array(gridV['nav_lat'][55:420,446:730]) 

U_vel_BC_unfiltered = np.array(gridU['vozocrtx'][:,:,55:420,446:730])
V_vel_BC_unfiltered = np.array(gridV['vomecrty'][:,:,55:420,446:730])


for a in np.arange(U_vel_BC_unfiltered.shape[0]):
    for i in np.arange(U_vel_BC_unfiltered.shape[1]):
        for l in np.arange(U_vel_BC_unfiltered.shape[2]):
            for m in np.arange(U_vel_BC_unfiltered.shape[3]):
                if U_vel_BC_unfiltered[a,i,l,m] == 0:
                    U_vel_BC_unfiltered[a,i,l,m] == [];


for a in np.arange(V_vel_BC_unfiltered.shape[0]):
    for i in np.arange(V_vel_BC_unfiltered.shape[1]):
        for l in np.arange(V_vel_BC_unfiltered.shape[2]):
            for m in np.arange(V_vel_BC_unfiltered.shape[3]):
                if V_vel_BC_unfiltered[a,i,l,m] == 0:
                    V_vel_BC_unfiltered[a,i,l,m] == [];

U_vel_BC = np.empty(U_vel_BC_unfiltered.shape)
V_vel_BC = np.empty(V_vel_BC_unfiltered.shape)

print("Starting to filter the JP velocities")

for idx, val in enumerate(U_vel_BC_unfiltered[:,...]):
    U_vel_BC[idx-1,...] = U_vel_BC_unfiltered[idx-2,...]*0.25 + U_vel_BC_unfiltered[idx-1,...]*0.5 + \
                               U_vel_BC_unfiltered[idx,...]*0.25
    V_vel_BC[idx-1,...] = V_vel_BC_unfiltered[idx-2,...]*0.25 + V_vel_BC_unfiltered[idx-1,...]*0.5 + \
                               V_vel_BC_unfiltered[idx,...]*0.25

print("The velocities are filtered")

for i in np.arange(U_vel_BC.shape[0]):
    U_vel_BC[i,...] = ma.masked_array(U_vel_BC[i,...], mask = umask_JP[...])
    V_vel_BC[i,...] = ma.masked_array(V_vel_BC[i, ...], mask = vmask_JP[...])

print("Beginning to do the grid transformation process for the vector fields to match WCVI domain")


# "Unstagger" the velocity values by interpolating them to the T-grid points
u_vel_BC_tzyx, v_vel_BC_tzyx = viz_tools.unstagger(U_vel_BC, V_vel_BC)

mag_vel_BC = np.sqrt(np.multiply(u_vel_BC_tzyx,u_vel_BC_tzyx), np.multiply(v_vel_BC_tzyx,v_vel_BC_tzyx));
ang_vel_BC = np.degrees(np.arctan2(v_vel_BC_tzyx, u_vel_BC_tzyx));


# First point
lonA = glamu[:,0:-1]
latA = gphiu[:,0:-1]
# Second point
lonB = glamu[:,1:]
latB = gphiu[:,1:]


bearing = cibc((latA,lonA),(latB,lonB))
angle_needed = 90 - bearing;

angle_unrotated = np.empty((ang_vel_BC.shape[0],ang_vel_BC.shape[1],ang_vel_BC.shape[2], ang_vel_BC.shape[3]))


for p in np.arange(ang_vel_BC.shape[0]):
    for i in np.arange(ang_vel_BC.shape[1]):
        angle_unrotated[p,i,...] = ang_vel_BC[p,i,...] + angle_needed[1:,:]
                               
                               
u_unrotated = mag_vel_BC*np.cos(np.radians(angle_unrotated))
v_unrotated = mag_vel_BC*np.sin(np.radians(angle_unrotated))


# First point
lonA_wcvi = glamu_wcvi[:,0:-1]
latA_wcvi = gphiu_wcvi[:,0:-1]
# Second point
lonB_wcvi = glamu_wcvi[:,1:]
latB_wcvi = gphiu_wcvi[:,1:]


bearing_wcvi = cibc((latA_wcvi,lonA_wcvi),(latB_wcvi,lonB_wcvi))
angle_needed_wcvi = 90 - bearing_wcvi



glamt_cut_JP = glamt_bc_JP[1:,1:]
gphit_cut_JP = gphit_bc_JP[1:,1:]

X = np.array(glamt_cut_JP).flatten()
Y = np.array(gphit_cut_JP).flatten()

points = (X[:],Y[:]);

xi = (glamt_wcvi.flatten(), gphit_wcvi.flatten());

u_unrotated_wcvi_t = np.empty((u_unrotated.shape[0],u_unrotated.shape[1],glamt_wcvi.shape[0], glamt_wcvi.shape[1]));
v_unrotated_wcvi_t = np.empty((v_unrotated.shape[0],v_unrotated.shape[1],glamt_wcvi.shape[0], glamt_wcvi.shape[1]));

for p in np.arange(u_unrotated.shape[0]):
    for i in np.arange(v_unrotated.shape[1]):
        u_unrotated_wcvi_t[p,i,...] = np.reshape(griddata(points, u_unrotated[p,i,:,...].flatten(), \
                                                xi, method= 'linear'), glamt_wcvi.shape)
        v_unrotated_wcvi_t[p,i,...] = np.reshape(griddata(points, v_unrotated[p,i,...].flatten(), \
                                                xi, method= 'linear'), glamt_wcvi.shape)


mag_vel_at_wcvi = np.sqrt(np.multiply(u_unrotated_wcvi_t,u_unrotated_wcvi_t), \
                         np.multiply(v_unrotated_wcvi_t,v_unrotated_wcvi_t));
ang_vel_at_wcvi = np.degrees(np.arctan2(v_unrotated_wcvi_t, u_unrotated_wcvi_t));

angle_unrotated_wcvi = ang_vel_at_wcvi[...,:,1:] - angle_needed_wcvi;

u_unrotated_rotated_to_wcvi = mag_vel_at_wcvi[...,:,1:]*(np.cos(np.radians(angle_unrotated_wcvi[...])));
v_unrotated_rotated_to_wcvi = mag_vel_at_wcvi[...,:,1:]*(np.sin(np.radians(angle_unrotated_wcvi[...])));


def stagger(ugrid, vgrid):
    u = np.add(ugrid[...,:-1], ugrid[...,1:]) / 2;
    v = np.add(vgrid[...,:-1, :], vgrid[...,1:, :]) / 2;
    return u[...,:, :], v[...,:, :]


u_rotated_WCVI, v_rotated_WCVI = stagger(u_unrotated_rotated_to_wcvi, v_unrotated_rotated_to_wcvi)

print("Vector Grid transformation complete and veloctities staggered")

U_wcvi_top = u_rotated_WCVI[...,-N:-1,:]
V_wcvi_top = v_rotated_WCVI[...,-N:-1,:]

U_wcvi_bottom = u_rotated_WCVI[...,0:N-1,:]
V_wcvi_bottom = v_rotated_WCVI[...,0:N-1,:]

U_wcvi_left = u_rotated_WCVI[...,:,0:N-1]
V_wcvi_left = v_rotated_WCVI[...,:,0:N-1]

U_wcvi_right = u_rotated_WCVI[...,:,-N:-1]
V_wcvi_right = v_rotated_WCVI[...,:,-N:-1]


print("Saving them into numpy arrays")

path = '/ocean/ssahu/CANYONS/bdy_files/numpy_arrays_saved/'


np.save(path+'U_3D_wcvi_top.npy', U_wcvi_top);       np.save(path+'V_3D_wcvi_top.npy', V_wcvi_top);
np.save(path+'U_3D_wcvi_bottom.npy', U_wcvi_bottom); np.save(path+'V_3D_wcvi_bottom.npy', V_wcvi_bottom);
np.save(path+'U_3D_wcvi_left.npy', U_wcvi_left);     np.save(path+'V_3D_wcvi_left.npy', V_wcvi_left);
np.save(path+'U_3D_wcvi_right.npy', U_wcvi_right);   np.save(path+'V_3D_wcvi_right.npy', V_wcvi_right);

np.save(path+'votemper_leftbc.npy', votemper_wcvi_left);          np.save(path+'votemper_rightbc.npy', votemper_wcvi_right)
np.save(path+'ssh_leftbc.npy', ssh_wcvi_left);                    np.save(path+'ssh_rightbc.npy', ssh_wcvi_right)
np.save(path+'vosaline_leftbc.npy', vosaline_wcvi_left);          np.save(path+'vosaline_rightbc.npy', vosaline_wcvi_right);
np.save(path+'ssh_topbc.npy', ssh_wcvi_top);                      np.save(path+'ssh_bottombc.npy', ssh_wcvi_bottom);
np.save(path+'votemper_topbc.npy', votemper_wcvi_top);            np.save(path+'votemper_bottombc.npy', votemper_wcvi_bottom);
np.save(path+'vosaline_topbc.npy', vosaline_wcvi_top);            np.save(path+'vosaline_bottombc.npy', vosaline_wcvi_bottom);



print("Working first with the West Boundary")


u_3d_left         = U_wcvi_left  
v_3d_left_oneless = V_wcvi_left 
ssh_left          = ssh_wcvi_left 
votemper_left     = votemper_wcvi_left
vosaline_left     = vosaline_wcvi_left


u_2d_left = np.mean(u_3d_left, axis= 1)

v_2d_left_oneless = np.mean(v_3d_left_oneless, axis=1)

v_2d_left = np.empty(u_2d_left.shape) 
v_2d_left[:,:-1,:] = v_2d_left_oneless
v_2d_left[:,-1,:] = v_2d_left[:,-2,:]

v_3d_left = np.empty(u_3d_left.shape)
v_3d_left[:,:,:-1,:] = v_3d_left_oneless
v_3d_left[:,:,-1,:] = v_3d_left[:,:,-2,:]


# Nemo reads as t,z,ybt,xbt so the largest dimension should be the last one meaning we need to swap axes for a couple of sides

print("Working with the west barotropic or 2d bdy conditions")

ssh_left_switched  = np.swapaxes(ssh_left, 1, 2)
u_2d_left_switched = np.swapaxes(u_2d_left, 1, 2)
v_2d_left_switched = np.swapaxes(v_2d_left, 1, 2)

ssh_left_twoless  = ssh_left_switched[:,:,1:-1]
u_2d_left_twoless = u_2d_left_switched[:,:,1:-1]
v_2d_left_twoless = v_2d_left_switched[:,:,1:-1]

Nt = ssh_left_twoless.shape[0]
Ny = ssh_left_twoless.shape[1]
Nx = ssh_left_twoless.shape[2]

sossheig_2d_left = np.reshape(a=ssh_left_twoless,  newshape = (Nt,1,Nx*Ny))
vobtcrtx_2d_left = np.reshape(a=u_2d_left_twoless, newshape = (Nt,1,Nx*Ny))
vobtcrty_2d_left = np.reshape(a=v_2d_left_twoless, newshape = (Nt,1,Nx*Ny))



print("Writing the 2d West bdy conditions")

bdy_file = nc.Dataset(path_bdy + '2d_west_flather_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', sossheig_2d_left.shape[2]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'indices along left boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'a strip of indices across all of left boundary';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

sossheig = bdy_file.createVariable('sossheig', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrtx = bdy_file.createVariable('vobtcrtx', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrty = bdy_file.createVariable('vobtcrty', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);


sossheig[...] = sossheig_2d_left[...];
vobtcrtx[...] = vobtcrtx_2d_left[...];
vobtcrty[...] = vobtcrty_2d_left[...];

bdy_file.close()

print("2D West Boundary file written successfully")


print("Working with the 3D NEMO West boundary conditions")

U_vel_JP_level_left        = np.swapaxes(u_3d_left, 3, 2)
V_vel_JP_level_left        = np.swapaxes(v_3d_left, 3, 2)
votemper_JP_level_PT_left  = np.swapaxes(votemper_left, 3, 2)
vosaline_JP_level_PSU_left = np.swapaxes(vosaline_left, 3, 2)

print("Calling GSW tools to convert to Conservative Temperature and Reference Salinity for west bdy")

mask_wcvi = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/grid/meshmask_files/mesh_mask_high_res.nc')

NEMO_depth_T = mask_wcvi.variables['gdept_0'][0,:,0,0]

depth_JP_T = nc.Dataset('/ocean/ssahu/JP_BC/cat_42_days_T.nc').variables['deptht'][:]

lat = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/initial_conditions/West_coast_NEMO_IC_high_resolution.nc')\
           .variables['nav_lat'][:]
lon = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/initial_conditions/West_coast_NEMO_IC_high_resolution.nc')\
           .variables['nav_lon'][:]


# Conversion of vosaline_PSU to vosaline_RS and votemper_PT to votemper_CT

z = np.dot(-1,depth_JP_T);

pressure = np.empty(z.shape)
lats = np.empty(pressure.shape)
lons = np.empty(pressure.shape)
lats[:] = np.mean(lat)
lons[:] = np.mean(lon)


vosaline_JP_level_SA_left = np.empty(vosaline_JP_level_PSU_left.shape)
vosaline_JP_level_RS_left = np.empty(vosaline_JP_level_PSU_left.shape)
votemper_JP_level_CT_left = np.empty(vosaline_JP_level_PSU_left.shape)

pressure = gsw_calls.generic_gsw_caller('gsw_p_from_z.m', [z, np.mean(lat)])

print("Converted z to p: first GSW call successful")

for i in np.arange(vosaline_JP_level_SA_left.shape[1]):
    vosaline_JP_level_SA_left[:,i,...] = gsw_calls.generic_gsw_caller('gsw_SA_from_SP', [vosaline_JP_level_PSU_left[:,i,...],pressure[i],lons[i],lats[i]])

print("Got SA from SP: GSW ran successfully inside loop")    
    
vosaline_JP_level_RS_left[:] = gsw_calls.generic_gsw_caller('gsw_SR_from_SP', [vosaline_JP_level_PSU_left[:]])

print("Reference Salinity obtained from PS: one more GSW call left")

votemper_JP_level_CT_left[:] = gsw_calls.generic_gsw_caller('gsw_CT_from_pt', [vosaline_JP_level_SA_left[:], votemper_JP_level_PT_left[:]])

print("GSW Calls successfull for west bdy")

print("Beginning Vertical interpolation of west BC")

left_temp_function = interp1d(depth_JP_T, votemper_JP_level_CT_left, axis=1,\
                             bounds_error=False, fill_value='extrapolate')
left_sal_function = interp1d(depth_JP_T, vosaline_JP_level_RS_left, axis=1,\
                             bounds_error=False, fill_value='extrapolate')
left_U_function = interp1d(depth_JP_T, U_vel_JP_level_left, axis=1,\
                             bounds_error=False, fill_value='extrapolate')
left_V_function = interp1d(depth_JP_T, V_vel_JP_level_left, axis=1,\
                             bounds_error=False, fill_value='extrapolate')

votemper_NEMO_left = np.empty((vosaline_JP_level_SA_left.shape[0], NEMO_depth_T.shape[0], \
                              vosaline_JP_level_SA_left.shape[2], vosaline_JP_level_SA_left.shape[3]))
vosaline_NEMO_left = np.empty((vosaline_JP_level_SA_left.shape[0], NEMO_depth_T.shape[0], \
                              vosaline_JP_level_SA_left.shape[2], vosaline_JP_level_SA_left.shape[3]))
U_NEMO_left = np.empty((vosaline_JP_level_SA_left.shape[0], NEMO_depth_T.shape[0], \
                              vosaline_JP_level_SA_left.shape[2], vosaline_JP_level_SA_left.shape[3]))
V_NEMO_left = np.empty((vosaline_JP_level_SA_left.shape[0], NEMO_depth_T.shape[0], \
                              vosaline_JP_level_SA_left.shape[2], vosaline_JP_level_SA_left.shape[3]))


for indx in np.arange(NEMO_depth_T.shape[0]):
    votemper_NEMO_left[:,indx,...] = left_temp_function(NEMO_depth_T[indx])
    vosaline_NEMO_left[:,indx,...] = left_sal_function(NEMO_depth_T[indx])
    U_NEMO_left[:,indx,...]        = left_U_function(NEMO_depth_T[indx])
    V_NEMO_left[:,indx,...]        = left_V_function(NEMO_depth_T[indx])

print("Vertical Interpolation of West BC complete")

for i in np.arange(vosaline_NEMO_left.shape[0]):
    for j in np.arange(vosaline_NEMO_left.shape[1]):
        for k in np.arange(vosaline_NEMO_left.shape[2]):
            for l in np.arange(vosaline_NEMO_left.shape[3]):
                if np.isnan(vosaline_NEMO_left[i,j,k,l]):
                    vosaline_NEMO_left[i,j,k,l] = vosaline_NEMO_left[i,j-1,k,l]
                else:
                    continue

for i in np.arange(votemper_NEMO_left.shape[0]):
    for j in np.arange(votemper_NEMO_left.shape[1]):
        for k in np.arange(votemper_NEMO_left.shape[2]):
            for l in np.arange(votemper_NEMO_left.shape[3]):
                if np.isnan(votemper_NEMO_left[i,j,k,l]):
                    votemper_NEMO_left[i,j,k,l] = votemper_NEMO_left[i,j-1,k,l]
                else:
                    continue


votemper_NEMO_left_twoless = votemper_NEMO_left[:,:,:,1:-1]                    
vosaline_NEMO_left_twoless = vosaline_NEMO_left[:,:,:,1:-1]
U_NEMO_left_twoless = U_NEMO_left[:,:,:,1:-1]
V_NEMO_left_twoless = V_NEMO_left[:,:,:,1:-1]


Nt = votemper_NEMO_left_twoless.shape[0]
Nz = votemper_NEMO_left_twoless.shape[1]
Ny = votemper_NEMO_left_twoless.shape[2]
Nx = votemper_NEMO_left_twoless.shape[3]

vozocrtx_3d_left = np.reshape(a = U_NEMO_left_twoless,        newshape= (Nt,Nz,1,Nx*Ny))
vomecrty_3d_left = np.reshape(a = V_NEMO_left_twoless,        newshape= (Nt,Nz,1,Nx*Ny))
votemper_3d_left = np.reshape(a = votemper_NEMO_left_twoless, newshape= (Nt,Nz,1,Nx*Ny))
vosaline_3d_left = np.reshape(a = vosaline_NEMO_left_twoless, newshape= (Nt,Nz,1,Nx*Ny))

#Now let us write the 3d boundary condition for the left boundary
print("Writing the 3D west bdy conditions")

bdy_file = nc.Dataset(path + '3d_west_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', vozocrtx_3d_left.shape[3]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('deptht', vozocrtx_3d_left.shape[1]);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'x indices along left boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'one y index across all of left boundary';

deptht = bdy_file.createVariable('deptht', 'float32', ('deptht',), zlib=True);
deptht.units = 'm';
deptht.longname = 'Vertical T Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

vozocrtx = bdy_file.createVariable('vozocrtx', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vomecrty = bdy_file.createVariable('vomecrty', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
votemper = bdy_file.createVariable('votemper', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vosaline = bdy_file.createVariable('vosaline', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);

vozocrtx[...] = vozocrtx_3d_left[...];
vomecrty[...] = vomecrty_3d_left[...];
votemper[...] = votemper_3d_left[...];
vosaline[...] = vosaline_3d_left[...];

bdy_file.close()

print("West Boundary 3D file is written successfully")

print("Working with the East boundary")


u_3d_right         = U_wcvi_right
v_3d_right_oneless = V_wcvi_right
ssh_right          = ssh_wcvi_right 
votemper_right     = votemper_wcvi_right
vosaline_right     = vosaline_wcvi_right

u_2d_right         = np.mean(u_3d_right, axis= 1)
v_2d_right_oneless = np.mean(v_3d_right_oneless, axis=1)

v_2d_right          = np.empty(u_2d_right.shape) 
v_2d_right[:,:-1,:] = v_2d_right_oneless
v_2d_right[:,-1,:]  = v_2d_right[:,-2,:]

v_3d_right            = np.empty(u_3d_right.shape)
v_3d_right[:,:,:-1,:] = v_3d_right_oneless
v_3d_right[:,:,-1,:]  = v_3d_right[:,:,-2,:]


print("Working with the east barotropic or 2d bdy conditions")

ssh_right_switched  = np.swapaxes(ssh_right, 1, 2)
u_2d_right_switched = np.swapaxes(u_2d_right, 1, 2)
v_2d_right_switched = np.swapaxes(v_2d_right, 1, 2)

ssh_right_twoless  = ssh_right_switched[:,:,1:-1]
u_2d_right_twoless = u_2d_right_switched[:,:,1:-1]
v_2d_right_twoless = v_2d_right_switched[:,:,1:-1]

Nt = ssh_right_twoless.shape[0]
Ny = ssh_right_twoless.shape[1]
Nx = ssh_right_twoless.shape[2]

sossheig_right_flipped = np.flip(ssh_right_twoless, axis=1)
vobtcrtx_right_flipped = np.flip(u_2d_right_twoless, axis=1)
vobtcrty_right_flipped = np.flip(v_2d_right_twoless, axis=1)

sossheig_2d_right = np.reshape(a=sossheig_right_flipped,  newshape = (Nt,1,Nx*Ny))
vobtcrtx_2d_right = np.reshape(a=vobtcrtx_right_flipped,  newshape = (Nt,1,Nx*Ny))
vobtcrty_2d_right = np.reshape(a=vobtcrty_right_flipped,  newshape = (Nt,1,Nx*Ny))

print("Writing the 2d East bdy conditions")

bdy_file = nc.Dataset(path_bdy + '2d_right_flather_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', sossheig_2d_right.shape[2]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'indices along right boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'a strip of indices across all of right boundary';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

sossheig = bdy_file.createVariable('sossheig', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrtx = bdy_file.createVariable('vobtcrtx', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrty = bdy_file.createVariable('vobtcrty', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);


sossheig[...] = sossheig_2d_right[...];
vobtcrtx[...] = vobtcrtx_2d_right[...];
vobtcrty[...] = vobtcrty_2d_right[...];

bdy_file.close()

print("2d East bdy conditions are written successfully")



print("Working with the 3D NEMO East boundary conditions")

U_vel_JP_level_right        = np.swapaxes(u_3d_right, 3, 2)[:];
V_vel_JP_level_right        = np.swapaxes(v_3d_right, 3, 2)[:];
votemper_JP_level_PT_right  = np.swapaxes(votemper_right, 3, 2)[:];
vosaline_JP_level_PSU_right = np.swapaxes(vosaline_right, 3, 2)[:];


vosaline_JP_level_SA_right = np.empty(vosaline_JP_level_PSU_right.shape);
vosaline_JP_level_RS_right = np.empty(vosaline_JP_level_PSU_right.shape);
votemper_JP_level_CT_right = np.empty(vosaline_JP_level_PSU_right.shape);


pressure = gsw_calls.generic_gsw_caller('gsw_p_from_z.m', [z, np.mean(lat)]);

print("Converted z to p: first GSW call successful")

for i in np.arange(vosaline_JP_level_SA_right.shape[1]):
    vosaline_JP_level_SA_right[:,i,...] = gsw_calls.generic_gsw_caller(\
                                    'gsw_SA_from_SP', [vosaline_JP_level_PSU_right[:,i,...],pressure[i],lons[i],lats[i]]);

print("Got SA from SP: GSW ran successfully inside loop")     

vosaline_JP_level_RS_right[:] = gsw_calls.generic_gsw_caller('gsw_SR_from_SP', [vosaline_JP_level_PSU_right[:]]);

print("Reference Salinity obtained from PS: one more GSW call left")

votemper_JP_level_CT_right[:] = gsw_calls.generic_gsw_caller('gsw_CT_from_pt', [vosaline_JP_level_SA_right[:], votemper_JP_level_PT_right[:]]);    

print("GSW Calls successfull for east bdy")

print("Beginning Vertical interpolation of east BC")

right_temp_function = interp1d(depth_JP_T, votemper_JP_level_CT_right, axis=1,\
                              bounds_error=False, fill_value='extrapolate');
right_sal_function = interp1d(depth_JP_T, vosaline_JP_level_RS_right, axis=1,\
                              bounds_error=False, fill_value='extrapolate');
right_U_function = interp1d(depth_JP_T, U_vel_JP_level_right,axis=1,\
                              bounds_error=False, fill_value='extrapolate');
right_V_function = interp1d(depth_JP_T, V_vel_JP_level_right, axis=1,\
                              bounds_error=False, fill_value='extrapolate');


votemper_NEMO_right = np.empty((vosaline_JP_level_SA_right.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_right.shape[2], vosaline_JP_level_SA_right.shape[3]));
vosaline_NEMO_right = np.empty((vosaline_JP_level_SA_right.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_right.shape[2], vosaline_JP_level_SA_right.shape[3]));
U_NEMO_right = np.empty((vosaline_JP_level_SA_right.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_right.shape[2], vosaline_JP_level_SA_right.shape[3]));
V_NEMO_right = np.empty((vosaline_JP_level_SA_right.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_right.shape[2], vosaline_JP_level_SA_right.shape[3]));


for indx in np.arange(NEMO_depth_T.shape[0]):
    votemper_NEMO_right[:,indx,...] = right_temp_function(NEMO_depth_T[indx]);
    vosaline_NEMO_right[:,indx,...] = right_sal_function(NEMO_depth_T[indx]);
    U_NEMO_right[:,indx,...]        = right_U_function(NEMO_depth_T[indx]);
    V_NEMO_right[:,indx,...]        = right_V_function(NEMO_depth_T[indx]);

print("Vertical Interpolation of East BC complete")

for i in np.arange(vosaline_NEMO_right.shape[0]):
    for j in np.arange(vosaline_NEMO_right.shape[1]):
        for k in np.arange(vosaline_NEMO_right.shape[2]):
            for l in np.arange(vosaline_NEMO_right.shape[3]):
                if np.isnan(vosaline_NEMO_right[i,j,k,l]):
                    vosaline_NEMO_right[i,j,k,l] = vosaline_NEMO_right[i,j-1,k,l]
                else:
                    continue
            

for i in np.arange(votemper_NEMO_right.shape[0]):
    for j in np.arange(votemper_NEMO_right.shape[1]):
        for k in np.arange(votemper_NEMO_right.shape[2]):
            for l in np.arange(votemper_NEMO_right.shape[3]):
                if np.isnan(votemper_NEMO_right[i,j,k,l]):
                    votemper_NEMO_right[i,j,k,l] = votemper_NEMO_right[i,j-1,k,l]
                else:
                    continue 
                    
for i in np.arange(vosaline_NEMO_right.shape[0]):
    for j in np.arange(vosaline_NEMO_right.shape[1]):
        for k in np.arange(vosaline_NEMO_right.shape[2]):
            for l in np.arange(vosaline_NEMO_right.shape[3]):
                if np.isnan(vosaline_NEMO_right[i,j,k,l]):
                    vosaline_NEMO_right[i,j,k,l] = vosaline_NEMO_right[i,j,k,l-1]
                else:
                    continue
                    
for i in np.arange(votemper_NEMO_right.shape[0]):
    for j in np.arange(votemper_NEMO_right.shape[1]):
        for k in np.arange(votemper_NEMO_right.shape[2]):
            for l in np.arange(votemper_NEMO_right.shape[3]):
                if np.isnan(votemper_NEMO_right[i,j,k,l]):
                    votemper_NEMO_right[i,j,k,l] = votemper_NEMO_right[i,j,k,l-1]
                else:
                    continue


votemper_NEMO_right_twoless = votemper_NEMO_right[:,:,:,1:-1]                    
vosaline_NEMO_right_twoless = vosaline_NEMO_right[:,:,:,1:-1]
U_NEMO_right_twoless = U_NEMO_right[:,:,:,1:-1]
V_NEMO_right_twoless = V_NEMO_right[:,:,:,1:-1]


Nt = votemper_NEMO_right_twoless.shape[0]
Nz = votemper_NEMO_right_twoless.shape[1]
Ny = votemper_NEMO_right_twoless.shape[2]
Nx = votemper_NEMO_right_twoless.shape[3]


vozocrtx_flipped_right = np.flip(U_NEMO_right_twoless, axis=2)
vomecrty_flipped_right = np.flip(V_NEMO_right_twoless, axis=2)
votemper_flipped_right = np.flip(votemper_NEMO_right_twoless, axis=2)
vosaline_flipped_right = np.flip(vosaline_NEMO_right_twoless, axis=2)



vozocrtx_3d_right = np.reshape(a=vozocrtx_flipped_right, newshape= (Nt,Nz,1,Nx*Ny))
vomecrty_3d_right = np.reshape(a=vomecrty_flipped_right, newshape= (Nt,Nz,1,Nx*Ny))
votemper_3d_right = np.reshape(a=votemper_flipped_right, newshape= (Nt,Nz,1,Nx*Ny))
vosaline_3d_right = np.reshape(a=vosaline_flipped_right, newshape= (Nt,Nz,1,Nx*Ny))


print("Writing the 3D east bdy conditions")

bdy_file = nc.Dataset(path_bdy + '3d_east_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', vozocrtx_3d_right.shape[3]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('deptht', vozocrtx_3d_right.shape[1]);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'x indices along east boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'one y index across all of east boundary';

deptht = bdy_file.createVariable('deptht', 'float32', ('deptht',), zlib=True);
deptht.units = 'm';
deptht.longname = 'Vertical T Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

vozocrtx = bdy_file.createVariable('vozocrtx', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vomecrty = bdy_file.createVariable('vomecrty', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
votemper = bdy_file.createVariable('votemper', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vosaline = bdy_file.createVariable('vosaline', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);

vozocrtx[...] = vozocrtx_3d_right[...];
vomecrty[...] = vomecrty_3d_right[...];
votemper[...] = votemper_3d_right[...];
vosaline[...] = vosaline_3d_right[...];

bdy_file.close()   

print("The East 3d bdy conditions are written successfully")
                    
                    
print("Working with the North boundary")

u_3d_top_twoless         = U_wcvi_top
v_3d_top_oneless         = V_wcvi_top
ssh_top                  = ssh_wcvi_top 
votemper_top             = votemper_wcvi_top
vosaline_top             = vosaline_wcvi_top

u_2d_top_twoless = np.mean(u_3d_top_twoless, axis= 1)
v_2d_top_oneless = np.mean(v_3d_top_oneless, axis= 1)


u_2d_top = np.empty((votemper_top.shape[0],votemper_top.shape[2],votemper_top.shape[3])); 
u_2d_top[:,:,:-2] = u_2d_top_twoless;
u_2d_top[:,:,-1]  = u_2d_top[:,:,-3];
u_2d_top[:,:,-2]  = u_2d_top[:,:,-3];

u_3d_top = np.empty((votemper_top.shape[0],votemper_top.shape[1],votemper_top.shape[2],votemper_top.shape[3])); 
u_3d_top[:,:,:,:-2] = u_3d_top_twoless;
u_3d_top[:,:,:,-1]  = u_3d_top[:,:,:,-3];
u_3d_top[:,:,:,-2]  = u_3d_top[:,:,:,-3];


v_2d_top = np.empty((votemper_top.shape[0],votemper_top.shape[2],votemper_top.shape[3])); 
v_2d_top[:,:,:-1] = v_2d_top_oneless;
v_2d_top[:,:,-1] = v_2d_top[:,:,-2];


v_3d_top = np.empty((votemper_top.shape[0],votemper_top.shape[1],votemper_top.shape[2],votemper_top.shape[3])); 
v_3d_top[:,:,:,:-1] = v_3d_top_oneless;
v_3d_top[:,:,:,-1] = v_3d_top[:,:,:,-2];


print("Working with the north barotropic or 2d bdy conditions")


ssh_north_twoless  = ssh_top [:,:,1:-1]
u_2d_north_twoless = u_2d_top[:,:,1:-1]
v_2d_north_twoless = v_2d_top[:,:,1:-1]

Nt = ssh_north_twoless.shape[0]
Ny = ssh_north_twoless.shape[1]
Nx = ssh_north_twoless.shape[2]

sossheig_flipped_top = np.flip(ssh_north_twoless, axis=1)
vobtcrtx_flipped_top = np.flip(u_2d_north_twoless, axis=1)
vobtcrty_flipped_top = np.flip(v_2d_north_twoless, axis=1)

sossheig_2d_top = np.reshape(a=sossheig_flipped_top, newshape= (Nt,1,Nx*Ny))
vobtcrtx_2d_top = np.reshape(a=vobtcrtx_flipped_top, newshape= (Nt,1,Nx*Ny))
vobtcrty_2d_top = np.reshape(a=vobtcrty_flipped_top, newshape= (Nt,1,Nx*Ny))

print("Writing the 2d North bdy conditions")

bdy_file = nc.Dataset(path_bdy + '2d_north_flather_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', sossheig_2d_top.shape[2]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'indices along north boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'a strip of indices across all of north boundary';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

sossheig = bdy_file.createVariable('sossheig', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrtx = bdy_file.createVariable('vobtcrtx', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrty = bdy_file.createVariable('vobtcrty', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);


sossheig[...] = sossheig_2d_top[...];
vobtcrtx[...] = vobtcrtx_2d_top[...];
vobtcrty[...] = vobtcrty_2d_top[...];

bdy_file.close()

print("The 2d North Boundary Conditions are written")

vosaline_JP_level_PSU_north = vosaline_top[:];
votemper_JP_level_PT_north  = votemper_top[:];
U_vel_JP_level_north        = u_3d_top[:];
V_vel_JP_level_north        = v_3d_top[:];



vosaline_JP_level_SA_north = np.empty(vosaline_JP_level_PSU_north.shape);
vosaline_JP_level_RS_north = np.empty(vosaline_JP_level_PSU_north.shape);
votemper_JP_level_CT_north = np.empty(vosaline_JP_level_PSU_north.shape);

pressure = gsw_calls.generic_gsw_caller('gsw_p_from_z.m', [z, np.mean(lat)]);
print("Converted z to p: first GSW call successful")
for i,j in enumerate(vosaline_JP_level_SA_north[0,:,...]):
    vosaline_JP_level_SA_north[:,i,...] = gsw_calls.generic_gsw_caller(\
                                    'gsw_SA_from_SP', [vosaline_JP_level_PSU_north[:,i,...],pressure[i],lons[i],lats[i]]);
print("Got SA from SP: GSW ran successfully inside loop") 
vosaline_JP_level_RS_north[:] = gsw_calls.generic_gsw_caller('gsw_SR_from_SP', [vosaline_JP_level_PSU_north[:]]);

print("Reference Salinity obtained from PS: one more GSW call left")
votemper_JP_level_CT_north[:] = gsw_calls.generic_gsw_caller('gsw_CT_from_pt', [vosaline_JP_level_SA_north[:], votemper_JP_level_PT_north[:]]);        
print("GSW calls successfull for the north bdy")


north_temp_function = interp1d(depth_JP_T, votemper_JP_level_CT_north, axis=1,\
                              bounds_error=False, fill_value='extrapolate');
north_sal_function = interp1d(depth_JP_T, vosaline_JP_level_RS_north, axis=1,\
                              bounds_error=False, fill_value='extrapolate');
north_U_function = interp1d(depth_JP_T, U_vel_JP_level_north,axis=1,\
                              bounds_error=False, fill_value='extrapolate');
north_V_function = interp1d(depth_JP_T, V_vel_JP_level_north, axis=1,\
                              bounds_error=False, fill_value='extrapolate');


votemper_NEMO_north = np.empty((vosaline_JP_level_SA_north.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_north.shape[2], vosaline_JP_level_SA_north.shape[3]));
vosaline_NEMO_north = np.empty((vosaline_JP_level_SA_north.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_north.shape[2], vosaline_JP_level_SA_north.shape[3]));
U_NEMO_north = np.empty((vosaline_JP_level_SA_north.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_north.shape[2], vosaline_JP_level_SA_north.shape[3]));
V_NEMO_north = np.empty((vosaline_JP_level_SA_north.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_north.shape[2], vosaline_JP_level_SA_north.shape[3]));


for indx in np.arange(NEMO_depth_T.shape[0]):
    votemper_NEMO_north[:,indx,...] = north_temp_function(NEMO_depth_T[indx]);
    vosaline_NEMO_north[:,indx,...] = north_sal_function(NEMO_depth_T[indx]);
    U_NEMO_north[:,indx,...]        = north_U_function(NEMO_depth_T[indx]);
    V_NEMO_north[:,indx,...]        = north_V_function(NEMO_depth_T[indx]);

for i in np.arange(vosaline_NEMO_north.shape[0]):
    for j in np.arange(vosaline_NEMO_north.shape[1]):
        for k in np.arange(vosaline_NEMO_north.shape[2]):
            for l in np.arange(vosaline_NEMO_north.shape[3]):
                if np.isnan(vosaline_NEMO_north[i,j,k,l]):
                    vosaline_NEMO_north[i,j,k,l] = vosaline_NEMO_north[i,j-1,k,l]
                else:
                    continue
            

for i in np.arange(votemper_NEMO_north.shape[0]):
    for j in np.arange(votemper_NEMO_north.shape[1]):
        for k in np.arange(votemper_NEMO_north.shape[2]):
            for l in np.arange(votemper_NEMO_north.shape[3]):
                if np.isnan(votemper_NEMO_north[i,j,k,l]):
                    votemper_NEMO_north[i,j,k,l] = votemper_NEMO_north[i,j-1,k,l]
                else:
                    continue     

    
for i in np.arange(vosaline_NEMO_north.shape[0]):
    for j in np.arange(vosaline_NEMO_north.shape[1]):
        for k in np.arange(vosaline_NEMO_north.shape[2]):
            for l in np.arange(vosaline_NEMO_north.shape[3]):
                if np.isnan(vosaline_NEMO_north[i,j,k,l]):
                    vosaline_NEMO_north[i,j,k,l] = vosaline_NEMO_north[i,j,k,l-1]
                else:
                    continue
                    
for i in np.arange(votemper_NEMO_north.shape[0]):
    for j in np.arange(votemper_NEMO_north.shape[1]):
        for k in np.arange(votemper_NEMO_north.shape[2]):
            for l in np.arange(votemper_NEMO_north.shape[3]):
                if np.isnan(votemper_NEMO_north[i,j,k,l]):
                    votemper_NEMO_north[i,j,k,l] = votemper_NEMO_north[i,j,k,l-1]
                else:
                    continue


votemper_NEMO_north_twoless = votemper_NEMO_north[:,:,:,1:-1]                    
vosaline_NEMO_north_twoless = vosaline_NEMO_north[:,:,:,1:-1]
U_NEMO_north_twoless = U_NEMO_north[:,:,:,1:-1]
V_NEMO_north_twoless = V_NEMO_north[:,:,:,1:-1]


Nt = votemper_NEMO_north_twoless.shape[0]
Nz = votemper_NEMO_north_twoless.shape[1]
Ny = votemper_NEMO_north_twoless.shape[2]
Nx = votemper_NEMO_north_twoless.shape[3]



vozocrtx_flipped_north = np.flip(U_NEMO_north_twoless, axis=2)
vomecrty_flipped_north = np.flip(U_NEMO_north_twoless, axis=2)
votemper_flipped_north = np.flip(votemper_NEMO_north_twoless, axis=2)
vosaline_flipped_north = np.flip(vosaline_NEMO_north_twoless, axis=2)


vozocrtx_3d_north = np.reshape(a=vozocrtx_flipped_north, newshape= (Nt,Nz,1,Nx*Ny))
vomecrty_3d_north = np.reshape(a=vomecrty_flipped_north, newshape= (Nt,Nz,1,Nx*Ny))
votemper_3d_north = np.reshape(a=votemper_flipped_north, newshape= (Nt,Nz,1,Nx*Ny))
vosaline_3d_north = np.reshape(a=vosaline_flipped_north, newshape= (Nt,Nz,1,Nx*Ny))

print("Now writing the 3D North bdy conditions")


bdy_file = nc.Dataset(path_bdy + '3d_north_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', vozocrtx_3d_north.shape[3]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('deptht', vozocrtx_3d_north.shape[1]);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'x indices along north boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'one y index across all of north boundary';

deptht = bdy_file.createVariable('deptht', 'float32', ('deptht',), zlib=True);
deptht.units = 'm';
deptht.longname = 'Vertical T Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

vozocrtx = bdy_file.createVariable('vozocrtx', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vomecrty = bdy_file.createVariable('vomecrty', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
votemper = bdy_file.createVariable('votemper', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vosaline = bdy_file.createVariable('vosaline', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);

vozocrtx[...] = vozocrtx_3d_north[...];
vomecrty[...] = vomecrty_3d_north[...];
votemper[...] = votemper_3d_north[...];
vosaline[...] = vosaline_3d_north[...];

bdy_file.close()

print("The 3D north boundary conditions are written")


print("Working with the South boundary")


u_3d_bottom_twoless = U_wcvi_bottom
v_3d_bottom_oneless = V_wcvi_bottom
ssh_bottom          = ssh_wcvi_bottom 
votemper_bottom     = votemper_wcvi_bottom
vosaline_bottom     = vosaline_wcvi_bottom

u_2d_bottom_twoless = np.mean(u_3d_bottom_twoless, axis= 1)
v_2d_bottom_oneless = np.mean(v_3d_bottom_oneless, axis=1)

u_2d_bottom = np.empty((votemper_bottom.shape[0],votemper_bottom.shape[2],votemper_bottom.shape[3])); 
u_2d_bottom[:,:,:-2] = u_2d_bottom_twoless;
u_2d_bottom[:,:,-1]  = u_2d_bottom[:,:,-3];
u_2d_bottom[:,:,-2]  = u_2d_bottom[:,:,-3];

u_3d_bottom = np.empty((votemper_bottom.shape[0],votemper_bottom.shape[1],votemper_bottom.shape[2],votemper_bottom.shape[3])); 
u_3d_bottom[:,:,:,:-2] = u_3d_bottom_twoless;
u_3d_bottom[:,:,:,-1]  = u_3d_bottom[:,:,:,-3];
u_3d_bottom[:,:,:,-2]  = u_3d_bottom[:,:,:,-3];


v_2d_bottom = np.empty((votemper_bottom.shape[0],votemper_bottom.shape[2],votemper_bottom.shape[3])); 
v_2d_bottom[:,:,:-1]  = v_2d_bottom_oneless;
v_2d_bottom[:,:,-1]  = v_2d_bottom[:,:,-2];


v_3d_bottom = np.empty((votemper_bottom.shape[0],votemper_bottom.shape[1],votemper_bottom.shape[2],votemper_bottom.shape[3])); 
v_3d_bottom[:,:,:,:-1] = v_3d_bottom_oneless;
v_3d_bottom[:,:,:,-1]  = v_3d_bottom[:,:,:,-2];
        
        
ssh_south_twoless  = ssh_bottom [:,:,1:-1]
u_2d_south_twoless = u_2d_bottom[:,:,1:-1]
v_2d_south_twoless = v_2d_bottom[:,:,1:-1]
                       
Nt = ssh_south_twoless.shape[0]
Ny = ssh_south_twoless.shape[1]
Nx = ssh_south_twoless.shape[2]

sossheig_2d_south = np.reshape(a= ssh_south_twoless, newshape= (Nt,1,Nx*Ny))
vobtcrtx_2d_south = np.reshape(a= u_2d_south_twoless, newshape= (Nt,1,Nx*Ny))
vobtcrty_2d_south = np.reshape(a= v_2d_south_twoless, newshape= (Nt,1,Nx*Ny))
    

print("Writing the 2d south bdy conditions")

bdy_file = nc.Dataset(path + '2d_south_flather_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', sossheig_2d_south.shape[2]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'indices along south boundary';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'a strip of indices across all of south boundary';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

sossheig = bdy_file.createVariable('sossheig', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrtx = bdy_file.createVariable('vobtcrtx', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);
vobtcrty = bdy_file.createVariable('vobtcrty', 'float32', ('time_counter', 'yb', 'xbT'), zlib=True);


sossheig[...] = sossheig_2d_south[...];
vobtcrtx[...] = vobtcrtx_2d_south[...];
vobtcrty[...] = vobtcrty_2d_south[...];

bdy_file.close()

print("The South 2d boundary conditions are written")


print("Working with the 3D NEMO South boundary conditions")

vosaline_JP_level_PSU_south = vosaline_bottom;
votemper_JP_level_PT_south = votemper_bottom;
U_vel_JP_level_south = u_3d_bottom;
V_vel_JP_level_south = v_3d_bottom;


vosaline_JP_level_SA_south = np.empty(vosaline_JP_level_PSU_south.shape);
vosaline_JP_level_RS_south = np.empty(vosaline_JP_level_PSU_south.shape);
votemper_JP_level_CT_south = np.empty(votemper_JP_level_PT_south.shape);

pressure = gsw_calls.generic_gsw_caller('gsw_p_from_z.m', [z, np.mean(lat)]);

print("Converted z to p: first GSW call successful")

for i,j in enumerate(vosaline_JP_level_SA_south[0,:,...]):
    vosaline_JP_level_SA_south[:,i,...] = gsw_calls.generic_gsw_caller(\
                                    'gsw_SA_from_SP', [vosaline_JP_level_PSU_south[:,i,...],pressure[i],lons[i],lats[i]]);
print("Got SA from SP: GSW ran successfully inside loop") 
vosaline_JP_level_RS_south[:] = gsw_calls.generic_gsw_caller('gsw_SR_from_SP', [vosaline_JP_level_PSU_south[:]]);
print("Reference Salinity obtained from PS: one more GSW call left")
votemper_JP_level_CT_south[:] = gsw_calls.generic_gsw_caller('gsw_CT_from_pt', [vosaline_JP_level_SA_south[:], votemper_JP_level_PT_south[:]]);  
print("GSW calls successfull for the south bdy")
                    
south_temp_function = interp1d(depth_JP_T, votemper_JP_level_CT_south, axis=1,\
                              bounds_error=False, fill_value='extrapolate');
south_sal_function = interp1d(depth_JP_T, vosaline_JP_level_RS_south, axis=1,\
                              bounds_error=False, fill_value='extrapolate');
south_U_function = interp1d(depth_JP_T, U_vel_JP_level_south,axis=1,\
                              bounds_error=False, fill_value='extrapolate');
south_V_function = interp1d(depth_JP_T, V_vel_JP_level_south, axis=1,\
                              bounds_error=False, fill_value='extrapolate');


votemper_NEMO_south = np.empty((vosaline_JP_level_SA_south.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_south.shape[2], vosaline_JP_level_SA_south.shape[3]));
vosaline_NEMO_south = np.empty((vosaline_JP_level_SA_south.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_south.shape[2], vosaline_JP_level_SA_south.shape[3]));
U_NEMO_south = np.empty((vosaline_JP_level_SA_south.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_south.shape[2], vosaline_JP_level_SA_south.shape[3]));
V_NEMO_south = np.empty((vosaline_JP_level_SA_south.shape[0], NEMO_depth_T.shape[0], \
                               vosaline_JP_level_SA_south.shape[2], vosaline_JP_level_SA_south.shape[3]));


for indx in np.arange(NEMO_depth_T.shape[0]):
    votemper_NEMO_south[:,indx,...] = south_temp_function(NEMO_depth_T[indx]);
    vosaline_NEMO_south[:,indx,...] = south_sal_function(NEMO_depth_T[indx]);
    U_NEMO_south[:,indx,...]        = south_U_function(NEMO_depth_T[indx]);
    V_NEMO_south[:,indx,...]        = south_V_function(NEMO_depth_T[indx]);

for i in np.arange(vosaline_NEMO_south.shape[0]):
    for j in np.arange(vosaline_NEMO_south.shape[1]):
        for k in np.arange(vosaline_NEMO_south.shape[2]):
            for l in np.arange(vosaline_NEMO_south.shape[3]):
                if np.isnan(vosaline_NEMO_south[i,j,k,l]):
                    vosaline_NEMO_south[i,j,k,l] = vosaline_NEMO_south[i,j-1,k,l]
                else:
                    continue
            

for i in np.arange(votemper_NEMO_south.shape[0]):
    for j in np.arange(votemper_NEMO_south.shape[1]):
        for k in np.arange(votemper_NEMO_south.shape[2]):
            for l in np.arange(votemper_NEMO_south.shape[3]):
                if np.isnan(votemper_NEMO_south[i,j,k,l]):
                    votemper_NEMO_south[i,j,k,l] = votemper_NEMO_south[i,j-1,k,l]
                else:
                    continue  

for i in np.arange(vosaline_NEMO_south.shape[0]):
    for j in np.arange(vosaline_NEMO_south.shape[1]):
        for k in np.arange(vosaline_NEMO_south.shape[2]):
            for l in np.arange(vosaline_NEMO_south.shape[3]):
                if np.isnan(vosaline_NEMO_south[i,j,k,l]):
                    vosaline_NEMO_south[i,j,k,l] = vosaline_NEMO_south[i,j,k,l-1]
                else:
                    continue
                    
                    
for i in np.arange(votemper_NEMO_south.shape[0]):
    for j in np.arange(votemper_NEMO_south.shape[1]):
        for k in np.arange(votemper_NEMO_south.shape[2]):
            for l in np.arange(votemper_NEMO_south.shape[3]):
                if np.isnan(votemper_NEMO_south[i,j,k,l]):
                    votemper_NEMO_south[i,j,k,l] = votemper_NEMO_south[i,j,k,l-1]
                else:
                    continue

        
votemper_NEMO_south_twoless = votemper_NEMO_south[:,:,:,1:-1]                    
vosaline_NEMO_south_twoless = vosaline_NEMO_south[:,:,:,1:-1]
U_NEMO_south_twoless = U_NEMO_south[:,:,:,1:-1]
V_NEMO_south_twoless = V_NEMO_south[:,:,:,1:-1]
   
Nt = U_NEMO_south_twoless.shape[0]
Nz = U_NEMO_south_twoless.shape[1]
Ny = U_NEMO_south_twoless.shape[2]
Nx = U_NEMO_south_twoless.shape[3]

vozocrtx_3d_south = np.reshape(a=U_NEMO_south_twoless, newshape= (Nt,Nz,1,Nx*Ny))
vomecrty_3d_south = np.reshape(a=V_NEMO_south_twoless, newshape= (Nt,Nz,1,Nx*Ny))
votemper_3d_south = np.reshape(a=votemper_NEMO_south_twoless, newshape= (Nt,Nz,1,Nx*Ny))
vosaline_3d_south = np.reshape(a=vosaline_NEMO_south_twoless, newshape= (Nt,Nz,1,Nx*Ny))
        
print("Now writing the 3D South bdy conditions")

bdy_file = nc.Dataset(path_bdy + '3d_south_yBT_looped_twoless_high_res_m04.nc', 'w', zlib=True);

bdy_file.createDimension('xbT', vozocrtx_3d_south.shape[3]);
bdy_file.createDimension('yb', 1);
bdy_file.createDimension('deptht', vozocrtx_3d_south.shape[1]);
bdy_file.createDimension('time_counter', None);

xbT = bdy_file.createVariable('xbT', 'int32', ('xbT',), zlib=True);
xbT.units = 'indices';
xbT.longname = 'x indices along south boundary ordered from outwards to inside (increasing nbr)';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'one y index across all of south boundary';

deptht = bdy_file.createVariable('deptht', 'float32', ('deptht',), zlib=True);
deptht.units = 'm';
deptht.longname = 'Vertical T Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';


vozocrtx = bdy_file.createVariable('vozocrtx', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vomecrty = bdy_file.createVariable('vomecrty', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
votemper = bdy_file.createVariable('votemper', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);
vosaline = bdy_file.createVariable('vosaline', 'float32', ('time_counter', 'deptht', 'yb', 'xbT'), zlib=True);

vozocrtx[...] = vozocrtx_3d_south[...];
vomecrty[...] = vomecrty_3d_south[...];
votemper[...] = votemper_3d_south[...];
vosaline[...] = vosaline_3d_south[...];

bdy_file.close()
    
print("The 3D south bdy conditions are successfully written")    
         
       
print("Reached the end of the code: Thank you")



















