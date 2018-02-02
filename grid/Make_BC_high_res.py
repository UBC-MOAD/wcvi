import numpy as np
import numpy.ma as ma
import netCDF4 as nc
from   salishsea_tools import viz_tools, geo_tools,nc_tools
from   scipy.interpolate import griddata, interp1d
import xarray as xr
from   grid_alignment import calculate_initial_compass_bearing as cibc



gridT = xr.open_dataset('/ocean/ssahu/JP_BC/cat_42_days_T.nc')

mask = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/grid/meshmask_files/JP_mesh_mask.nc')

tmask_JP = mask.variables['tmask'][0,:,55:400,446:701]
umask_JP = mask.variables['umask'][0,:,55:400,446:701]
vmask_JP = mask.variables['vmask'][0,:,55:400,446:701]


ssh_unfiltered = np.array(gridT['sossheig'][1:,55:400,446:701])
votemper_unfiltered = np.array(gridT['votemper'][1:,:,55:400,446:701])
vosaline_unfiltered = np.array(gridT['vosaline'][1:,:,55:400,446:701])


grid_JP = xr.open_dataset('/ocean/ssahu/JP_BC/coordinates_JP.nc')

glamt_bc_JP = np.array(grid_JP.variables['glamt'][55:400,446:701]); gphit_bc_JP = np.array(grid_JP.variables['gphit'][55:400,446:701])

glamu = np.array(grid_JP.variables['glamu'][55:400,446:701]); gphiu = np.array(grid_JP.variables['gphiu'][55:400,446:701])
glamv = np.array(grid_JP.variables['glamv'][55:400,446:701]); gphiv = np.array(grid_JP.variables['gphiv'][55:400,446:701])



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

N = 10

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
    ssh_wcvi_left[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_left, method= 'cubic'), glamt_wcvi_bc_left.shape)
    ssh_wcvi_right[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_right, method= 'cubic'), glamt_wcvi_bc_right.shape)
    ssh_wcvi_top[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_top, method= 'cubic'), glamt_wcvi_bc_top.shape)
    ssh_wcvi_bottom[p,...] = np.reshape(griddata(points, np.array(ssh[p,...]).flatten(), xi_bottom, method= 'cubic'), glamt_wcvi_bc_bottom.shape)
    for i in np.arange(votemper.shape[1]):
        votemper_wcvi_left[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_left, method= 'cubic'), glamt_wcvi_bc_left.shape)
        votemper_wcvi_right[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_right, method= 'cubic'), glamt_wcvi_bc_right.shape)
        vosaline_wcvi_left[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_left, method= 'cubic'), glamt_wcvi_bc_left.shape)
        vosaline_wcvi_right[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_right, method= 'cubic'), glamt_wcvi_bc_right.shape)
        votemper_wcvi_top[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_top, method= 'cubic'), glamt_wcvi_bc_top.shape)
        votemper_wcvi_bottom[p,i,...] = np.reshape(griddata(points, votemper[p,i,...].flatten(), xi_bottom, method= 'cubic'), glamt_wcvi_bc_bottom.shape)
        vosaline_wcvi_top[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_top, method= 'cubic'), glamt_wcvi_bc_top.shape)
        vosaline_wcvi_bottom[p,i,...] = np.reshape(griddata(points, vosaline[p,i,...].flatten(), xi_bottom, method= 'cubic'), glamt_wcvi_bc_bottom.shape)

print("The ssh, temperature and Salinity data has been interpolated to WCVI horizontal grids successfully")


gridU = xr.open_dataset('/home/ssahu/saurav/JP_BC/cat_43_U.nc')
gridV = xr.open_dataset('/home/ssahu/saurav/JP_BC/cat_43_V.nc')


nav_lon_U = np.array(gridU['nav_lon'][55:400,446:701])
nav_lat_U = np.array(gridU['nav_lat'][55:400,446:701])

nav_lon_V = np.array(gridV['nav_lon'][55:400,446:701])
nav_lat_V = np.array(gridV['nav_lat'][55:400,446:701]) 

U_vel_BC_unfiltered = np.array(gridU['vozocrtx'][1:,:,55:400,446:701])
V_vel_BC_unfiltered = np.array(gridV['vomecrty'][1:,:,55:400,446:701])


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

for i in np.arange(U_vel_BC,shape[0]):
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
                                                xi, method= 'cubic'), glamt_wcvi.shape)
       v_unrotated_wcvi_t[p,i,...] = np.reshape(griddata(points, v_unrotated[p,i,...].flatten(), \
                                                xi, method= 'cubic'), glamt_wcvi.shape)


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

#ssh_left_switched  = np.swapaxes(ssh_left, 1, 2)
#u_2d_left_switched = np.swapaxes(u_2d_left, 1, 2)
#v_2d_left_switched = np.swapaxes(v_2d_left, 1, 2)

U_vel_JP_level_left        = np.swapaxes(u_3d_left, 3, 2)
V_vel_JP_level_left        = np.swapaxes(v_3d_left, 3, 2)
votemper_JP_level_PT_left  = np.swapaxes(votemper_wcvi_left, 3, 2)
vosaline_JP_level_PSU_left = np.swapaxes(vosaline_wcvi_left, 3, 2)


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

for i in np.arange(vosaline_JP_level_SA_left.shape[1]):
   vosaline_JP_level_SA_left[:,i,...] = gsw_calls.generic_gsw_caller('gsw_SA_from_SP', [vosaline_JP_level_PSU_left[:,i,...],pressure[i],lons[i],lats[i]])

vosaline_JP_level_RS_left[:] = gsw_calls.generic_gsw_caller('gsw_SR_from_SP', [vosaline_JP_level_PSU_left[:]])
votemper_JP_level_CT_left[:] = gsw_calls.generic_gsw_caller('gsw_CT_from_pt', [vosaline_JP_level_SA_left[:], votemper_JP_level_PT_left[:]])

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


#Now let us write the 3d boundary condition for the left boundary

bdy_file = nc.Dataset('/ocean/ssahu/CANYONS/bdy_files/3d_NEMO_west_m04_high_res.nc', 'w', zlib=True);


bdy_file.createDimension('xb', U_NEMO_left.shape[3]);
bdy_file.createDimension('yb', U_NEMO_left.shape[2]);
bdy_file.createDimension('deptht', U_NEMO_left.shape[1]);
bdy_file.createDimension('time_counter', None);


xb = bdy_file.createVariable('xb', 'int32', ('xb',), zlib=True);
xb.units = 'indices';
xb.longname = 'x indices along left boundary';

yb = bdy_file.createVariable('yb', 'int32', ('yb',), zlib=True);
yb.units = 'indices';
yb.longname = 'a strip of y indices across all of left boundary';

deptht = bdy_file.createVariable('deptht', 'float32', ('deptht',), zlib=True);
deptht.units = 'm';
deptht.longname = 'Vertical T Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';

vozocrtx = bdy_file.createVariable('vozocrtx', 'float32', ('time_counter', 'deptht', 'yb', 'xb'), zlib=True);
vomecrty = bdy_file.createVariable('vomecrty', 'float32', ('time_counter', 'deptht', 'yb', 'xb'), zlib=True);
votemper = bdy_file.createVariable('votemper', 'float32', ('time_counter', 'deptht', 'yb', 'xb'), zlib=True);
vosaline = bdy_file.createVariable('vosaline', 'float32', ('time_counter', 'deptht', 'yb', 'xb'), zlib=True);

vozocrtx[:] = U_NEMO_left[:];
vomecrty[:] = V_NEMO_left[:];
votemper[:] = votemper_NEMO_left[:];
vosaline[:] = vosaline_NEMO_left[:];
deptht[:] = NEMO_depth_T[:];

#vozocrtx[:] = vozocrtx[:,:,::-1,:];### This is done because NEMO reads the file the other way around
#vomecrty[:] = vomecrty[:,:,::-1,:];
#votemper[:] = votemper[:,:,::-1,:];
#vosaline[:] = vosaline[:,:,::-1,:];


bdy_file.close()

print("Reached the end of the code: Thank you")



















