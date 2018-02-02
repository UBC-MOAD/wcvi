import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.interpolate import griddata, interp1d
from salishsea_tools import (nc_tools, gsw_calls,viz_tools)
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

fname = '/home/ssahu/saurav/JP_BC/cat_42_days_T.nc'

gridT = xr.open_dataset(fname)

mask = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/grid/meshmask_files/JP_mesh_mask.nc')

tmask_JP = mask.variables['tmask'][0,:,55:420,446:730]

votemper_JP = np.array(gridT['votemper'][1,:,55:420,446:730]) #Our 1st day of NEMO run (set in namelist and atmos files from that day)  is second day of data which starts from beginning of month
votemper_JP[...] = np.ma.masked_array(votemper_JP[...], mask = tmask_JP[...])
votemper_JP[votemper_JP == 0] =['Nan']

vosaline_JP = np.array(gridT['vosaline'][1,:,55:420,446:730])
vosaline_JP[...] = np.ma.masked_array(vosaline_JP[...], mask = tmask_JP[...])
vosaline_JP[vosaline_JP == 0] = ['Nan']

glamt_bc_JP = np.array(gridT['nav_lon'][55:420,446:730])
gphit_bc_JP = np.array(gridT['nav_lat'][55:420,446:730])
deptht_JP = np.array(gridT['deptht'][:])

print("Removing Nan values from JP's data, since we replaced the zero masks with Nans")


for i in np.arange(1,votemper_JP.shape[0]):
    for p in np.arange(votemper_JP.shape[1]):
        for l in np.arange(votemper_JP.shape[2]):
            if np.isnan(votemper_JP[i,p,l]):
                votemper_JP[i,p,l] = votemper_JP[i-1,p,l]
            else:
                continue


for i in np.arange(1,vosaline_JP.shape[0]):
    for p in np.arange(vosaline_JP.shape[1]):
        for l in np.arange(vosaline_JP.shape[2]):
            if np.isnan(vosaline_JP[i,p,l]):
                vosaline_JP[i,p,l] = vosaline_JP[i-1,p,l]
            else:
                continue
                
                
for i in np.arange(votemper_JP.shape[0]):
    for p in np.arange(votemper_JP.shape[1]):
        for l in np.arange(votemper_JP.shape[2]):
            if np.isnan(votemper_JP[i,p,l]):
                votemper_JP[i,p,l] = votemper_JP[i,p,l-1]
            else:
                continue


for i in np.arange(vosaline_JP.shape[0]):
    for p in np.arange(vosaline_JP.shape[1]):
        for l in np.arange(vosaline_JP.shape[2]):
            if np.isnan(vosaline_JP[i,p,l]):
                vosaline_JP[i,p,l] = vosaline_JP[i,p,l-1]
            else:
                continue
                
for i in np.arange(vosaline_JP.shape[1]):
    for j in np.arange(vosaline_JP.shape[2]):
        if np.isnan(vosaline_JP[0,i,j]):
            vosaline_JP[0,i,j] = vosaline_JP[1,i,j]
        else:
            continue
            
            
for i in np.arange(votemper_JP.shape[1]):
    for j in np.arange(votemper_JP.shape[2]):
        if np.isnan(votemper_JP[0,i,j]):
            votemper_JP[0,i,j] = votemper_JP[1,i,j]
        else:
            continue

fname_wcvi = '/ocean/ssahu/CANYONS/wcvi/grid/coordinates.nc'

with nc.Dataset(fname_wcvi, 'r') as coord:
    gphit_wcvi = coord.variables['gphit'][0,...];
    glamt_wcvi =  coord.variables['glamt'][0,...];

X = glamt_bc_JP.flatten();

Y = gphit_bc_JP.flatten();

points = (X[:],Y[:]);

xi = (glamt_wcvi.flatten(), gphit_wcvi.flatten());

votemper_ic = np.empty((votemper_JP.shape[0], glamt_wcvi.shape[0], glamt_wcvi.shape[1]));
vosaline_ic = np.empty((vosaline_JP.shape[0], glamt_wcvi.shape[0], glamt_wcvi.shape[1]));

for i in np.arange(votemper_JP.shape[0]):
    votemper_ic[i,...] = np.reshape(griddata(points, votemper_JP[i,...].flatten(), xi, method= 'cubic'), glamt_wcvi.shape)
    vosaline_ic[i,...] = np.reshape(griddata(points, vosaline_JP[i,...].flatten(), xi, method= 'cubic'), glamt_wcvi.shape)



print("Interpolation to WCVI horizontal points successful")

print("Calling GSW tools to convert to Conservative Temperature and Reference Salinity")


lat = np.empty_like(gphit_wcvi)
lon = np.empty_like(gphit_wcvi)
depth = np.empty_like(deptht_JP)

lat[:] = gphit_wcvi[:]
lon[:] = glamt_wcvi[:]

depth[:] = deptht_JP[:]

z = np.multiply(depth[:],-1)

votemper_PT = np.empty_like(votemper_ic)
vosaline_PSU =np.empty_like(vosaline_ic)  

 
votemper_PT[:] = votemper_ic[:] 
vosaline_PSU[:] = vosaline_ic[:]

pressure = np.empty(z.shape)
lats = np.empty(pressure.shape)
lons = np.empty(pressure.shape)
lats[:] = np.mean(lat)
lons[:] = np.mean(lon)


vosaline_SA = np.empty(vosaline_PSU.shape)
vosaline_RS = np.empty(vosaline_PSU.shape)
votemper_CT = np.empty(votemper_PT.shape)

pressure = gsw_calls.generic_gsw_caller('gsw_p_from_z.m', [z, np.mean(lat)])

print("Converted z to p: first GSW call successful")

for i in np.arange(vosaline_SA.shape[0]):
    vosaline_SA[i,...] = gsw_calls.generic_gsw_caller('gsw_SA_from_SP', [vosaline_PSU[i,...],pressure[i],lons[i],lats[i]])

print("Got SA from SP: GSW ran successfully inside loop")

vosaline_RS[:] = gsw_calls.generic_gsw_caller('gsw_SR_from_SP', [vosaline_PSU[:]])

print("Reference Salinity obtained from PS: one more GSW call left")

votemper_CT[:] = gsw_calls.generic_gsw_caller('gsw_CT_from_pt', [vosaline_SA[:], votemper_PT[:]])


print("GSW Calls successfull")

mask_wcvi = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/grid/meshmask_files/mesh_mask_high_res.nc')

NEMO_depth = mask_wcvi.variables['gdept_0'][0,:,0,0]
tmask_WCVI = mask_wcvi.variables['tmask'][:]


salinity_function = interp1d(depth, vosaline_RS, axis = 0, bounds_error=False, fill_value='extrapolate')
temperature_function = interp1d(depth, votemper_CT, axis = 0, bounds_error=False, fill_value='extrapolate')

vosaline_NEMO = np.empty((NEMO_depth.shape[0], vosaline_RS.shape[1], vosaline_RS.shape[2]));
votemper_NEMO = np.empty((NEMO_depth.shape[0], vosaline_RS.shape[1], vosaline_RS.shape[2]));

for indx in np.arange(NEMO_depth.shape[0]):
    vosaline_NEMO[indx,...] = salinity_function(NEMO_depth[indx]);
    votemper_NEMO[indx,...] = temperature_function(NEMO_depth[indx]);

for i in np.arange(votemper_NEMO.shape[0]):
    for p in np.arange(votemper_NEMO.shape[1]):
        for l in np.arange(votemper_NEMO.shape[2]):
            if np.isnan(votemper_NEMO[i,p,l]):
                votemper_NEMO[i,p,l] = votemper_NEMO[i-1,p,l]
            else:
                continue

                
for i in np.arange(vosaline_NEMO.shape[0]):
    for p in np.arange(vosaline_NEMO.shape[1]):
        for l in np.arange(vosaline_NEMO.shape[2]):
            if np.isnan(vosaline_NEMO[i,p,l]):
                vosaline_NEMO[i,p,l] = vosaline_NEMO[i-1,p,l]
            else:
                continue



print("Vertical Interpolation to WCVI depth levels successful")

print("Now writing into a binary file to be used as IC for NEMO")

file_temp = nc.Dataset('/ocean/ssahu/CANYONS/wcvi/initial_conditions/West_coast_NEMO_IC_high_resolution.nc', 'w', zlib=True)
# dataset attributes
nc_tools.init_dataset_attrs(
    file_temp, 
    title='Temperature and salinity Initial Condition', 
    notebook_name='Making_IC_from_JP', 
    nc_filepath='/ocean/ssahu/CANYONS/wcvi/initial_conditions/West_coast_NEMO_IC_high_resolution.nc',
    comment='Temperature and salinity from JP Model, high_resolution__grid; used at all grid points and interpolated vertically')

file_temp.createDimension('xb', votemper_NEMO.shape[2]);
file_temp.createDimension('yb', votemper_NEMO.shape[1]);
file_temp.createDimension('deptht', votemper_NEMO.shape[0]);
file_temp.createDimension('time_counter', None);


nav_lat = file_temp.createVariable('nav_lat', 'float32', ('yb','xb'));
nav_lat.long_name = 'Latitude';
nav_lat.units = 'degrees_north';


nav_lon = file_temp.createVariable('nav_lon', 'float32', ('yb','xb'));
nav_lon.long_name = 'Longitude';
nav_lon.units = 'degrees_east';


deptht = file_temp.createVariable('deptht', 'float32', ('deptht'));
deptht.long_name = 'Vertical T Levels';
deptht.units = 'm';
deptht.positive = 'down';


time_counter = file_temp.createVariable('time_counter', 'float32', ('time_counter'));
time_counter.units = 's';
time_counter.long_name = 'time';
time_counter.calendar = 'noleap';


votemper = file_temp.createVariable('votemper', 'float32', ('time_counter','deptht','yb','xb'));
votemper.units = 'degC'
votemper.long_name = 'Conservative Temperature (CT)';
votemper.grid = 'WCVI';


vosaline = file_temp.createVariable('vosaline', 'float32', ('time_counter','deptht','yb','xb'));
vosaline.units = 'g/Kg';
vosaline.long_name = 'Reference Salinity (SR)';
vosaline.grid = 'WCVI';




nav_lat[:] = lat[:];
nav_lon[:] = lon[:];
deptht[:] = NEMO_depth[:];
time_counter[0] = 1;
votemper[0,...] = votemper_NEMO[:]
vosaline[0,...]= vosaline_NEMO[:]

file_temp.close()


print("File written: Thank you")



	






