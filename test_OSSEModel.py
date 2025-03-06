# # get SuperDARN data (line-of-sight plasma convection measurements)

# import xarray as xr
# import pandas as pd
# import numpy as np
# import datetime as dt
# import matplotlib.pyplot as plt
# import lompe
# from apexpy import Apex
# from scipy.interpolate import interp1d, griddata
# from lompe.utils.geodesy import geoc2geod
# from ppigrf import igrf_gc
# from polplot import Polarplot
# from lompe import visualization
# from lompe.utils import sunlight
# from lompe.utils.conductance import EUV_conductance
# import cdflib # to read Champ data file
# import warnings
# warnings.filterwarnings('ignore')

# event = '2001-08-17'

# wicfn = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/20010817_wic_image.nc'
# sdarnfn = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/20010817_superdarn_grdmap.h5'

# ds = xr.open_dataset(wicfn)

# EE  = np.array([0.2, 0.5, 1.0, 5.0, 10.0, 25.0])
# E1  = np.array([446, 470, 511, 377, 223, 101])
# fE1 = interp1d(EE, E1, fill_value = (446, 101), bounds_error = False)

# assumed_mean_energy = 2.56 # keV
# ds['eFlux'] = ds['shimage'] / fE1(assumed_mean_energy)
# ds['eFlux'].attrs = {'long_name' : 'Energy flux', 'units' : '$mW/m^2$'}
# ds['eFlux'] = ds['eFlux'].clip(0, None)      # set negatives to zero

# # use Robinson formulae to calculate conductances
# ds['SPedersen'] = 40 * assumed_mean_energy / (16 * assumed_mean_energy**2) * np.sqrt(ds['eFlux'])
# ds['SHall'] = 0.45 * assumed_mean_energy ** 0.85 * ds['SPedersen']

# DT = dt.timedelta(seconds = 60*4) # model a four minute time interval
# date = pd.to_datetime(ds.date.values)

# # model time interval
# t0 = date - DT/2
# t1 = date + DT/2
# # print('Model interval is ' + str(t0) + ' UT to ' + str(t1) + ' UT')

# # grid specs
# L, W, Lres, Wres = 6000.e3, 3000.e3, 75.e3, 75.e3 # dimensions and resolution of grid, in meters
# pos = (0, 90)  # defualt center of cubed sphere grid
# CENTER_IN_SPOT = True # will center grid in designated spot

# # apex object for coordinate systems
# a = Apex(date = date.year)

# if CENTER_IN_SPOT:
#     # find the geographic coordinate of the spot
#     mlat = 78
#     mlt  = 16.5

#     mlon = a.mlt2mlon(mlt, date)
#     glat, glon, error = a.apex2geo(mlat, mlon, 130)
#     # print('grid lat, lon :', glat, glon)
#     pos = (glon, glat)
# p = lompe.cs.CSprojection(pos, 20)

# # image grid
# imgrid = lompe.cs.CSgrid(p, L, W, Lres, Wres, R = (6371.2 + 110)*1e3)

# lon, lat = ds.glon.values.flatten(), ds.glat.values.flatten()
# SH_num = ds['SHall'    ].values.flatten()
# SP_num = ds['SPedersen'].values.flatten()

# iii = imgrid.ingrid(lon, lat, ext_factor = 1.3)

# xi, eta = p.geo2cube(lon[iii], lat[iii])

# # interpolate pixels to the grid:
# SH_grd = griddata(np.vstack((xi, eta)).T, SH_num[iii], np.vstack((imgrid.xi.flatten(), imgrid.eta.flatten())).T).reshape(imgrid.shape)
# SP_grd = griddata(np.vstack((xi, eta)).T, SH_num[iii], np.vstack((imgrid.xi.flatten(), imgrid.eta.flatten())).T).reshape(imgrid.shape)

# # make a function that picks values from the right grid cell
# def get_conductance(lon, lat, time, type = 'Hall'):
#     shape = lon.shape
#     if type == 'Hall':
#         C = SH_grd
#     if type == 'Pedersen':
#         C = SP_grd

#     sza = sunlight.sza(lat.flatten(), lon.flatten(), time)
    
#     i, j = imgrid.bin_index(lon.flatten(), lat.flatten())
#     conductance = np.sqrt(C[i, j].flatten()**2 + EUV_conductance(sza, 100, type[0].lower())**2)
    
#     return conductance.reshape(shape)

# # construct tuple of Hall / Pedersen functions to pass to lompe Emodel object:
# conductances = (lambda lon, lat : get_conductance(lon, lat, t0, type = 'Hall'),
#                 lambda lon, lat : get_conductance(lon, lat, t0, type = 'Pedersen'))

# # create grid object
# grid = lompe.cs.CSgrid(p, L, W, Lres, Wres, R = (6371.2 + 110)*1e3)


# def get_data_subsets(t0, t1):
#     """ return subsets of data loaded above, between t0 and t1 """

    
#     # SuperDARN data:
#     sd = superdarn # the file contains data only from the correct time interval, no time selection needed
#     vlos = sd['v'].values     # this is median value in gridded obs.
#     sd_coords = np.vstack((sd['glon'].values, sd['glat'].values))
#     los  = np.vstack((sd['le'].values, sd['ln'].values))
#     sd_err = sd['std']           # standard deviation as error
    
#     # Make the data objects. The scale keyword determines a weight for the dataset. Increase it to reduce weight
#     #supermag_data  = lompe.Data(smag_B * 1e-9, smag_coords,            datatype = 'ground_mag', scale = 100e-9)
#     #superdarn_data = lompe.Data(vlos         , sd_coords  , LOS = los, datatype = 'convection', scale = 500 )
#     #scale keyword deprecated as of June 2023 in favor of 'error' and 'iweight' (importance weight) keywords
#     superdarn_data = lompe.Data(vlos         , sd_coords  , LOS = los, datatype = 'convection', iweight = 1.0, error = 100 )

#     return superdarn_data

# def superdarn_fix(superdarn):
#     ''' Add geographic components to superDARN data set.'''
    
#     SDmlat  = superdarn['mlat'].values
#     SDmlon  = superdarn['mlon'].values
#     bearing = superdarn['bearing'].values

#     # convert to geographic coordinates
#     SDglat, SDglon, _ = a.apex2geo(SDmlat, SDmlon, 110)
    
#     # line-of-sight vector components, in magnetic AACGM
#     le_m, ln_m = np.sin(bearing * np.pi / 180), np.cos(bearing * np.pi / 180)
    
#     # find bearing angle in geographic
#     f1, f2 = a.basevectors_qd(SDglat, SDglon, 110) # vectors pointing roughly east, north AACGM
    
#     # normalize the northward vector, and define a new eastward vector that is perpendicular:
#     f2 = f2 / np.linalg.norm(f2, axis = 0)
#     f1 = np.cross(np.vstack((f2, np.zeros(f2.shape[1]))).T, np.array([[0, 0, 1]])).T[:2]
    
#     # line of sight vector components in geographic
#     le, ln = f1 * le_m + f2 * ln_m
    
#     superdarn['glat'], superdarn['glon'] = SDglat, SDglon
#     superdarn['le'], superdarn['ln'] = le, ln
    
#     return superdarn

# # load data
# superdarn = pd.read_hdf(sdarnfn)

# # correct superdarn data
# superdarn = superdarn_fix(superdarn)

# # get correct time
# superdarn_data = get_data_subsets(t0, t1)

#%%

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import apexpy
import lompe
import lompe.data
from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *

event = '2014-12-15'
hour = 1
minute = 19
stime = dt.datetime(int(event[0:4]), int(event[5:7]), int(event[8:10]), hour, minute) # the specific time to model
DT = dt.timedelta(seconds = 2*60) # will select data from time +- DT

position = (-98,73) # lon, lat
orientation = -36 #(-0.1, 1) # east, north
L, W, Lres, Wres = 2500e3, 2500e3, 70.e3, 70.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

# plot grid and coastlines
fig, ax = plt.subplots(figsize = (10, 10))
ax.set_axis_off()
for lon, lat in grid.get_grid_boundaries():
    xi, eta = grid.projection.geo2cube(lon, lat)
    ax.plot(xi, eta, color = 'grey', linewidth = .4)

xlim, ylim = ax.get_xlim(), ax.get_ylim()
for cl in grid.projection.get_projected_coastlines():
    ax.plot(cl[0], cl[1], color = 'C0')
    
ax.set_xlim(xlim)
ax.set_ylim(ylim);

# wicfn = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/20010817_wic_image.nc'

tempfile_path = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/' # where .nc SSUSI-files are saved. You can change to fit your system.
cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = tempfile_path, basepath = tempfile_path + '/raw/') #1000

# file names
sdarnfn = tempfile_path + '20141215_superdarn_grdmap.h5'
f17fn = tempfile_path + '20141215_ssies_f17.h5'
f18fn = tempfile_path + '20141215_ssies_f18_hairston.h5'
smagfn = tempfile_path + '20141215_supermag.h5'

# load data
superdarn = pd.read_hdf(sdarnfn)
ssies17 = pd.read_hdf(f17fn)
ssies18 = pd.read_hdf(f18fn)
supermag = pd.read_hdf(smagfn)


def get_data_subsets(t0, t1):
    """ return subsets of data loaded above, between t0 and t1 """
    
    # SuperDARN data:
    #sd = superdarn.loc[t0:t1, :]
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1) & 
                        (superdarn.vlos < 2000)].dropna()
    sd_vlos = sd['vlos'].values
    sd_coords = np.vstack((sd['glon'].values, sd['glat'].values))
    sd_los  = np.vstack((sd['le'].values, sd['ln'].values))
    
    # SSIES (DMSP F17) data:
    f17 = ssies17[t0 - DT : t1 + DT].dropna() # why +- TWO DT??
    v_crosstrack17 = np.abs(f17.hor_ion_v).values
    f17_coords = np.vstack((f17.glon.values, f17.gdlat.values))
    f17_los  = np.vstack((f17['le'].values, f17['ln'].values))
    
    # SSIES (DMSP F18) data:
    f18 = ssies18[t0 - DT : t1 + DT].dropna()
    v_crosstrack18 = np.abs(f18.hor_ion_v).values
    f18_coords = np.vstack((f18.glon.values, f18.glat.values))
    f18_los  = np.vstack((f18['le'].values, f18['ln'].values))
    # add large error for poor F18 measurements
    error = np.zeros(len(f18))
    error[f18.vyqual > 2] = 10000

    # SuperMAG data:
    smag = supermag[supermag.lat <= 90] # select northern hemisphere magnetometers? Necessarry?
    smag = smag[t0 : t1].dropna()
    smag_B = np.vstack((smag.Be.values, smag.Bn.values, smag.Bu.values)) # nT
    smag_coords = np.vstack((smag.lon.values, smag.lat.values))
        
    # Make the data objects. The scale keyword determines a weight for the dataset. Increase it to reduce weight
    # superdarn_data = lompe.Data(sd_vlos        , sd_coords  , LOS = sd_los , datatype = 'convection' , scale = 500)
    # ssies_data1    = lompe.Data(v_crosstrack17 , f17_coords , LOS = f17_los, datatype = 'convection' , scale = 500)
    # ssies_data2    = lompe.Data(v_crosstrack18 , f18_coords , LOS = f18_los, datatype = 'convection' , scale = 500   , error = error)
    # supermag_data  = lompe.Data(smag_B * 1e-9  , smag_coords,                datatype = 'ground_mag' , scale = 100e-9, error = 1e10)
    # note the large error added to SuperMAG data to produce zero weight

    #'scale' keyword deprecated in favor of 'error' and 'iweight' (importance weight) keywords
    superdarn_data = lompe.Data(sd_vlos        , sd_coords  , LOS = sd_los , datatype = 'convection' , iweight = 1.0, error = 50)
    ssies_data1    = lompe.Data(v_crosstrack17 , f17_coords , LOS = f17_los, datatype = 'convection' , iweight = 1.0, error = 50)
    ssies_data2    = lompe.Data(v_crosstrack18 , f18_coords , LOS = f18_los, datatype = 'convection' , iweight = 1.0, error = 50)
    supermag_data  = lompe.Data(smag_B * 1e-9  , smag_coords,                datatype = 'ground_mag' , iweight = 0.0, error = 10e-9)
    # note the iweight=0.0 given to SuperMAG data to produce zero weight 
    
    return(superdarn_data, ssies_data1, ssies_data2, supermag_data)

# get data from specified time interval:
sd_data, ssies_data1, ssies_data2, sm_data = get_data_subsets(stime - DT, stime + DT)

# apex object for plotting in magnetic
a = apexpy.Apex(stime, refh = 110)

# create Emodel object. Pass grid and Hall/Pedersen conductance from SSUSI image
model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

# add data to model
model.add_data(sd_data, ssies_data1, ssies_data2) #, sm_data

#%% 

"""
Here is what the user could do to test things after creating their lompe model
"""

import lompe.data_tools.dataloader as dataloader
import xarray as xr
import copy
from lompe.lompeOSSE.OSSEModel import *
from lompe.utils.time import yearfrac_to_datetime
import secsy as cs
import apexpy
import polplot
from scipy.interpolate import griddata, RectBivariateSpline

# ########
# Lompe object set-up
# ########

# Define epoch used for IGRF dependent calculations and apex object for ... (magnetic coordinate)
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])
apx = apexpy.Apex(time[0].year)

# Define grid
# position = (0,90)
# orientation = 0
# projection = cs.CSprojection(position, orientation)
# # L, W, Lres, Wres = 220000e3,220000e3,100e3,100e3 # 180000e3,180000e3,300e3,300e3
# L, W, Lres, Wres = 20000e3,20000e3,400e3,400e3 # 180000e3,180000e3,300e3,300e3
# RE = 6371.2 # Earth radius in kilometers
# R = RE + 120 # Ionospheric radius in kilometers (distance from center of Earth)
# grid = cs.CSgrid(projection, L, W, Lres, Wres, R=R*1e3) 

# position = (-98,73) # lon, lat
# orientation = 0 #(-0.1, 1) # east, north
# L, W, Lres, Wres = 10500e3, 10500e3, 350.e3, 350.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
# grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

# Get OSSE model (GAMERA datasets and conductances)
osse_model = lompeOSSE(model, Gstep=1, epoch=epoch).osse_model

#%%

# run inversion #FIX REGULARIZATION PARAMETERS
model.run_inversion(l1 = 1, l2 = 10) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction
osse_model.run_inversion(l1 = 1, l2 = 10) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

# Plot Lompe ouput
fig = lompe.model.visualization.lompeplot(model, include_data = True, time = time, apex = apx)
fig = lompe.model.visualization.lompeplot(osse_model, include_data = True, time = time, apex = apx)

fig = lompe.lompeplot(model, include_data = True, time = stime, apex = a, 
                      colorscales = {'fac'        : np.linspace(-0.55, 0.55, 40) * 1e-6 * 2,
                                     'ground_mag' : np.linspace(-380, 380, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 4, 32), # mho
                                     'pedersen'   : np.linspace(0, 4, 32)}, # mho
                        quiverscales = {'ground_mag'       : 50*1e-9, 
                                        'space_mag_fac'    : 100*1e-9, 
                                        'space_mag_full'   : 100*1e-9, 
                                        'electric_current' : 100 * 1e-3})

#######
# Compare with original Gamera potential (interpolated)
#######

mixFiles = '/Users/margot/Downloads/msphere.mix.h5' # Kareem's dataset
hem = 'NORTH' # or 'SOUTH'# initialise for specified hemisphere 
# Gdata = get_Gdata(mixFiles, step=1, hem=hem) 
Gdata = lompeOSSE(model, Gstep=1, epoch=epoch).gamera_data # fix that

# Orignal Gamera potential at glonG, glatG
lon, lat = Gdata['glon'], Gdata['glat']
potG = Gdata['Potential'] # in V

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(lat, lon/15, potG, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('GAMERA potential')
plt.show()

#######
# Reconstruct electric potential using Lompe
#######

# Determine the reconstructed potential at glonG, glatG (original Gamera coordinates)
osse_pot = osse_model.E_pot(lon=lon, lat=lat) * 1e-3 # V

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(lat, lon/15, osse_pot, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('OSSE potential')
plt.show()

# Gamera potential VS lompe reconstructed potential (should be a line)
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(potG.flatten(), osse_pot, alpha=.3, color='grey')
plt.xlabel('Gamera potential')
plt.ylabel('OSSE potential')
plt.show()

# %%

        # if test:

        #     # First plot the electric field vector in its original coordinates
        #     fig,axs = plt.subplots(2,2,figsize=(10,10))
        #     csax0 = cs.CSplot(axs[0][0], self.grid,gridtype='geo')
        #     csax0.add_coastlines(color='grey')
        #     csax0.scatter(glonG[19], glatG[19], s=10, color='red')
        #     csax0.quiver(EeG, EnG, glonG, glatG, color='k')
        #     axs[0][0].set_xlabel('Longitude')
        #     axs[0][0].set_ylabel('Latitude')
        #     axs[0][0].set_title(r"$E_{field}$ in GAMERA spherical coordinates ($E_\phi$, $E_\theta$)")

        #     # Then plot E_xi, E_eta and compare direction and magnitude to E_phi, E_theta
        #     csax1 = cs.CSplot(axs[0][1], self.grid,gridtype='cs')
        #     csax1.add_coastlines(color='grey')
        #     axs[0][1].scatter(xiG[19], etaG[19], s=10, color='red')
        #     axs[0][1].quiver(xiG, etaG, E_xi, E_eta, color='k') # use matplotlib quiver function when it comes to xi and eta coordinates
        #     axs[0][1].set_title(r"$E_{field}$ in GAMERA cube coordinates ($E_\xi$, $E_\eta$)")

        #     # Now plot the electric field vector for the input lon, lat values
        #     csax2 = cs.CSplot(axs[1][0], self.grid,gridtype='cs')
        #     csax2.add_coastlines(color='grey')
        #     # csax2.scatter(xi, eta, s=2, color='red')
        #     axs[1][0].scatter(xi[5], eta[5], s=10, color='green')
        #     axs[1][0].quiver(xi, eta, E_xi_interp, E_eta_interp, color='k') # scale???
        #     axs[1][0].set_title(r"Interpolated $E_{field}$ (cube coord. $E_\xi$, $E_\eta$)")

        #     # Finally, plot the electric field back in a spherical system
        #     csax3 = cs.CSplot(axs[1][1], self.grid,gridtype='geo')
        #     csax3.add_coastlines(color='grey')
        #     csax3.scatter(glon[5], glat[5], s=10, color='green')
        #     csax3.quiver(Ee, En, glon, glat) #, scale=900
        #     axs[1][1].set_xlabel('Longitude')
        #     axs[1][1].set_ylabel('Latitude')
        #     axs[1][1].set_title(r"Interpolated $E_{field}$ (spherical coord. $E_\phi$, $E_\theta$)")
        #     plt.tight_layout()
        #     plt.show()