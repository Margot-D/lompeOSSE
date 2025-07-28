import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lompe
import lompe.data
from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *
from lompe.utils.time import yearfrac_to_datetime
import apexpy
# from lompeosse import get_B
from magnetic_field import get_B


import kaipy.remix.remix as remix
import os
import h5py
import matplotlib.pyplot as plt

# Define event
event = '2014-12-15'
hour = 1
minute = 19
stime = dt.datetime(int(event[0:4]), int(event[5:7]), int(event[8:10]), hour, minute) # the specific time to model
DT = dt.timedelta(seconds = 2*60) # will select data from time +- DT

# Define grid
position = (0,90) # lon, lat
orientation = 0 #(-0.1, 1) # east, north
L, W, Lres, Wres = 25000.e3, 25000.e3, 200.e3, 200.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = (6500)*1e3)

# Define conductance model using SSUSI image
tempfile_path = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/' # where .nc SSUSI-files are saved. You can change to fit your system.

cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = tempfile_path, basepath = tempfile_path + '/raw/') #1000

RI = 6500e3 #m
r = RI + 50e3 #m

# RE= 6371.2e3
# r = r/RE #trying to find how to make it correspond with X in gamera data

nstep = 0 # time step in Gamera simulation

# Create Emodel object
test_model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

# Extract grid coordinates
lat, lon = test_model.grid_E.lat.flatten(), test_model.grid_E.lon.flatten()
coords = np.vstack((lon, lat))

#######
# import initialize_lompe_model
# from lompeosse import LompeOSSE

# from scipy.interpolate import griddata, RectBivariateSpline

# epoch = 2015. # decimal year
# time = yearfrac_to_datetime([epoch])
# apx = apexpy.Apex(time[0].year)
# mlt_offset = 6
# hemisphere = 'NORTH' if initialize_lompe_model.latc > 0 else 'SOUTH'
# grid = initialize_lompe_model.grid
# model = initialize_lompe_model.model
# lompeosse_obj = LompeOSSE(model, nstep=nstep, hem=hemisphere, mlt_off=mlt_offset, epoch=epoch)
# data = lompeosse_obj.gamera_data
# r = data['r']
# th = data['theta']
# ph = data['phi']
# # r, th, ph = r.flatten(), th.flatten(), ph.flatten()

# def interp2lompegrid(var):
#     glonG, glatG = data['glon'], data['glat']

#     # Identify Gamera grid points that fall inside the Lompe cubed sphere grid
#     iii = grid.ingrid(glonG, glatG, ext_factor = 1.5)

#     # Convert valid Gamera (glon, glat) coordinates to Lompe's cubed sphere coordinates (xi, eta)
#     xiG, etaG = grid.projection.geo2cube(glonG[iii],glatG[iii]) 

#     # Extract the xi, eta coordinates of the Lompe grid
#     xi, eta = grid.xi, grid.eta

#     # Interpolate Gamera variable values to the Lompe grid
#     varinterp = griddata((xiG,etaG), var[iii], (xi.flatten(), eta.flatten()))

#     # Reshape the interpolated data to match the original 2D grid structure
#     varinterp = varinterp.reshape(grid.shape)

#     return varinterp

# rinterp, thinterp, phinterp = interp2lompegrid(r), interp2lompegrid(th), interp2lompegrid(ph)

# newcoords = np.vstack((model.grid_J.lon.flatten(), model.grid_J.lat.flatten())) # should be grid_E though...

# Bs = get_B(rinterp.flatten(), thinterp.flatten(), phinterp.flatten(), RI, nstep) # Br, Btheta, Bphi

#######

# Compute magnetic field (TODO in tesla?) at these coordinates
Bs = get_B(r, 90 - lat, lon, RI, nstep) # Br, Btheta, Bphi

# Lompe requires east, north, up components
Benu = np.empty(Bs.shape)
Benu[0] = Bs[2] #east 
Benu[1] = -Bs[1] # north
Benu[2] = Bs[0] # up

# synth_data = lompe.Data(Bs * 1e-9, coords, datatype = 'space_mag_full', iweight = 1, error = 10e-9)
synth_data = lompe.Data(Benu * 1e-9, coords, datatype = 'space_mag_full', iweight = 1, error = 1e-9)
# TODO does not work with B in T... 

#########

# # convection data 
# sdarnfn = tempfile_path + '20141215_superdarn_grdmap.h5'
# f17fn = tempfile_path + '20141215_ssies_f17.h5'
# f18fn = tempfile_path + '20141215_ssies_f18_hairston.h5'

# superdarn = pd.read_hdf(sdarnfn)
# ssies17 = pd.read_hdf(f17fn)
# ssies18 = pd.read_hdf(f18fn)

# def get_data_subsets(t0, t1):
#     """ return subsets of data loaded above, between t0 and t1 """
    
#     # SuperDARN data:
#     #sd = superdarn.loc[t0:t1, :]
#     sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1) & 
#                         (superdarn.vlos < 2000)].dropna()
#     sd_vlos = sd['vlos'].values
#     sd_coords = np.vstack((sd['glon'].values, sd['glat'].values))
#     sd_los  = np.vstack((sd['le'].values, sd['ln'].values))
    
#     # SSIES (DMSP F17) data:
#     f17 = ssies17[t0 - DT : t1 + DT].dropna() # why +- TWO DT??
#     v_crosstrack17 = np.abs(f17.hor_ion_v).values
#     f17_coords = np.vstack((f17.glon.values, f17.gdlat.values))
#     f17_los  = np.vstack((f17['le'].values, f17['ln'].values))
    
#     # SSIES (DMSP F18) data:
#     f18 = ssies18[t0 - DT : t1 + DT].dropna()
#     v_crosstrack18 = np.abs(f18.hor_ion_v).values
#     f18_coords = np.vstack((f18.glon.values, f18.glat.values))
#     f18_los  = np.vstack((f18['le'].values, f18['ln'].values))
#     # add large error for poor F18 measurements
#     error = np.zeros(len(f18))
#     error[f18.vyqual > 2] = 10000

#     # Make the data objects
#     superdarn_data = lompe.Data(sd_vlos        , sd_coords  , LOS = sd_los , datatype = 'convection' , iweight = 1.0, error = 50)
#     ssies_data1    = lompe.Data(v_crosstrack17 , f17_coords , LOS = f17_los, datatype = 'convection' , iweight = 1.0, error = 50)
#     ssies_data2    = lompe.Data(v_crosstrack18 , f18_coords , LOS = f18_los, datatype = 'convection' , iweight = 1.0, error = 50)
#     # note the iweight=0.0 given to SuperMAG data to produce zero weight 
    
#     return(superdarn_data, ssies_data1, ssies_data2)

# # Get the data objects for specified time interval
# sd_data, ssies_data1, ssies_data2 = get_data_subsets(stime - DT, stime + DT)

#########


# Add data to model
test_model.add_data(synth_data)
# test_model.add_data(synth_data, sd_data, ssies_data1, ssies_data2)

# Run inversion
test_model.run_inversion(l1 = 1, l2 = 1) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

# Define epoch and initialize Apex object for magnetic coordinate calculations 
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])
apx = apexpy.Apex(time[0].year)

# fig = lompe.lompeplot(test_model, include_data = True, time = time, apex = apx)
fig = lompe.lompeplot(test_model, include_data = True, time = time, apex = apx, 
                      colorscales = {'fac'        : np.linspace(-2, 2, 40) * 1e-6 /20,
                                     'ground_mag' : np.linspace(-500, 500, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 20, 32), # mho
                                     'pedersen'   : np.linspace(0, 20, 32)}, # mho
                        quiverscales = {'ground_mag'       : 600*1e-9, 
                                        'space_mag_fac'    : 600*1e-9, 
                                        'space_mag_full'   : 200*1e-9, 
                                        'electric_current' : 1}) # 1000*1e-3

#%% Compare Lompe FACS with current figure from Remix module

mixFiles = '/Users/margot/Docs/Academia/Research/Python/lompe_osse/Gamera_data.h5'

nstep = 0

# use the remix portion of the kaipy package to create the ionospheric (mix) object "data"
data = remix.remix(mixFiles,nstep)
data.init_vars('NORTH') 

plt.figure()
data.plot('current')
plt.show()
# %%
