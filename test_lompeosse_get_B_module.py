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
from magnetic_field_utils import get_B


import kaipy.remix.remix as remix
import os
import h5py
import matplotlib.pyplot as plt

# Define epoch and initialize Apex object for magnetic coordinate calculations 
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])
apx = apexpy.Apex(time[0].year)

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
L, W, Lres, Wres = 15000.e3, 15000.e3, 150.e3, 150.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)

grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = (6500)*1e3)

# Define conductance model using SSUSI image
# tempfile_path = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/' # where .nc SSUSI-files are saved. You can change to fit your system.
lompe_dir = os.path.dirname(os.path.abspath(lompe.__file__))
tempfile_path = os.path.join(lompe_dir, '../examples/sample_dataset/')

cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = tempfile_path, basepath = tempfile_path + '/raw/') #1000

RE = 6371.2 # Earth radius in km

# RE= 6371.2e3
# r = r/RE #trying to find how to make it correspond with X in gamera data

nstep = 0 # time step in Gamera simulation

# Create Emodel object
test_model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

glat, glon = test_model.grid_E.lat.flatten(), test_model.grid_E.lon.flatten()
coords = np.vstack((glon, glat))
mlat,mlon = apx.geo2apex(coords[1], coords[0], RE-RE) #lat, lon, height of the data points
theta = 90 - mlat
phi = mlon+(6*15)
refB = get_B(RE*1e3, theta, phi, nstep, no_df_current=False) 

# in magnetic coordinates
Br_getB     = refB[0].flatten()
Btheta_getB = refB[1].flatten()
Bphi_getB   = refB[2].flatten()

# in geographic coordinates
f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = apx.basevectors_apex(coords[1], coords[0], height=RE-RE, coords = 'geo')
B_east_getB, B_north_getB = Bphi_getB*f1 - Btheta_getB*f2
B_up_getB = Br_getB

Benu = np.empty(refB.shape)
Benu[0] = B_east_getB #east 
Benu[1] = B_north_getB #north
Benu[2] = B_up_getB #up

synth_data = lompe.Data(Benu * 1e-9, coords, datatype = 'ground_mag', iweight = 1, error = 1e-9)
# TODO does not work with B in T... 

# Add data to model
test_model.add_data(synth_data)
# test_model.add_data(synth_data, sd_data, ssies_data1, ssies_data2)

# Run inversion
test_model.run_inversion(l1 = 1, l2 = 1) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

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

# mixFiles = '/Users/margot/Docs/Academia/Research/Python/lompe_osse/Gamera_data.h5'
path = os.path.abspath(os.path.dirname(__file__))
mixFiles = os.path.join(path, 'data/Gamera_data.h5') 
print(mixFiles)

nstep = 0

# use the remix portion of the kaipy package to create the ionospheric (mix) object "data"
data = remix.remix(mixFiles,nstep)
data.init_vars('NORTH') 

plt.figure()
data.plot('current')
plt.show()
# %%



