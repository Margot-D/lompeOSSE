import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import lompe
import apexpy
import os 

import sys, os
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # go one level up from notebook dir
sys.path.append(repo_root)
from lompeosse import LompeOSSE

run_with_real_data = False # set to True to run Lompe inversion on the real data and not only synthetic 

def get_data_subsets(t0, t1):
    """ return subsets of data loaded above, between t0 and t1 """
    
    # Iridium data:
    irid = iridium[(iridium.time >= t0) & (iridium.time <= t1)]
    irid_B = np.vstack((irid.B_e.values, irid.B_n.values, irid.B_r.values))
    irid_coords = np.vstack((irid.lon.values, irid.lat.values, irid.r.values))

    # SuperMAG data:
    smag = supermag.loc[t0:t1, :]
    smag_B = np.vstack((smag.Be.values, smag.Bn.values, smag.Bu.values))
    smag_coords = np.vstack((smag.lon.values, smag.lat.values))
    
    # SuperDARN data:
    sd = superdarn.loc[t0:t1, :]
    vlos = sd['vlos'].values
    sd_coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))

    
    # Make the data objects. The scale keyword determines a weight for the dataset. Increase it to reduce weight
    # 'scale' keyword deprecated as of June 2023 in favor of 'error' and 'iweight' keywords
    #iridium_data   = lompe.Data(irid_B * 1e-9, irid_coords,            datatype = 'space_mag_fac', scale = 200e-9)
    #supermag_data  = lompe.Data(smag_B * 1e-9, smag_coords,            datatype = 'ground_mag'   , scale = 100e-9)
    #superdarn_data = lompe.Data(vlos         , sd_coords  , LOS = los, datatype = 'convection'   , scale = 500 )
    
    iridium_data   = lompe.Data(irid_B * 1e-9, irid_coords,            datatype = 'space_mag_fac', iweight = 1.0, error = 30e-9)
    supermag_data  = lompe.Data(smag_B * 1e-9, smag_coords,            datatype = 'ground_mag'   , iweight = 0.4, error = 10e-9)
    superdarn_data = lompe.Data(vlos         , sd_coords  , LOS = los, datatype = 'convection'   , iweight = 1.0, error = 50 )

    return(iridium_data, supermag_data, superdarn_data)

datapath = os.path.join(os.path.dirname(os.path.abspath(lompe.__file__)), '../examples/sample_dataset/')
files = os.listdir(datapath)
if '20120405_supermag.h5' not in files:
    raise Exception('could not find supermag datafile in lompe install path. Try installing lompe with -e or edit this script to use a different dataset')

event = '2012-04-05'
# file names and location
supermagfn = datapath + '/20120405_supermag.h5'
superdarnfn = datapath + '/20120405_superdarn_grdmap.h5' 
iridiumfn = datapath + '/20120405_iridium.h5'


# cubed sphere grid parameters:
position = (-90, 68)  # lon, lat for grid center
orientation = 0       # angle of grid x axis - anti-clockwise from east direction
L, W = 7000e3, 3800e3 # extents [m] of grid
dL, dW = 100e3, 100e3 # spatial resolution [m] of grid 

# create grid object
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, dL, dW, R = 6481.2e3)


# load data
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn) 
iridium   = pd.read_hdf(iridiumfn)


T0 = dt.datetime(2012, 4, 5, 5, 12)
DT = dt.timedelta(seconds = 60 * 4) # length of time interval

# apex object for plotting in magnetic
apex = apexpy.Apex(T0, refh = 110)

# making conductance tuples
Kp = 4 # this is the input to the Hardy model
SH = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, T0, 'hall')
SP = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, T0, 'pedersen')

# Create Emodel object. Pass grid and Hall/Pedersen conductance functions
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))

# add datasets to model
iridium_data, supermag_data, superdarn_data = get_data_subsets(T0 - DT/2, T0 + DT/2) # data from new model time
model.add_data(iridium_data, supermag_data, superdarn_data)

if run_with_real_data:
    # Run inversion. l1 and l2 are regularization parameters that control the damping of 
    # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction
    model.run_inversion(l1 = 1, l2 = 10)

    # finally, plot (plot is saved as specified path):
    fig = lompe.lompeplot(model, include_data = True, time = T0, apex = apex, savekw = {'fname':'./north_america.pdf'})

osse_obj = LompeOSSE(model, nstep = 0, mlt_off = mlt_offset, epoch = T0.year)
osse_obj.osse_model.run_inversion(l1 = 1e-2, l2 = 1e-2)
lompe.lompeplot(osse_obj.osse_model, include_data = True, time = T0, apex = apex)



