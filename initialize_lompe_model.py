"""
Demo script for defining a Lompe electric field model. 

This script demonstrates how to construct a Lompe model, 
which is the first step required before performing an OSSE 
(Observation System Simulation Experiment) with the lompeOSSE module. 


Part 1: Setting up the user model
=================================
- Define the event of interest  
- Set up a regional grid  
- Define a conductance model  
- Create a lompe.Emodel object (electric field model)
- Load and prepare datasets of choice
- Add selected datasets to Emodel (to be used as input to the inversion)

Suported datasets include:
 - Magnetic field perturbations on ground
 - Magnetic field perturbations in space associated with field-aligned currents
 - Magnetic field perturbations in space associated with both field-aligned currents 
 and horizontal divergence-free currents below the satellite
 - Ionospheric convection velocity (perpendicular to the magnetic field and 
 mapped to the ionospheric radius)
 - Ionospheric convection electric field (perpendicular to B and 
 mapped to the ionospheric radius)
 - Field-aligned electric current density (in A/m^2)

TODO: CHECK IF INFO IS CORRECT
"""

import numpy as np
import pandas as pd
import datetime as dt
import lompe
import lompe.data
from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *

# Define event
event = '2014-12-15'
hour = 1
minute = 19
stime = dt.datetime(int(event[0:4]), int(event[5:7]), int(event[8:10]), hour, minute) # the specific time to model
DT = dt.timedelta(seconds = 2*60) # will select data from time +- DT

# Define grid
position = (-98,73) # lon, lat
orientation = -36 #(-0.1, 1) # east, north
# L, W, Lres, Wres = 2500e3, 2500e3, 70.e3, 70.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
L, W, Lres, Wres = 5000.e3, 5000.e3, 70.e3, 70.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
refh = 120 # reference height in km
R = 6371.2 + refh # Inospheric radius in km
RG = 6500 # Ionospheric radius in Gamera in km
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = RG*1e3) #R = (R)*1e3

# # plot grid and coastlines
# fig, ax0 = plt.subplots(figsize = (10, 10))
# ax0.set_axis_off()
# for lon, lat in grid.get_grid_boundaries():
#     xi, eta = grid.projection.geo2cube(lon, lat)
#     ax0.plot(xi, eta, color = 'grey', linewidth = .4)

# xlim, ylim = ax0.get_xlim(), ax0.get_ylim()
# for cl in grid.projection.get_projected_coastlines():
#     ax0.plot(cl[0], cl[1], color = 'C0')
    
# ax0.set_xlim(xlim)
# ax0.set_ylim(ylim)

# Define conductance model using SSUSI image
tempfile_path = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/' # where .nc SSUSI-files are saved. You can change to fit your system.

cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = tempfile_path, basepath = tempfile_path + '/raw/') #1000

# File names
sdarnfn = tempfile_path + '20141215_superdarn_grdmap.h5'
f17fn = tempfile_path + '20141215_ssies_f17.h5'
f18fn = tempfile_path + '20141215_ssies_f18_hairston.h5'
smagfn = tempfile_path + '20141215_supermag.h5'
# TODO: !! add more options? at the moment it's only convection and ground mag data!

# Load data
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
        
    # Make the data objects
    superdarn_data = lompe.Data(sd_vlos        , sd_coords  , LOS = sd_los , datatype = 'convection' , iweight = 1.0, error = 50)
    ssies_data1    = lompe.Data(v_crosstrack17 , f17_coords , LOS = f17_los, datatype = 'convection' , iweight = 1.0, error = 50)
    ssies_data2    = lompe.Data(v_crosstrack18 , f18_coords , LOS = f18_los, datatype = 'convection' , iweight = 1.0, error = 50)
    supermag_data  = lompe.Data(smag_B * 1e-9  , smag_coords,                datatype = 'ground_mag' , iweight = 0.0, error = 10e-9)
    # note the iweight=0.0 given to SuperMAG data to produce zero weight 
    
    return(superdarn_data, ssies_data1, ssies_data2, supermag_data)

# Get the data objects for specified time interval
sd_data, ssies_data1, ssies_data2, sm_data = get_data_subsets(stime - DT, stime + DT)

# Create Emodel object
model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

# Add data to model
model.add_data(sd_data, ssies_data1, ssies_data2) #, sm_data
