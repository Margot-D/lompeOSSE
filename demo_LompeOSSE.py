"""
Demo script for using the lompeOSSE module. 

This script demonstrates how to create and compare an OSSE model with a user-defined Lompe model.  
The OSSE model is a copy of the original Lompe model, but its datasets are replaced with  
synthetic data from Gamera simulations, including Gamera-derived conductances. 
All other model properties remain unchanged.


Part 1: Setting up the user model
=================================
- Define the event of interest.  
- Set up a grid.  
- Load datasets of choice.  
- Define a conductance model.  
- Create a Lompe Emodel object and add the selected datasets.

Part 2: Using the lompeOSSE module
==================================
    Example of how to initialize and test the OSSE Lompe model.  

Part 3: Validating the model
============================
    Plot the Gamera electric potential against the electric potential derived from osse_lompe

(?) This demo provides a template for users to explore the capabilities of Lompe in an OSSE framework.

"""

#%% 

# Start by defining event and grid, loading datasets, and creating lompe model. 

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import apexpy
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
L, W, Lres, Wres = 2500e3, 2500e3, 70.e3, 70.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
refh = 120 # reference height in km
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = (6371.2 + refh)*1e3)

# L, W, Lres, Wres = 20000e3,20000e3,400e3,400e3 # 180000e3,180000e3,300e3,300e3
# L, W, Lres, Wres = 10500e3, 10500e3, 350.e3, 350.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)

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
user_model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

# Add data to model
user_model.add_data(sd_data, ssies_data1, ssies_data2) #, sm_data

#%% 

# Use lompeOSSE to replace real observations with Gamera-simulated data in the Lompe model

import sys
sys.path.append('/Users/margot/Docs/Academia/Research/Python/')

import lompeosse
# from lompeosse import create_lompeOSSE
from lompe.utils.time import yearfrac_to_datetime
import apexpy
import secsy as cs

# Define epoch used for IGRF dependent calculations and apex object for ... (magnetic coordinate)
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])
apx = apexpy.Apex(time[0].year)

# Derive OSSE model
lompeosse_obj = lompeosse.osseEmodel(user_model, Gstep=1, mlt_offset=6, epoch=epoch)
osse_model = lompeosse_obj.osse_model
# osse_model, osse_stuff = create_lompeOSSE(model, Gstep=1, mlt_offset=6, epoch=epoch) # CHANGE NAME!

# run inversion #FIX REGULARIZATION PARAMETERS
user_model.run_inversion(l1 = 1, l2 = 10) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction
osse_model.run_inversion(l1 = 1, l2 = 10) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

# Plot Lompe ouput

# fig = lompe.lompeplot(model, include_data = True, time = time, apex = apx)
fig = lompe.lompeplot(user_model, include_data = True, time = time, apex = apx, 
                      colorscales = {'fac'        : np.linspace(-0.55, 0.55, 40) * 1e-6 * 2,
                                     'ground_mag' : np.linspace(-380, 380, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 4, 32), # mho
                                     'pedersen'   : np.linspace(0, 4, 32)}, # mho
                        quiverscales = {'ground_mag'       : 50*1e-9, 
                                        'space_mag_fac'    : 100*1e-9, 
                                        'space_mag_full'   : 100*1e-9, 
                                        'electric_current' : 100 * 1e-3})

# fig = lompe.lompeplot(osse_model, include_data = True, time = time, apex = apx)
fig = lompe.lompeplot(osse_model, include_data = True, time = time, apex = apx, 
                      colorscales = {'fac'        : np.linspace(-2, 2, 40) * 1e-6 * 2,
                                     'ground_mag' : np.linspace(-500, 500, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 20, 32), # mho
                                     'pedersen'   : np.linspace(0, 20, 32)}, # mho
                        quiverscales = {'ground_mag'       : 600*1e-9, 
                                        'space_mag_fac'    : 600*1e-9, 
                                        'space_mag_full'   : 600*1e-9, 
                                        'electric_current' : 1}) # 1000*1e-3

# %%

# Validation: Plot Gamera electric potential VS the electric potential derived from osse_lompe

# Gamera coordinates and electric potential (after interpolation!)
# Gdata = osse_stuff.gamera_data
Gdata = lompeosse_obj.gamera_data
potG = Gdata['Potential'] # in V

# interp_potG = osse_stuff.interp2lompegrid(potG)
interp_potG = lompeosse_obj.interp2lompegrid(potG)

# Reconstructed potential
potOSSE = osse_model.E_pot(lon=grid.lon, lat=grid.lat) * 1e-3 # V
potOSSE = potOSSE.reshape(grid.lon.shape)

# Plot

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# First panel (top-left): Gamera potential
ax1 = fig.add_subplot(gs[0, 0])  
csax1 = cs.CSplot(ax1, grid, gridtype='cs')
csax1.contour(grid.lon, grid.lat, interp_potG)
ax1.set_title("Gamera electric potential")

# Second panel (top-right): LompeOSSE-reconstructed potential
ax2 = fig.add_subplot(gs[0, 1])  
csax2 = cs.CSplot(ax2, grid, gridtype='cs')
csax2.contour(grid.lon, grid.lat, potOSSE)
ax2.set_title("LompeOSSE reconstructed potential")

# Third panel (bottom): Scatter plot Gamera potential VS Lompe-reconstructed potential (should be a line ish)
ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(interp_potG, potOSSE, alpha=.3, color='grey')
ax3.set_xlabel("Gamera Potential")
ax3.set_ylabel("LompeOSSE Potential")
ax3.set_title("Gamera vs LompeOSSE potential")
# plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# %%
