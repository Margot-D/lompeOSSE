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

TODO: CORRECT INFO ABOUT SUPPORTED DATASETS
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as plt
import lompe
import lompe.data
from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *
import os

# Define event
event = '2014-12-15' 
event_date = event[0:4]+event[5:7]+event[8:10] # format YYYYMMDD
hour = 1
minute = 19
stime = dt.datetime(int(event[0:4]), int(event[5:7]), int(event[8:10]), hour, minute) # the specific time to model
DT = dt.timedelta(seconds = 2*60) # will select data from stime +- DT

# Define grid
lonc, latc = -90, 83 # grid will be centered at these lon/lat coordinates
position = (lonc,latc) # center position
orientation = -36 #(-0.1, 1) # east, north

L, W, Lres, Wres = 3000e3, 3000e3, 70.e3, 70.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
L, W, Lres, Wres = 5000.e3, 5000.e3, 70.e3, 70.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)

refh = 120 # reference height in km # TODO useful to keep here?
R = 6371.2 + refh # Inospheric radius in km # TODO useful to keep here?
RG = 6500 # Ionospheric radius in Gamera in km
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = RG*1e3) #R = (R)*1e3

# Plot grid and coastlines
print('User grid and coastlines:')
fig, ax0 = plt.subplots(figsize = (5, 5))
ax0.set_axis_off()
for lon, lat in grid.get_grid_boundaries():
    xi, eta = grid.projection.geo2cube(lon, lat)
    ax0.plot(xi, eta, color = 'grey', linewidth = .4)

xlim, ylim = ax0.get_xlim(), ax0.get_ylim()
for cl in grid.projection.get_projected_coastlines():
    ax0.plot(cl[0], cl[1], color = 'C0')
    
ax0.set_xlim(xlim)
ax0.set_ylim(ylim)
plt.show()


lompe_dir = os.path.dirname(os.path.abspath(lompe.__file__))
data_dir = os.path.join(lompe_dir, '../examples/sample_dataset')
print(data_dir)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Could not find sample_dataset folder at {data_dir}")

# Dictionary of files: key = variable name, value = (filename, description)
files = {
    "sdarnfn": (f"{event_date}_superdarn_grdmap.h5", "SuperDARN (radar)"),
    "smagfn":  (f"{event_date}_supermag.h5", "SuperMAG (ground magnetometers)"),
    "f17fn":   (f"{event_date}_ssies_f17.h5", "DMSP F17 SSIES"),
    "f18fn":   (f"{event_date}_ssies_f18_hairston.h5", "DMSP F18 SSIES (Hairston)"),
    # "iridfn":  (f"{event_date}_iridium.h5", "Iridium"),
    # "ampfn":   (f"{event_date}_ampere.h5", "AMPERE")
}

# Build absolute paths and check files
for var, (filename, desc) in files.items():
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {desc} data file at {path}")
    files[var] = (path, desc)  # overwrite filename with full path

# Load data
superdarn = pd.read_hdf(files["sdarnfn"][0])
supermag  = pd.read_hdf(files["smagfn"][0])
ssies17   = pd.read_hdf(files["f17fn"][0])
ssies18   = pd.read_hdf(files["f18fn"][0])
# iridium   = pd.read_hdf(files["iridfn"][0])
# ampere    = pd.read_hdf(files["ampfn"][0])

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

    # TODO add iridium and ampere here!
        
    # Make the data objects
    superdarn_data = lompe.Data(sd_vlos        , sd_coords  , LOS = sd_los , datatype = 'convection' , iweight = 1.0, error = 50)
    ssies_data1    = lompe.Data(v_crosstrack17 , f17_coords , LOS = f17_los, datatype = 'convection' , iweight = 1.0, error = 50)
    ssies_data2    = lompe.Data(v_crosstrack18 , f18_coords , LOS = f18_los, datatype = 'convection' , iweight = 1.0, error = 50)
    supermag_data  = lompe.Data(smag_B * 1e-9  , smag_coords,                datatype = 'ground_mag' , iweight = 0.0, error = 10e-9)
    # note the iweight=0.0 given to SuperMAG data to produce zero weight 
    
    return(superdarn_data, ssies_data1, ssies_data2, supermag_data)

# Get the data objects for specified time interval
sd_data, ssies_data1, ssies_data2, sm_data = get_data_subsets(stime - DT, stime + DT)

# Define conductance model using SSUSI image
cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = data_dir, basepath = data_dir + '/raw/') #1000

# Create Emodel object
model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

# Add data to model
model.add_data(sd_data, ssies_data1, ssies_data2) #, sm_data
