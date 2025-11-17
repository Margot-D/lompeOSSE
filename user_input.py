"""
Stage 1: Configure model input
=================================

This script serves as an example template for configuring a LompeOSSE run. 
All parameters defined here — including the event date and time, Lompe grid setup, 
conductance model, selected datasets, and Gamera simulation snapshot — can be modified 
by the user to suit their specific study or scenario.


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

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib as plt
import apexpy
import h5py
import lompe
from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *
import os

#############
# Event date and time
#############

# Define event
event = '2014-12-15'
hour = 1
minute = 19

# Derived parameters (no user action required)
def compute_event_params(event, hour, minute):
    event_date = event.replace('-', '') # format YYYYMMDD
    time = pd.to_datetime(event)
    stime = dt.datetime(time.year, time.month, time.day, hour, minute) # specific time to model
    DT = dt.timedelta(minutes=2) # time interval
    apx = apexpy.Apex(time.year) # apex object for magnetic coordinate calculations

    return event_date, time, stime, DT, apx

event_date, time, stime, DT, apx = compute_event_params(event, hour, minute)

#############
# Grid
#############

# Define grid center and orientation
lonc, latc = -90, 83 # center coordinates of the grid
position = (lonc,latc)
orientation = -36 #(-0.1, 1) # east, north

# Define grid dimensions and resolution un meters (L and Lres are along the orientation vector, W, Wres are perpendicular)
# L, W, Lres, Wres = 3000e3, 3000e3, 70.e3, 70.e3 # example of fine, small grid
L, W, Lres, Wres = 15000e3, 15000e3, 150e3, 150e3 # example of larger grid

# Grid (no user action required)
RG = 6500 # (km) Ionospheric radius used in Gamera output
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = RG * 1e3) # L,W,Lres,Wres and R in the same unit 

# Plot grid and coastlines (optional)
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

#############
# Conductance model (to be used to build the baseline electric field model)
#############

# Build absolute paths 
lompe_dir = os.path.dirname(os.path.abspath(lompe.__file__))
data_dir = os.path.join(lompe_dir, '../examples/sample_dataset')
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Could not find sample_dataset folder at {data_dir}")

# Define conductance model using SSUSI image
cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = data_dir, basepath = data_dir + '/raw/') #1000

#############
# Datasets (to be used to build the baseline electric field model)
#############

# Dictionnary of datasets (TODO specify supported datasets)
files = {
    "superdarn": (f"{event_date}_superdarn_grdmap.h5", "SuperDARN (radar)"),
    "supermag":  (f"{event_date}_supermag.h5", "SuperMAG (ground magnetometers)"),
    "ssies17":   (f"{event_date}_ssies_f17.h5", "DMSP F17 SSIES"),
    "ssies18":   (f"{event_date}_ssies_f18_hairston.h5", "DMSP F18 SSIES (Hairston)"),
    # "iridfn":  (f"{event_date}_iridium.h5", "Iridium"),
    # "ampfn":   (f"{event_date}_ampere.h5", "AMPERE")
}

print("Selected datasets:")
for key, (filename, description) in files.items():
    print(f"  • {description}")

# Check if user datafiles exist
for var, (filename, desc) in files.items():
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {desc} data file at {path}")
    files[var] = (path, desc)  # overwrite filename with full path

# Load selected datasets 
datasets = {}
for key, (path, desc) in files.items():
    df = pd.read_hdf(path)
    datasets[key] = df

# conductance model #TODO remove???
toymodel = lompe.Emodel(grid, (cmod.hall, cmod.pedersen)) #TODO what was this for?

# Prepare datasets and return Lompe Data objects 
def get_data_subsets(datasets, t0, t1):
    """
    Return subsets of datasets between t0 and t1 as Lompe Data objects.

    Args:
        datasets (dict): dictionary of loaded datasets
        t0, t1 (pd.Timestamp): time range

    Returns:
        dict: dictionary of Lompe Data objects keyed by dataset name
    """
    lompe_data_dict = {}
    print("Generating Lompe Data objects...")

    for key, df in datasets.items():

        # Superdarn
        if key in ['superdarn']:
            # sub = df.loc[(df.index >= t0) & (df.index <= t1) & (df.vlos < 2000)].dropna()
            # values = np.vstack((sub.grid_J.lon.flatten(), sub.grid_J.lat.flatten()))
            # coords = np.vstack((sub.grid_J.lon.flatten(), sub.grid_J.lat.flatten()))
            # LOS = np.vstack((sub.grid_J.lon.flatten(), sub.grid_J.lat.flatten()))
            # test: 
            values = np.vstack((toymodel.grid_J.lon.flatten(), toymodel.grid_J.lat.flatten()))
            coords = np.vstack((toymodel.grid_J.lon.flatten(), toymodel.grid_J.lat.flatten()))
            LOS = np.vstack((toymodel.grid_J.lon.flatten(), toymodel.grid_J.lat.flatten()))
            
            datatype = 'convection'
            iweight = 1.0
            error = 50

        elif key in ['ssies17']:
            sub = df.loc[t0 - DT : t1 + DT].dropna()
            values = np.abs(sub.hor_ion_v).values
            coords = np.vstack((sub.glon.values, sub.gdlat.values)) 
            LOS = np.vstack((sub['le'].values, sub['ln'].values))
            datatype = 'convection'
            iweight = 1.0
            error = 50

        elif key in ['ssies18']:
            sub = df.loc[t0 - DT : t1 + DT].dropna()
            values = np.abs(sub.hor_ion_v).values
            coords = np.vstack((sub.glon.values, sub.glat.values)) #glat or gdlat??
            LOS = np.vstack((sub['le'].values, sub['ln'].values))
            datatype = 'convection'
            iweight = 1.0
            error = 50

            # Add large error for F18 poor measurements
            if key == 'ssies18' and 'vyqual' in sub.columns:
                error_array = np.zeros(len(sub))
                error_array[sub.vyqual > 2] = 10000
                error = error_array

        elif key in ['supermag']:
            sub = df[df.lat <= 90].loc[t0:t1].dropna()  # northern hemisphere
            # values = np.vstack((sub.grid_E.lon.flatten(), sub.grid_E.lat.flatten())) * 1e-9
            # coords = np.vstack((sub.grid_E.lon.flatten(), sub.grid_E.lat.flatten()))
            # test:
            values = np.vstack((toymodel.grid_E.lon.flatten(), toymodel.grid_E.lat.flatten())) * 1e-9
            coords = np.vstack((toymodel.grid_E.lon.flatten(), toymodel.grid_E.lat.flatten()))
            
            LOS = None
            datatype = 'ground_mag'
            iweight = 0.0
            error = 10e-9

        # TODO add iridium and ampere here!

        # Create Lompe Data object
        lompe_data_dict[key] = lompe.Data(values, coords, LOS=LOS, datatype=datatype, iweight=iweight, error=error)

    return lompe_data_dict

lompe_datasets = get_data_subsets(datasets, stime - DT, stime + DT)
print("Lompe Data objects ready.")

#############
# Gamera simulation snapshot (to be used for generating synthetic data)
#############

path = os.path.abspath(os.path.dirname(__file__))
datapath = os.path.join(path, 'data/Gamera_data.h5') 

if not os.path.exists(datapath):
    raise FileNotFoundError(
        f"Required file not found: {datapath}\n"
        "Please download it (https://zenodo.org/records/16882035) and place it in the 'data' folder."
    )

with h5py.File(datapath, "r") as Gdata:
    print("Available time steps in Gamera dataset:")
    step_keys = [k for k in Gdata.keys() if k.startswith("Step#")]
    step_keys = sorted(step_keys, key=lambda k: int(k.split('#')[1]))    
    for key in step_keys:
        print("  •", key)

# Select time step of interest for your OSSE (use find-Gamera-snapshot.py to inspect available snapshots)
Gstep = 0 # e.g., if Gstep = 0, the selected time step is Step#0

# MLT offset: rotate the Gamera snapshot in magnetic local time (hours)
# This allows exploring multiple OSSE configurations from the same Gamera snapshot
mlt_offset = 6

print(f"Selected snapshot: Step#{Gstep} with {mlt_offset} hours MLT offset")
