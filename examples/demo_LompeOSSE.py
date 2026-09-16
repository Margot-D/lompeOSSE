"""
Built on the Lompe technique, the LompeOSSE Python module reconstructs ionospheric 
electrodynamics based on synthetic (Gamera model) data.
This script serves as an example template for configuring a LompeOSSE run. 
To explore the capabilities of Lompe in specific OSSE frameworks, see the three Jupyter notebooks.


Stage 1: Configure model input and generate the baseline electric field model
=================================
The user provides the standard Lompe inputs: event date, local grid, conductance model, 
and observational datasets. These are generic Lompe settings and are not implemented by 
LompeOSSE, but they are required to generate the baseline electric field model.

Stage 2: Extract Gamera simulation data
=================================
The GameraData module extracts the user-selected simulation snapshot (time step), 
which is then used as input for the LompeOSSE calculations.

Stage 3: Derive the OSSE model (lompeosse.py)
==================================
Generate the synthetic OSSE model by replacing the real observational datasets 
in the baseline electric field model with synthetic data extracted from 
the selected Gamera snapshot. 
The result is a fully synthetic model that preserves the user-defined grid and configuration.

Stage 4: Run the inversion and vizualize the results
==================================
Apply the Lompe technique to the synthetic OSSE model to reconstruct the 
electrodynamic quantities. This step produces figures showing the LompeOSSE outputs, 
such as electric potential, electric fields, and ionospheric currents.

Stage 5: Validate the OSSE setup
==================================
Validate the output of a LompeOSSE run by comparing the OSSE-reconstructed fields 
with the Gamera “ground truth” using a scatter plot.


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import apexpy
import lompe
# from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *

from lompeosse import LompeOSSE, GameraData, plot_gamera_lompe_style, validate
# from lompeosse import LompeOSSE, GameraData, validate

from pathlib import Path

#%% Lompe electric field model 

# -------------------------
# Define event and time interval
# -------------------------

event = '2014-12-15'
event = '2012-04-05'
hour = 1
minute = 19

DT = dt.timedelta(minutes=2)

# Derived parameters 
event_date = event.replace('-', '') # format YYYYMMDD
time = pd.to_datetime(event)
stime = dt.datetime(time.year, time.month, time.day, hour, minute) # specific time to model
apx = apexpy.Apex(time.year) # apex object for magnetic coordinate calculations

# -------------------------
# Define analysis grid
# -------------------------

# Grid center coordinates and orientation
lonc, latc = 90, 83 # center coordinates of the grid
orientation = 0 #(-0.1, 1) # east, north

# Grid dimensions and resolution
# L and Lres are along the orientation vector, W, Wres are perpendicular
# L, W, Lres, Wres = 3000e3, 3000e3, 70.e3, 70.e3 # example of fine, small grid
# L, W, Lres, Wres = 15000e3, 15000e3, 150e3, 150e3 # example of larger grid
L, W, Lres, Wres = 3000e3, 3000e3, 200e3, 200e3 # in [m]

# Build grid
RG = 6500*1e3 # Ionospheric radius used in Gamera output in [m]
grid = lompe.cs.CSgrid(lompe.cs.CSprojection((lonc, latc), orientation), L, W, Lres, Wres, R = RG) # L,W,Lres,Wres and R in the same unit 

if grid.lat.min() < 50:
    print(np.min(grid.lat))
    print('Your grid should not extend below 50 degrees mlat')

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

# -------------------------
# Select datasets
# -------------------------

files = {"superdarn": (f"{event_date}_superdarn_grdmap.h5", "SuperDARN (radar)"),
        "supermag":  (f"{event_date}_supermag.h5", "SuperMAG (ground magnetometers)"),
        # "ssies17":   (f"{event_date}_ssies_f17.h5", "DMSP F17 SSIES (ion drift and plasma parameters)"),
        # "ssies18":   (f"{event_date}_ssies_f18_hairston.h5", "DMSP F18 SSIES (ion drift and plasma parameters)"),
        "ampere":  (f"{event_date}_iridium.h5", "Iridium AMPERE (space magnetometers FAC data) "),
        }

print("Selected datasets:")
for key, (filename, description) in files.items():
    print(f"  • {description}")

# Check if datafiles exist
example_dir = Path(__file__).resolve().parent
data_dir = example_dir / "sample_datasets"

for var, (filename, desc) in files.items():
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Could not find {desc} data file at {path}")

    files[var] = (path, desc)

# -------------------------
# Prepare Lompe data objects
# -------------------------

# Load selected datasets 
datasets = {}
for key, (path, desc) in files.items():
    df = pd.read_hdf(path)
    datasets[key] = df

# Prepare data objects for Lompe
def get_data_subsets(datasets, t0, t1):
    """
    Return subsets of datasets between t0 and t1 as Lompe Data objects.
    """

    lompe_data_dict = {}

    for key, df in datasets.items():

        if key in ['superdarn']:
            sub = df.loc[(df.index >= t0) & (df.index <= t1) & (df.vlos < 2000)].dropna()
            sub = sub[np.abs(sub.glat) > 50]
            values = sub['vlos'].values
            coords = np.vstack((sub['glon'].values, sub['glat'].values))
            LOS = np.vstack((sub['le'].values, sub['ln'].values))     
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
            sub = sub[np.abs(sub.lat) > 50]
            values = np.vstack((sub.Be.values, sub.Bn.values, sub.Bu.values)) # nT
            coords = np.vstack((sub.lon.values, sub.lat.values))            
            LOS = None
            datatype = 'ground_mag'
            iweight = 1.0
            error = 10e-9

        elif key in ['ampere']:
            sub = df[(df.time >= t0) & (df.time <= t1)]
            sub = sub[np.abs(sub.lat) > 50]

            if not sub.empty:
                values = np.vstack((sub.B_e.values, sub.B_n.values, sub.B_r.values))
                coords = np.vstack((sub.lon.values, sub.lat.values, sub.r.values))
            else:
                values = np.empty((3, 0))
                coords = np.empty((2, 0))

            LOS = None
            datatype = 'space_mag_fac'
            iweight = 1.0
            error = 30e-9

        # Create Lompe Data object
        lompe_data_dict[key] = lompe.Data(values, coords, LOS=LOS, datatype=datatype, iweight=iweight, error=error)

    print("\n Lompe data objects generated")

    return lompe_data_dict

lompe_datasets = get_data_subsets(datasets, stime - DT/2, stime + DT/2)

# -------------------------
# Electric field model
# -------------------------

# Create Emodel object (here with a toy conductance model that gives one for every grid.lon, grid.lat) 
model = lompe.Emodel(grid, (lambda x, y: np.ones_like(x*y), lambda x, y: np.ones_like(x*y)))

for data_obj in lompe_datasets.values():
    model.add_data(data_obj)

#%% LompeOSSE

# -------------------------
# Extract Gamera simulation data
# -------------------------

# Available timesteps: #0 #2 #3 #12 #13 #14 #16 #19 #20 #21 #22 (Use find-simulation-snapshot.py to inspect the different snapshots)
hemisphere = 'NORTH' if latc > 0 else 'SOUTH'
gamera = GameraData(stime, timestep = 0, hemisphere = hemisphere)

# -------------------------
# Derive synthetic model
# -------------------------

# TODO do i need to run the inversion before feeding lompeosse with "model"???
osse_object = LompeOSSE(model, gamera)
osse_Emodel = osse_object.make_OSSE_model(time_offset = 0) #TODO what is the point of adding time offset here rather thsn in the class directly? 

# -------------------------
# Run inversion on OSSE model and plot output using Lompe 
# -------------------------

osse_Emodel.run_inversion(l1 = 1, l2 = 10) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

ntime = osse_object.timestamp + dt.timedelta(hours=osse_object.time_offset)

suptitle = f"LompeOSSE-reconstructed electrodynamics"
fig = lompe.lompeplot(osse_Emodel, include_data = True, time = ntime, apex = apx, 
                      colorscales = {'fac'        : np.linspace(-2, 2, 40) * 1e-6 * 2,
                                     'ground_mag' : np.linspace(-500, 500, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 20, 32), # mho
                                     'pedersen'   : np.linspace(0, 20, 32)}, # mho
                        quiverscales = {'ground_mag'       : 600*1e-9, 
                                        'space_mag_fac'    : 600*1e-9, 
                                        'space_mag_full'   : 600*1e-9, 
                                        'electric_current' : 1}, # 1000*1e-3 #TODO ok?
                        suptitle=suptitle) 
plt.show()

# -------------------------
# Gamera plot
# -------------------------

fig =  plot_gamera_lompe_style(osse_Emodel, gamera, ntime)
plt.show()

# -------------------------
# Validation metrics
# -------------------------

fig, metrics = validate(osse_Emodel, gamera, ntime, primary='potential', overlay='fac')
plt.show()
# print(metrics)
