"""
This script serves as an example template for configuring a LompeOSSE run. 


The LompeOSSE Python module enables the automatic acquisition of synthetic data 
from Gamera simulation given an input regional grid, which are used to replace 
the input datasets and conductances of a user-defined electric field model, 
while preserving its original spatial and temporal configuration. Built on the 
Lompe technique, LompeOSSE reconstructs ionospheric electrodynamics based on synthetic 
(Gamera model) data and provides quantitative metrics to evaluate the accuracy of 
the Lompe output against the ground truth from the simulated data.

Stage 1: Configure model input
=================================
The user provides two types of inputs:
- the standard Lompe inputs: event date, local grid, conductance model, and observational datasets. 
These are generic Lompe settings and are not implemented by LompeOSSE, but they are required to 
generate the baseline electric field model;
- the LompeOSSE-specific inputs, which are handled by the LompeOSSE module. 
These include selecting the Gamera simulation snapshot (time step) to generate synthetic 
observations, as well as an optional magnetic local time (MLT) offset that allows exploration 
of multiple configurations from a single snapshot.

All parameters defined here can be modified by the user to suit their specific study or scenario.

Regarding the observational datasets, suported datasets include:
 - Magnetic field perturbations on ground
 - Magnetic field perturbations in space associated with field-aligned currents
 - Magnetic field perturbations in space associated with both field-aligned currents 
 and horizontal divergence-free currents below the satellite
 - Ionospheric convection velocity (perpendicular to the magnetic field and 
 mapped to the ionospheric radius)
 - Ionospheric convection electric field 
 TODO OK?

[Feature to be released soon]  
Datasets for a given date and grid can be collected through the Swarm Data Fusion toolbox (SwarmDF). 
TODO add link to github repo when SwarmDF is published

Stage 2: Generate the electric field model
=================================
Construct the baseline electric field model used in LompeOSSE, using 
all user-defined settings from user_input.py. The baseline model reflects 
realistic observational data and will later be replaced by synthetic Gamera data.

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


This demo provides an example of how to initialize and run the LompeOSSE module. 
To explore the capabilities of Lompe in specific OSSE frameworks, see the example folder with 3 (?) different OSSEs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import secsy as cs
import datetime as dt
import apexpy
import lompe
# from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *
from lompeosse import LompeOSSE
from lompeosse import Gamera_output

#%% Stage 1:

# User input

#############
# Event date and time
#############

# Define event
event = '2014-12-15'
event = '2012-04-05'
hour = 1
minute = 19

# Derived parameters 
event_date = event.replace('-', '') # format YYYYMMDD
time = pd.to_datetime(event)
stime = dt.datetime(time.year, time.month, time.day, hour, minute) # specific time to model
DT = dt.timedelta(minutes=2) # time interval
apx = apexpy.Apex(time.year) # apex object for magnetic coordinate calculations

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
hemisphere = 'NORTH' if latc > 0 else 'SOUTH'
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

# Define conductance model using SSUSI image (or use a toy model when create the Emodel object -- see demo script)
#cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
#              param = 'lbhs', tempfile_path = data_dir, basepath = data_dir + '/raw/') #1000

#############
# Datasets (to be used to build the baseline electric field model)
#############

# Dictionnary of datasets
files = {
    "superdarn": (f"{event_date}_superdarn_grdmap.h5", "SuperDARN (radar)"),
    "supermag":  (f"{event_date}_supermag.h5", "SuperMAG (ground magnetometers)"),
    # "ssies17":   (f"{event_date}_ssies_f17.h5", "DMSP F17 SSIES (ion drift and plasma parameters)"),
    # "ssies18":   (f"{event_date}_ssies_f18_hairston.h5", "DMSP F18 SSIES (ion drift and plasma parameters)"),
    "ampere":  (f"{event_date}_iridium.h5", "Iridium AMPERE (space magnetometers FAC data) "),
}

print("Selected datasets:")
for key, (filename, description) in files.items():
    print(f"  • {description}")

# Check if datafiles exist
lompe_dir = os.path.dirname(os.path.abspath(lompe.__file__))
data_dir = os.path.join(lompe_dir, '../examples/sample_dataset')

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

# Prepare datasets and return Lompe data objects 
def get_data_subsets(datasets, t0, t1):
    """
    Return subsets of datasets between t0 and t1 as Lompe Data objects.
    """

    lompe_data_dict = {}
    print("Generating Lompe data objects")

    for key, df in datasets.items():

        # Superdarn
        if key in ['superdarn']:
            sub = df.loc[(df.index >= t0) & (df.index <= t1) & (df.vlos < 2000)].dropna()
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
            values = np.vstack((sub.Be.values, sub.Bn.values, sub.Bu.values)) # nT
            coords = np.vstack((sub.lon.values, sub.lat.values))            
            LOS = None
            datatype = 'ground_mag'
            iweight = 1.0
            error = 10e-9

        elif key in ['ampere']:
            sub = df[(df.time >= t0) & (df.time <= t1)]

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

    return lompe_data_dict

lompe_datasets = get_data_subsets(datasets, stime - DT/2, stime + DT/2)

#############
# Gamera simulation snapshot (to be used for generating synthetic data)
#############

# Available snapshots: #0 #2 #3 #12 #13 #14 #16 #19 #20 #21 #22
gamera_output = Gamera_output(stime, timestep = 0, mlt_offset = 6, hemisphere = hemisphere)
gamera_data = gamera_output.gamera_data

#%% Stage 2: 

# Create Emodel object (with a toy conductance model)
model = lompe.Emodel(grid, (lambda x, y: np.ones_like(x*y), lambda x, y: np.ones_like(x*y)))

# Add data to model
for data_obj in lompe_datasets.values():
    model.add_data(data_obj)

#%% Stage 3: 

# Derive synthetic model
lompeosse_obj = LompeOSSE(model, gamera_output, mlt_offset = 6)
osse_model = lompeosse_obj.synthetic_model

#%% Stage 4: 

# Run inversion and show output
osse_model.run_inversion(l1 = 1, l2 = 10) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

# fig = lompe.lompeplot(osse_model, include_data = True, time = time, apex = apx)
fig = lompe.lompeplot(osse_model, include_data = True, time = stime, apex = apx, 
                      colorscales = {'fac'        : np.linspace(-2, 2, 40) * 1e-6 * 2,
                                     'ground_mag' : np.linspace(-500, 500, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 20, 32), # mho
                                     'pedersen'   : np.linspace(0, 20, 32)}, # mho
                        quiverscales = {'ground_mag'       : 600*1e-9, 
                                        'space_mag_fac'    : 600*1e-9, 
                                        'space_mag_full'   : 600*1e-9, 
                                        'electric_current' : 1}) # 1000*1e-3
plt.show()

#%% Stage 5: 
# Validate synthetic model

# Load Gamera data (in Gamera grid)
potG = gamera_data['Potential']
facG = gamera_data['Field-aligned current']

# Interpolate to Lompe grid
interp_potG = gamera_output.interp_to_usergrid(grid, potG)
interp_facG = gamera_output.interp_to_usergrid(grid, facG)

# LompeOSSE-reconstruted quantities
potOSSE = osse_model.E_pot(lon=grid.lon, lat=grid.lat) * 1e-3 # V
potOSSE = potOSSE.reshape(grid.lon.shape)

facOSSE = osse_model.FAC(lon=grid.lon, lat=grid.lat)
facOSSE = facOSSE.reshape(grid.lon.shape)


# Plot

fac_levels = np.linspace(-1.95, 1.95, 40) * 1e-6 * 2

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# Top-left: Gamera quantities
ax1 = fig.add_subplot(gs[0, 0])  
csax1 = cs.CSplot(ax1, grid, gridtype='cs')
csax1.contour(grid.lon, grid.lat, interp_potG, colors='k')
csax1.contourf(grid.lon, grid.lat, interp_facG*(-1), cmap='bwr', levels=fac_levels*1e6)
ax1.set_title("Gamera electric potential (black) \n and field-aligned currents (color)")

# Top-right: LompeOSSE-reconstructed quantities
ax2 = fig.add_subplot(gs[0, 1])  
csax2 = cs.CSplot(ax2, grid, gridtype='cs')
csax2.contour(grid.lon, grid.lat, potOSSE, colors='k')
csax2.contourf(grid.lon, grid.lat, facOSSE, cmap='bwr', levels=fac_levels)
ax2.set_title("LompeOSSE reconstructed potential (black) \n and field-aligned currents (color)")

# Bottom: scatter
ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(interp_potG.flatten(), potOSSE.flatten(), alpha=.3, color='grey')
# ax3.scatter(interp_potG, potOSSE, alpha=.3, color='grey')
ax3.set_xlabel("Gamera Potential")
ax3.set_ylabel("LompeOSSE Potential")
ax3.set_title("Gamera vs LompeOSSE electric potential")

plt.tight_layout()
plt.show()

# Compute correlation coefficient
corr_coef = np.corrcoef(interp_potG.flatten(), potOSSE.flatten())[0, 1]
print(f"Correlation coefficient (Gamera vs LompeOSSE potential): \n {corr_coef:.3f}")

# %%
