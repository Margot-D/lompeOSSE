"""
Demo script for using the lompeOSSE module. 

This script demonstrates how to create and compare an OSSE model with a user-defined Lompe model.  
The OSSE model is a copy of the original Lompe model, but its datasets are replaced with  
synthetic data from Gamera simulations, including Gamera-derived conductances. 
All other model properties remain unchanged.


Part 1: Setting up the user model (initialize_lompe_model.py)
=================================
- Define the event of interest  
- Set up a regional grid  
- Define a conductance model  
- Create a lompe.Emodel object (electric field model)
- Load and prepare datasets of choice
- Add selected datasets to Emodel (to be used as input to the inversion)

Part 2: Deriving the OSSE model (lompeOSSE module)
==================================
Create synthetic model by replacing real observations with Gamera-simulated data in the Lompe model. 
See details in lompeOSSE.

Part 3: Running the inversion
==================================
Solve for the electrodynamic quantities. 
This step produces model outputs such as electric fields, ionospheric currents, and potential patterns based on Gamera datasets and conductance model.

Part 4: Validating the OSSE model
==================================
Plot the Gamera electric potential against the electric potential derived from the OSSE lompe model.


This demo provides an example of how to initialize and test an OSSE Lompe model, using the lompeOSSE module. 
TODO: To explore the capabilities of Lompe in an OSSE framework, see .... .py.

(? put that in an other script) This demo provides a template for users to explore the capabilities of Lompe in an OSSE framework.

"""
import initialize_lompe_model
from lompeosse import LompeOSSE

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import apexpy
import secsy as cs
import h5py
import os

import lompe
from lompe.utils.time import yearfrac_to_datetime

from magnetic_field_utils import get_B # TODO remove

#%% Initialization
# User input:

# Define epoch and initialize Apex object for magnetic coordinate calculations 
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])
apx = apexpy.Apex(time[0].year)

# Build path to Gamera data file
path = os.path.abspath(os.path.dirname(__file__))
datapath = os.path.join(path, 'data/Gamera_data.h5') 
# print(datapath)

# Check if data file exists
if not os.path.exists(datapath):
    raise FileNotFoundError(
        f"Required file not found: {datapath}\n"
        "Please download it (https://zenodo.org/records/16882035) and place it in the 'data' folder." # TODO add zenodo link
    )

# Open Gamera data file
Gdata = h5py.File(datapath, 'r')

print("Available time steps in Gamera dataset:")
for key in Gdata.keys():
    print(key)

# Select time step of interest from the Gamera simulation
Gstep = 0 # e.g., if Gstep = 0, the selected time step is Step#0

# Oher user parameters
mlt_offset = 0
hemisphere = 'NORTH' if initialize_lompe_model.latc > 0 else 'SOUTH'
# TODO do that inside LompeOSSE?? (here I need to keep it for later in the script, but it is useless to have hemisphere as an input parameter in lompeosse)

#%% Part 1: 
# Load grid and Lompe model defined in user_model_script
grid = initialize_lompe_model.grid
model = initialize_lompe_model.model

#%% Part 2: 
# Derive synthetic model
lompeosse_obj = LompeOSSE(model, nstep=Gstep, mlt_off=mlt_offset, epoch=epoch)
osse_model = lompeosse_obj.osse_model

#%% Part 3: 
# Run inversion and show output
osse_model.run_inversion(l1 = .01, l2 = .01) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction


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
#TODO add datasets to it 
plt.show()

#%% Part 4: 
# Validate synthetic model

# Extract electric potential and field-aligned currents from Gamera dataset
potG = Gdata[f'Step#{Gstep}']['Potential ' +hemisphere][:]
facG = Gdata[f'Step#{Gstep}']['Field-aligned current ' +hemisphere][:]

# Interpolate Gamera quantities to the user Lompe grid
interp_potG = lompeosse_obj.interp2lompegrid(potG)
interp_facG = lompeosse_obj.interp2lompegrid(facG)

# LompeOSSE-reconstructed quantities
potOSSE = osse_model.E_pot(lon=grid.lon, lat=grid.lat) * 1e-3 # V
potOSSE = potOSSE.reshape(grid.lon.shape)

facOSSE = osse_model.FAC(lon=grid.lon, lat=grid.lat)
facOSSE = facOSSE.reshape(grid.lon.shape)

# Plot

fac_levels = np.linspace(-1.95, 1.95, 40) * 1e-6 * 2

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# First panel (top-left): Gamera quantities
ax1 = fig.add_subplot(gs[0, 0])  
csax1 = cs.CSplot(ax1, grid, gridtype='cs')
csax1.contour(grid.lon, grid.lat, interp_potG, colors='k')
csax1.contourf(grid.lon, grid.lat, interp_facG*(-1), cmap='bwr', levels=fac_levels*1e6)
ax1.set_title("Gamera electric potential (black) \n and field-aligned currents (color)")

# Second panel (top-right): LompeOSSE-reconstructed quantities
ax2 = fig.add_subplot(gs[0, 1])  
csax2 = cs.CSplot(ax2, grid, gridtype='cs')
csax2.contour(grid.lon, grid.lat, potOSSE, colors='k')
csax2.contourf(grid.lon, grid.lat, facOSSE, cmap='bwr', levels=fac_levels)
ax2.set_title("LompeOSSE reconstructed potential (black) \n and field-aligned currents (color)")

# Third panel (bottom): Scatter plot Gamera VS Lompe-reconstructed potential (should be a line ish)
ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(interp_potG, potOSSE, alpha=.3, color='grey')
ax3.set_xlabel("Gamera Potential")
ax3.set_ylabel("LompeOSSE Potential")
ax3.set_title("Gamera vs LompeOSSE electric potential")
# plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()


#%% TEST

RE = 6371.2 # Earth radius in km

# reference (from get_B)
lat, lon = osse_model.grid_E.lat.flatten(), osse_model.grid_E.lon.flatten()
coords = np.vstack((lon, lat))
latG,lonG = apx.geo2apex(coords[1], coords[0], RE-RE) #lat, lon, height of the point
theta = 90 - latG
phi = lonG
refB = get_B(RE*1e3, theta, phi, Gstep, no_df_current=False) 

Br_ref     = refB[0].flatten()
Btheta_ref = refB[1].flatten()
Bphi_ref   = refB[2].flatten()

f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = apx.basevectors_apex(coords[1], coords[0], height=RE-RE, coords = 'geo')

B_geo_east, B_geo_north = Bphi_ref*f1 - Btheta_ref*f2
B_geo_up = Br_ref

B_east_ref = B_geo_east
B_north_ref = B_geo_north
B_up_ref = B_geo_up

# predictions
lompeB = osse_model.B_ground(lon=lon, lat=lat) #nT

B_east_pred     = lompeB[0].flatten()*1e9
B_north_pred = lompeB[1].flatten()*1e9
B_up_pred   = lompeB[2].flatten()*1e9

# components = [
#     ("Beast", B_east_pred, B_east_ref),
#     ("Bnorth", B_north_pred, B_north_ref),
#     ("Bup", B_up_pred, B_up_ref),
# ]

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for ax, (label, pred, ref) in zip(axes, components):
#     ax.scatter(ref, pred, s=10, alpha=0.6)
#     # ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], 'k--', lw=1)  # 1:1 line
#     ax.set_xlabel(f"{label} (get_B)")
#     ax.set_ylabel(f"{label} (Lompe-predicted)")
#     ax.set_title(label)

# plt.tight_layout()
# plt.show()

# Plot

# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# # Be ref
# ax2 = fig.add_subplot(gs[0, 0])  
# csax2 = cs.CSplot(ax2, osse_model.grid_E, gridtype='cs')
# csax2.contour(lonG.reshape(osse_model.grid_E.lon.shape), latG.reshape(osse_model.grid_E.lon.shape), B_east_ref.reshape(osse_model.grid_E.lon.shape), cmap = plt.cm.bwr)

# # Be predicted
# ax1 = fig.add_subplot(gs[0, 1])  
# csax1 = cs.CSplot(ax1, osse_model.grid_E, gridtype='cs')
# csax1.contour(lonG.reshape(osse_model.grid_E.lon.shape), latG.reshape(osse_model.grid_E.lon.shape), B_east_pred.reshape(osse_model.grid_E.lon.shape), cmap = plt.cm.bwr)

# # Scatter plot
# ax3 = fig.add_subplot(gs[1, :])
# ax3.scatter(B_east_ref, B_east_pred, alpha=.3, color='grey')
# ax3.set_xlabel("from get_B")
# ax3.set_ylabel("from LompeOSSE")
# # ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], 'k--', lw=1)  # 1:1 line

# plt.tight_layout()
# plt.show()


# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# ax1 = fig.add_subplot(gs[0, 1])  
# csax1 = cs.CSplot(ax1, osse_model.grid_E, gridtype='cs')
# csax1.contour(osse_model.grid_E.lon, osse_model.grid_E.lat, B_north_pred.reshape(osse_model.grid_E.lon.shape), cmap = plt.cm.bwr)

# ax2 = fig.add_subplot(gs[0, 0])  
# csax2 = cs.CSplot(ax2, osse_model.grid_E, gridtype='cs')
# csax2.contour(osse_model.grid_E.lon, osse_model.grid_E.lat, B_north_ref.reshape(osse_model.grid_E.lon.shape), cmap = plt.cm.bwr)

# # Scatter plot
# ax3 = fig.add_subplot(gs[1, :])
# ax3.scatter(B_north_ref, B_north_pred, alpha=.3, color='grey')
# ax3.set_xlabel("from get_B")
# ax3.set_ylabel("from LompeOSSE")
# # ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], 'k--', lw=1)  # 1:1 line


# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(8, 8))
# gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# ax1 = fig.add_subplot(gs[0, 1])  
# csax1 = cs.CSplot(ax1, osse_model.grid_E, gridtype='cs')
# csax1.contour(osse_model.grid_E.lon, osse_model.grid_E.lat, B_up_pred.reshape(osse_model.grid_E.lon.shape), cmap = plt.cm.bwr)

# ax2 = fig.add_subplot(gs[0, 0])  
# csax2 = cs.CSplot(ax2, osse_model.grid_E, gridtype='cs')
# csax2.contour(osse_model.grid_E.lon, osse_model.grid_E.lat, B_up_ref.reshape(osse_model.grid_E.lon.shape), cmap = plt.cm.bwr)

# # Scatter plot
# ax3 = fig.add_subplot(gs[1, :])
# ax3.scatter(B_up_ref, B_up_pred, alpha=.3, color='grey')
# ax3.set_xlabel("from get_B")
# ax3.set_ylabel("from LompeOSSE")
# # ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], 'k--', lw=1)  # 1:1 line


# plt.tight_layout()
# plt.show()


# %%

import matplotlib.gridspec as gridspec

# Bundle the components for easy looping
components = [
    ("B_east",  B_east_ref,  B_east_pred),
    ("B_north", B_north_ref, B_north_pred),
    ("B_up",    B_up_ref,    B_up_pred)
]

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(len(components), 3, height_ratios=[1]*len(components))

for i, (label, ref, pred) in enumerate(components):
    # Reference map
    ax_ref = fig.add_subplot(gs[i, 0])
    csax_ref = cs.CSplot(ax_ref, osse_model.grid_E, gridtype='cs')
    im = csax_ref.contour(
        osse_model.grid_E.lon,
        osse_model.grid_E.lat,
        ref.reshape(osse_model.grid_E.lon.shape),
        cmap=plt.cm.bwr
    )
    ax_ref.set_title(f"{label} (reference)")

    # Predicted map
    ax_pred = fig.add_subplot(gs[i, 1])
    csax_pred = cs.CSplot(ax_pred, osse_model.grid_E, gridtype='cs')
    csax_pred.contour(
        osse_model.grid_E.lon,
        osse_model.grid_E.lat,
        pred.reshape(osse_model.grid_E.lon.shape),
        cmap=plt.cm.bwr
    )
    ax_pred.set_title(f"{label} (predicted)")

    # Scatter plot
    ax_scatter = fig.add_subplot(gs[i, 2])
    ax_scatter.scatter(ref, pred, alpha=0.3, color="grey")
    minv = min(ref.min(), pred.min())
    maxv = max(ref.max(), pred.max())
    # ax_scatter.plot([minv, maxv], [minv, maxv], "k--", lw=1)  # 1:1 line
    ax_scatter.set_xlabel("Reference (get_B)")
    ax_scatter.set_ylabel("Predicted (LompeOSSE)")
    ax_scatter.set_title(f"{label} scatter")

plt.tight_layout()
plt.show()
# %%
