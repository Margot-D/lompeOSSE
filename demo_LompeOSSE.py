"""
Demo script for using the LompeOSSE module. 

The LompeOSSE Python module enables the automatic acquisition of synthetic data 
from Gamera simulation given an input regional grid, which are used to replace 
the input datasets and conductances of a user-defined electric field model, 
while preserving its original spatial and temporal configuration. Built on the 
Lompe technique, LompeOSSE reconstructs ionospheric electrodynamics based on synthetic 
(Gamera model) data and provides quantitative metrics to evaluate the accuracy of 
the Lompe output against the ground truth from the simulated data.

Stage 1: Configure model input (user_input.py)
=================================
Example user inputs for LompeOSSE. 
Users can freely modify any of the parameters in this file, including:
  • Event date and time
  • Lompe grid configuration (center, orientation, dimensions, resolution)
  • Conductance model
  • Observational datasets used to build the baseline electric field
  • Gamera simulation snapshot for synthetic data

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import secsy as cs
import h5py

import lompe
import user_input
from lompeosse import LompeOSSE

from magnetic_field_utils import get_B # TODO remove

#%% Stage 1:

# Import user input
apx = user_input.apx
time = user_input.time
Gstep = user_input.Gstep
mlt_offset = user_input.mlt_offset
grid = user_input.grid
cmod = user_input.cmod
lompe_datasets = user_input.lompe_datasets

#%% Stage 2: 

# Create Emodel object
model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

# Add data to model
for data_obj in lompe_datasets.values():
    model.add_data(data_obj)

#%% Stage 3: 

# Derive synthetic model
lompeosse_obj = LompeOSSE(model, nstep=Gstep, mlt_off=mlt_offset, epoch=time.year)
osse_model = lompeosse_obj.osse_model

#%% Stage 4: 

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
plt.show()

#%% Stage 5: 
# Validate synthetic model

# Load Gamera data

hemisphere = 'NORTH' if user_input.latc > 0 else 'SOUTH'

with h5py.File(user_input.datapath, 'r') as Gdata:
    potG = Gdata[f'Step#{Gstep}'][f'Potential {hemisphere}'][:]
    facG = Gdata[f'Step#{Gstep}'][f'Field-aligned current {hemisphere}'][:]

# Interpolate to Lompe grid
interp_potG = lompeosse_obj.interp2lompegrid(potG)
interp_facG = lompeosse_obj.interp2lompegrid(facG)

# LompeOSSE-reconstruted quantities

potOSSE = osse_model.E_pot(lon=grid.lon, lat=grid.lat) * 1e-3 # V
potOSSE = potOSSE.reshape(grid.lon.shape)

facOSSE = osse_model.FAC(lon=grid.lon, lat=grid.lat)
facOSSE = facOSSE.reshape(grid.lon.shape)

# Set FAC levels
fac_levels = np.linspace(-1.95, 1.95, 40) * 1e-6 * 2


# Plot

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

#%% TEST

RE = 6371.2 # Earth radius in km

## reference (from get_B)
glat, glon = osse_model.grid_E.lat.flatten(), osse_model.grid_E.lon.flatten()
coords = np.vstack((glon, glat))
mlat,mlon = apx.geo2apex(coords[1], coords[0], RE-RE) #lat, lon, height of the data points
theta = 90 - mlat
phi = mlon+(0*15)
refB = get_B(RE*1e3, theta, phi, Gstep, no_df_current=False) 

# in magnetic coordinates
Br_ref     = refB[0].flatten()
Btheta_ref = refB[1].flatten()
Bphi_ref   = refB[2].flatten()

# in geographic coordinates
f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = apx.basevectors_apex(coords[1], coords[0], height=RE-RE, coords = 'geo')
B_east_ref, B_north_ref = Bphi_ref*f1 - Btheta_ref*f2
B_up_ref = Br_ref

# predictions (lompe)
lompeB = osse_model.B_ground(lon=glon, lat=glat) #nT #geographic coordinates

B_east_pred     = lompeB[0].flatten()*1e9
B_north_pred = lompeB[1].flatten()*1e9
B_up_pred   = lompeB[2].flatten()*1e9


# %%

import matplotlib.gridspec as gridspec

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
        glon.reshape(osse_model.grid_E.lon.shape),
        glat.reshape(osse_model.grid_E.lon.shape),
        ref.reshape(osse_model.grid_E.lon.shape),
        cmap=plt.cm.bwr
    )
    ax_ref.set_title(f"{label} (reference)")

    # Predicted map
    ax_pred = fig.add_subplot(gs[i, 1])
    csax_pred = cs.CSplot(ax_pred, osse_model.grid_E, gridtype='cs')
    csax_pred.contour(
        glon.reshape(osse_model.grid_E.lon.shape),
        glat.reshape(osse_model.grid_E.lon.shape),
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
