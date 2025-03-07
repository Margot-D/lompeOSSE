#%%

################################################
################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import polplot
import matplotlib.cm as cm
import datetime as dt

import kaipy.remix.remix as remix
import secsy as cs

from ppigrf import igrf_gc
import lompe
from lompe.utils.time import yearfrac_to_datetime
from lompe.utils.conductance import hardy_EUV


import apexpy


import os

import sys
sys.path.append('/Users/margot/Docs/Academia/Research/Python/')

import get_gamera_EVB
from importlib import reload
reload(get_gamera_EVB) 

from get_gamera_EVB import get_Gdata, get_E, get_V, get_B, get_conductance_functions, interp2lompegrid

plt.rcParams['figure.dpi'] = 300 

################################################
################################################

# ########
# Loading datasets
# ########

# import ionospheric data from REMIX
# basedir = '/Users/margot/Docs/Academia/Research/Python/MAGEColab'
# fdir = os.path.join(basedir,'GrossREU') # directory containing the output from a MAGE magnetosphere run
# ftag = 'GrossREUSlim' # name that identifies the MAGE output
# mixFiles = os.path.join(fdir,"%s.mix.h5"%(ftag))
mixFiles = '/Users/margot/Downloads/msphere.mix.h5' # Kareem's dataset

# ########
# Get data
# ########

# initialise for specified hemisphere 
hN, hS = 'NORTH', 'SOUTH'
hem = hN

nstep = 1 # step number in the GAMERA run

data = get_Gdata(mixFiles, nstep, hem) # retrieve data manually
glonG = data['glon']
glatG = data['glat']

# ########
# 
# ########


# ########
# Quick tests
# ########

# ### 1) using polplot

potG = data['Potential']
plt.imshow(potG) # to get an image of how the potential varies with latitude and longitude

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(data['mlat'], data['mlt'], potG, cmap='viridis') 
# pax.contour(glatG, glonG/15, potG, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.show()

# ### 2) using the remix plotting module

# datar = remix.remix(mixFiles,nstep)
# datar.init_vars(hem)
# datar.plot('potential')

#%% 

# ########
# Lompe USER set-up
# ########

# epoch used for IGRF dependent calculations
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])

# make apex object for magnetic coordinates
apx = apexpy.Apex(time[0].year)

# Set up Lompe model grid (user)
position = (0,90)
orientation = 0
projection = cs.CSprojection(position, orientation)
# L, W, Lres, Wres = 220000e3,220000e3,100e3,100e3 # 180000e3,180000e3,300e3,300e3
L, W, Lres, Wres = 10000e3,10000e3,400e3,400e3 # 180000e3,180000e3,300e3,300e3
RE = 6371.2 # Earth radius in kilometers
R = RE + 120 # Ionospheric radius in kilometers (distance from center of Earth)
grid = cs.CSgrid(projection, L, W, Lres, Wres, R=R*1e3) 

print('\n Info cubed sphere grid: ', grid, '\n')

# ########
# user coordinates
# ########

# Define longitude and latitude as 2D or 1D (???) arrays
length = 50 # lon and lat should be the same length
lon_values = np.linspace(-179, 180, length) # is it the same to use -180 - 180 or 0 - 360?
lat_values = np.linspace(50, 89, length) # does not work below 40 deg, why? i guess outside the grid (I thought there was something checking for that)
lon, lat = np.meshgrid(lon_values, lat_values) # Create a meshgrid for a 2D grid
iii = grid.ingrid(lon, lat) # lonG and latG must be in degrees
lon, lat = lon[iii], lat[iii] # does not work as expected
user_coords = {'lon': lon, 'lat': lat}


# ########
# Extract GAMERA data for given input grid,lon,lat
# ########

# Extract GAMERA electric field for user lon, lat (grid)
Er, Eph, Eth, interp_coords = get_E(grid, user_coords, data, time[0], test=True) # returns E radial, phi, theta, the corresponding coordinates and the main magnetic field

# Derive GAMERA convection using the GAMERA electric field
Vr, Vph, Vth, _ = get_V(grid, user_coords, data, time[0]) # returns convection

################
# Run Lompe inversion using the GAMERA E, V data (and B in the future)
################

#######
# Create data objects for Lompe
#######

coords = np.vstack((interp_coords['glon'].flatten(), interp_coords['glat'].flatten()))

# Electric field data object 
Ee = Eph
En = -Eth
E_values = np.vstack((Ee.flatten(), En.flatten()))
efield_Gdata = lompe.model.Data(E_values, coords, datatype = 'Efield', iweight=1.0, error=1e-3)

# Convection data object  
Ve = Vph
Vn = -Vth
V_values = np.vstack((Ve.flatten(), Vn.flatten())) 
conv_Gdata = lompe.model.Data(V_values, coords, datatype = 'convection', iweight=1.0, error=50) 


# Magnetic field data object


#######
# Run LOMPE
#######

SHfunc, SPfunc = get_conductance_functions(grid, data)

# Create lompe model object
emodele = lompe.model.Emodel(grid, Hall_Pedersen_conductance=(SHfunc, SPfunc), dipole = True)
emodelc = lompe.model.Emodel(grid, Hall_Pedersen_conductance=(SHfunc, SPfunc), dipole = True)

# add datasets
emodele.add_data(efield_Gdata) 
emodelc.add_data(conv_Gdata) 

# emodel.add_data(efield_Gdata, conv_Gdata)

# run inversion #FIX REGULARIZATION PARAMETERS
emodele.run_inversion(l1 = 1, l2 = 0) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction
emodelc.run_inversion(l1 = 1, l2 = 0) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

# Lompe plot
fig = lompe.model.visualization.lompeplot(emodele, include_data = True, time = time, apex = apx)
fig = lompe.model.visualization.lompeplot(emodelc, include_data = True, time = time, apex = apx)

#%%

# Electric potential test (meant to check if the Gamera data can be used in Lompe)

#######
# Reconstruct electric potential using Lompe
#######

# Determine the reconstructed potential at glonG, glatG (original Gamera coordinates)
potLe = emodele.E_pot(lon=glonG, lat=glatG) * 1e-3 # V
potLc = emodelc.E_pot(lon=glonG, lat=glatG) * 1e-3 # V

# Stack arrays to find where NaNs exist (is there any nans in these arrays?)
mask = ~np.isnan(glatG.flatten()) & ~np.isnan(glonG.flatten()) & ~np.isnan(potLe)
mask = ~np.isnan(glatG.flatten()) & ~np.isnan(glonG.flatten()) & ~np.isnan(potLc)

# mask = ~np.isnan(glatG) & ~np.isnan(glonG) & ~np.isnan(potL)

# Apply mask to keep only valid values
glatG_clean = glatG.flatten()[mask]
glonG_clean = glonG.flatten()[mask]
potLe_clean = potLe[mask]
potLc_clean = potLc[mask]

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(glatG_clean, glonG_clean/15, potLe_clean, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('potL (efield data)')
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(glatG_clean, glonG_clean/15, potLc_clean, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('potL (convection data)')
plt.show()

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(glatG_clean, glonG_clean/15, potLe_clean-potLc_clean, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('potL (efield) - potL (convection)')
plt.show()

#######
# Compare with original Gamera potential (same coordinates as the ones used for the reconstruction)
#######

# Orignal Gamera potential at glonG, glatG
potG = data['Potential'] # in V

potG_clean = potG.flatten()
glatG_clean = glatG.flatten()
glonG_clean = glonG.flatten()

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(glatG_clean, glonG_clean/15, potG_clean-potLe_clean, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('potG - potL (original gamera glon,glat)')
plt.show()

# Gamera potential VS lompe reconstructed potential (should be a line)
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(potG_clean, potLe_clean, alpha=.3, color='grey')
plt.xlabel('Gamera potential')
plt.ylabel('Lompe potential')
plt.show()

# fig, ax = plt.subplots(figsize=(8,8))
# csax0 = cs.CSplot(ax, grid, gridtype='geo')
# csax0.contour(potL.reshape(lon.shape))




#%%

# Electric potential test (meant to check if the Gamera data can be used in Lompe)

#######
# Reconstruct electric potential using Lompe 
#######

# Determine the reconstructed potential at glonG, glatG (interpolated Gamera coordinates)
# potL = emodele.E_pot(lon=interp_coords['glon'], lat=interp_coords['glat']) * 1e-3 # V
potL = emodele.E_pot(lon=grid.lon, lat=grid.lat) * 1e-3 # V

# # Stack arrays to find where NaNs exist (is there any nans in these arrays?)
# mask = ~np.isnan(interp_coords['glon'].flatten()) & ~np.isnan(interp_coords['glat'].flatten()) & ~np.isnan(potL)

glat = interp_coords['glat'].flatten()#[mask]
glon = interp_coords['glon'].flatten()#[mask]
glat = grid.lat.flatten()
glon = grid.lon.flatten()


fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
pax.contour(glat, glon/15, potL, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('potL (grid lon,lat)')
plt.show()


#######
# Compare with original Gamera potential (interpolated)
#######

# Orignal Gamera potential at glonG, glatG
potG = data['Potential'] # in V
potinterp = interp2lompegrid(grid, glonG, glatG, potG)

# # remove nans
# mask = ~np.isnan(potinterp)
# potG_clean = potinterp[mask].flatten()

potG_clean = potinterp.flatten()

fig, ax = plt.subplots(figsize = (8, 8))
pax = polplot.Polarplot(ax, minlat = 10)
# pax.contour(glat, glon/15, potG_clean, cmap='viridis') 
pax.contour(glat, glon/15, potG_clean-potL, cmap='viridis') 
textargs = {'fontsize':15, 'color':'grey'}
pax.writeLATlabels()
pax.writeLTlabels(lat=8, **textargs)
plt.title('potG - potL (grid lon,lat)')
plt.show()

# Gamera potential VS lompe reconstructed potential (should be a line)
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(potG_clean, potL, alpha=.3, color='grey')
plt.xlabel('Gamera potential')
plt.ylabel('Lompe potential')
plt.show()


#%%

# pass magnetic field data
# pass the electric field data into lompe.OSSE_data (it's like lompe.Data but for simulation data)
