import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lompe
import lompe.data
from lompe.model.cmodel import Cmodel
from lompe.model.visualization import *
from lompe.utils.time import yearfrac_to_datetime
import apexpy
from lompeosse import get_B_mag

# Define event
event = '2014-12-15'
hour = 1
minute = 19
stime = dt.datetime(int(event[0:4]), int(event[5:7]), int(event[8:10]), hour, minute) # the specific time to model
DT = dt.timedelta(seconds = 2*60) # will select data from time +- DT

# Define grid
position = (0,90) # lon, lat
orientation = 0 #(-0.1, 1) # east, north
L, W, Lres, Wres = 20000.e3, 20000.e3, 200.e3, 200.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
refh = 120 # reference height in km
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = (6371.2 + refh)*1e3)

# Define conductance model using SSUSI image
tempfile_path = '/Users/margot/Docs/Academia/Research/Python/lompe/examples/sample_dataset/' # where .nc SSUSI-files are saved. You can change to fit your system.

cmod = Cmodel(grid, event, stime, spline_smoothing = 10, EUV = True, filtersize = 2, how = 'median', 
              param = 'lbhs', tempfile_path = tempfile_path, basepath = tempfile_path + '/raw/') #1000

RI = 6500e3 #m
r = RI + 50e3 #m

nstep = 0

# Create Emodel object
test_model = lompe.Emodel(grid, (cmod.hall, cmod.pedersen))

lat, lon = test_model.grid_E.lat.flatten(), test_model.grid_E.lon.flatten()

Bs = get_B_mag(r, 90 - lat, lon, RI, nstep)
coords = np.vstack((lon, lat))

Benu = np.empty(Bs.shape)
Benu[0] = Bs[2]
Benu[1] = -Bs[1]
Benu[2] = Bs[0]

#east,north,
mydata = lompe.Data(Bs * 1e-9, coords, datatype = 'space_mag_full', iweight = 1, error = 10e-9)


# Add data to model
test_model.add_data(mydata)

test_model.run_inversion(l1 = 1, l2 = 1) # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction

# Define epoch used for IGRF dependent calculations and apex object for ... (magnetic coordinate)
epoch = 2015. # decimal year
time = yearfrac_to_datetime([epoch])
apx = apexpy.Apex(time[0].year)

# fig = lompe.lompeplot(test_model, include_data = True, time = time, apex = apx)
fig = lompe.lompeplot(test_model, include_data = True, time = time, apex = apx, 
                      colorscales = {'fac'        : np.linspace(-2, 2, 40) * 1e-6 /7,
                                     'ground_mag' : np.linspace(-500, 500, 50) * 1e-9 / 3, # upward component
                                     'hall'       : np.linspace(0, 20, 32), # mho
                                     'pedersen'   : np.linspace(0, 20, 32)}, # mho
                        quiverscales = {'ground_mag'       : 600*1e-9, 
                                        'space_mag_fac'    : 600*1e-9, 
                                        'space_mag_full'   : 200*1e-9, 
                                        'electric_current' : 1}) # 1000*1e-3