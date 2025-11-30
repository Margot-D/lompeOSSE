"""
Browse individual snapshots from the provided Gamera dataset and find time step of interest 
"""

import kaipy.remix.remix as remix
import matplotlib as plt
import h5py
import os

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

with h5py.File(datapath, "r") as Gdata:
    print("Available time steps in Gamera dataset:")
    step_keys = [k for k in Gdata.keys() if k.startswith("Step#")]
    step_keys = sorted(step_keys, key=lambda k: int(k.split('#')[1]))    
    for key in step_keys:
        print("  •", key)

# Select snapshot number
step = 12

data = remix.remix(datapath,step) # use the remix portion of the <kaipy package to create the ionospheric (mix) object "data"
data.init_vars('NORTH') # use a sub-module of the remix class to initialise the variables based on the specified hemisphere

# plot parameter of interest, for example:
data.plot('current')
print('\n snapshot #', step)

# # list of the variables included in the remix-processed data object
# keys = data.variables.keys()
# print(keys)
# -> ['potential', 'current', 'sigmap', 'sigmah', 'energy', 'flux', 'eflux', 
# 'efield', 'joule', 'jhall', 'gtype', 'npsp', 'Menergy', 'Mflux', 'Meflux',
# 'Denergy', 'Dflux', 'Deflux', 'Penergy', 'Pflux', 'Peflux']

# to check default data limits, print data.variables