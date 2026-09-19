"""
Browse the available snapshots in the provided Gamera dataset and inspect
individual snapshots to find the time step of interest for a LompeOSSE experiment.

Workflow:
    1. Run this script to list all available Gamera snapshots.
    2. Set `step` below to the snapshot you want to inspect.
    3. The selected snapshot is loaded and plotted using helper functions provided in gamera_tools.

The Gamera dataset is available from:
https://zenodo.org/records/16882035
"""


import matplotlib.pyplot as plt
import h5py
import os
from gamera_tools import *

# ---------------------------------------------------------------------------
# Load Gamera dataset
# ---------------------------------------------------------------------------

path = os.path.abspath(os.path.dirname(__file__))
datapath = os.path.join(path, 'lompeosse/data/Gamera_data.h5') 

if not os.path.exists(datapath):
    raise FileNotFoundError(f"Required Gamera dataset not found: {datapath}\n"
                            "Please download it from https://zenodo.org/records/16882035 "
                            "and place it in the 'lompeosse/data' folder.")

# ---------------------------------------------------------------------------
# List available snapshots
# ---------------------------------------------------------------------------

with h5py.File(datapath, "r") as Gdata:
    print("Available time steps in the provided Gamera dataset:")

    step_keys = [k for k in Gdata.keys() if k.startswith("Step#")]
    step_keys = sorted(step_keys, key=lambda k: int(k.split('#')[1]))    

    for key in step_keys:
        print("  •", key)

# ---------------------------------------------------------------------------
# Select a snapshot to inspect
# ---------------------------------------------------------------------------

# Change this number to the Gamera snapshot you want to inspect
step = 12

# ---------------------------------------------------------------------------
# Load and plot the selected snapshot
# ---------------------------------------------------------------------------

data = load_gamera_snapshot(datapath, step, hemisphere="NORTH")

plot_gamera(data, "potential")
plot_gamera(data, "current")
plot_gamera(data, "joule")


# plot_gamera(data, "sigmap")
# plot_gamera(data, "sigmah")
# plot_gamera(data, "energy")
# plot_gamera(data, "flux")
# plot_gamera(data, "eflux")

plt.show()