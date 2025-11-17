# LompeOSSE (draft)

LompeOSSE is a tool based on the **Local mapping of polar ionospheric electrodynamics (Lompe)** technique, designed for use in **Observation System Simulation Experiments (OSSEs)**. LompeOSSE provides a practical framework for testing the performance of Lompe in controlled, simulation-based scenarios — answering questions such as: Given realistic measurement geometries, how accurately can Lompe reconstruct the underlying electrodynamics represented in the simulation?

## Overview

Similar to the original Lompe implementation, LompeOSSE computes an electric field model based on a user-defined configuration. The user begins by creating a standard Lompe model – specifying a local grid (i.e, spatial extent and resolution), selecting the input datasets, and defining ionospheric conductances. LompeOSSE then takes this user-defined model and replaces the observation data and conductances with synthetic counterparts extracted from the high-resolution Gamera simulation. The module supports synthetic data extraction for ionospheric convection (line-of-sight measurements), electric fields (derived from plasma drifts), and magnetic field perturbations. The result is a synthetic electric field model that preserves the structural characteristics of the original setup while enabling the reconstruction of local ionospheric electrodynamics using the Lompe technique within a fully controlled, synthetic OSSE environment.

For more information on the original Lompe technique, visit the [Lompe GitHub repository](https://github.com/klaundal/lompe).


## Module contents

- **lompeosse.py** – core functionality of LompeOSSE
- **demo_LompeOSSE.py** – example usage and validation of a synthetic model
- **user_input.py** – example script to build a user-defined electric field model
- **Gamera snapshots (11 representative events)** – example synthetic data *(provided via a [Zenodo repository](https://zenodo.org/records/16882035))*
- **find-Gamera-snapshot.py** – helper script for browsing individual snapshots from the provided Gamera dataset to identify the time step of interest
**
- **Jupyter notebooks** – three representative OSSE case studies *(coming soon)*
- **magnetic_field_utils** – internal submodule for magnetic field processing (not intended for direct user use)


## Module structure

1. Configure model input (user_input.py)
The user specifies the event date, defines the local grid, selects the conductance model, and chooses the observational datasets that will serve as the template for the OSSE configuration. The user also selects the desired snapshot (time step) from the Gamera simulation, corresponding to the physical scenario of interest (default: snapshot #0). Additionally, the user can specify a magnetic local time (MLT) offset, allowing to explore multiple configurations from a single Gamera snapshot.

2. Generate the electric field model
LompeOSSE calls the Lompe module to compute the baseline electric field model using the user-defined settings.

3. Derive the OSSE model (lompeosse.py)
LompeOSSE scans the user-selected input datasets and extracts the corresponding synthetic quantities from the Gamera simulation at the correct locations and time. The initial Lompe model is then reset and populated with these synthetic datasets (including synthetic condutances), producing an electric field model that matches the user-defined grid and configuration, but with fully synthetic inputs.

4. Run the inversion to solve for electrodynamic quantities and vizualise the results
The Lompe technique is applied to the synthetic model to reconstruct the electrodynamic quantities (e.g., electric potential, electric field, and current systems). A figure with the Lompe outputs is generated.

5. Validate the OSSE setup
The reconstructed fields are then compared with the corresponding “ground truth” values from the Gamera simulation, allowing the user to assess the performance and accuracy of their OSSE setup.


## Dependencies

LompeOSSE shares dependencies with the [Lompe tool](https://github.com/klaundal/lompe). Ensure the following dependencies are installed:

- `apexpy <https://github.com/aburrell/apexpy/>`_
- matplotlib
- numpy
- pandas
- `ppigrf <https://github.com/klaundal/ppigrf/>`_ (install with pip install ppigrf)
- scipy
- xarray
- `astropy <https://github.com/astropy/astropy/>`_ (if you use the AMPERE Iridium data preprocessing scripts)
- `cdflib <https://github.com/MAVENSDC/cdflib/>`_ (for running lompe paper figures example 05)
- `madrigalWeb <https://pypi.org/project/madrigalWeb/>`_ (if you use the DMSP SSIES data preprocessing scripts)
- `netCDF4 <https://github.com/Unidata/netcdf4-python/>`_ (if you use the DMSP SSUSI data preprocessing scripts)
- `pyAMPS <https://github.com/klaundal/pyAMPS/>`_ (for running code paper figures example 08)
- `pydarn <https://github.com/SuperDARN/pydarn/>`_ (if you use the SuperDARN data preprocessing scripts)


## Installation

To install LompeOSSE, you can either clone the repository or install it directly from GitHub. ...


## Example usage

from lompeosse import LompeOSSE  
lompeosse_obj = LompeOSSE(model, nstep=1, mlt_off=mlt_offset, epoch=2015)  
osse_model = lompeosse_obj.osse_model  
osse_model.run_inversion(l1 = 1, l2 = 1)

## Documentation 

LompeOSSE follows a similar structure and documentation as the Lompe tool. For detailed information on specific features and usage, refer to the Lompe documentation or check the examples provided in the repository.

### Lompe papers:
- Main Lompe paper that describes the technique: `Local Mapping of Polar Ionospheric Electrodynamics <https://doi.org/10.1029/2022JA030356>`_
- Paper about the Lompe code: `The Lompe code: A Python toolbox for ionospheric data analysis <https://doi.org/10.3389/fspas.2022.1025823>`_



## Funding

The development of LompeOSSE is funded by the ESA/PRODEX Experiment Arrangement.

