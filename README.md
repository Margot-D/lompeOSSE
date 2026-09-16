# LompeOSSE: Observation System Simulation Experiments for Lompe

LompeOSSE is a Python toolbox designed for evaluating how well the **Local mapping of polar ionospheric electrodynamics (Lompe)** technique can reconstruct ionospheric electrodynamics.

It uses synthetic data from Gamera simulations in an **Observation System Simulation Experiment (OSSE)** framework. In this context, the OSSE provides a controlled way to test Lompe: a known “truth” is taken from a Gamera simulation, synthetic measurements are generated from this truth, and Lompe is used to reconstruct the electrodynamics from these measurements. The reconstruction can then be compared with the original simulation to assess Lompe's performance under different measurement configurations. 

## Overview

Similar to the original Lompe implementation, LompeOSSE computes an electric field model based on a user-defined configuration. The user begins by creating a standard Lompe model: specifying a local grid (i.e, spatial extent and resolution), selecting the input datasets, and defining ionospheric conductances. LompeOSSE then takes this user-defined model and replaces the observation data and conductances with synthetic counterparts extracted from the high-resolution Gamera simulation. The module supports synthetic data extraction for ionospheric convection (line-of-sight measurements), electric fields (derived from plasma drifts), and magnetic field perturbations. The result is a synthetic electric field model that preserves the structural characteristics of the original setup while enabling the reconstruction of local ionospheric electrodynamics using the Lompe technique within a fully controlled, synthetic OSSE environment.



<!--### Key features: -->

## Module contents

- **lompeosse.py** – core functionality of LompeOSSE
- **demo_LompeOSSE.py** – example usage of LompeOSSE, including validation of Lompe reconstruction against simulation
- **user_input.py** – example script to build a user-defined electric field model
- **Gamera snapshots (11 representative events)** – example synthetic data *(provided via a [Zenodo repository](https://zenodo.org/records/16882035))*
- **find-Gamera-snapshot.py** – helper script for browsing individual snapshots from the provided Gamera dataset to identify the time step of interest
- **Jupyter notebooks** – three representative OSSE case studies *(coming soon)*
- **magnetic_field_utils** – internal submodule for magnetic field processing (not intended for direct user use)

## Installation

### System prerequisites

LompeOSSE requires Python 3.11 or newer. 

LompeOSSE also depends on ApexPy (a Python wrapper for Apex coordinates), which uses Fortran code. Depending on your system, installing ApexPy may require a Fortran compiler and runtime.

For the most reliable installation, we recommend installing the required compilers before installing LompeOSSE, for example using Conda:
<!-- check if it works on windows! -->

```bash
conda install conda-forge::compilers
```

### Install LompeOSSE

```bash
git clone https://github.com/Margot-D/LompeOSSE.git
cd <path/to/LompeOSSE>

conda create -n LompeOSSE python=3.11
conda activate LompeOSSE

pip install .
```

The `pip install .` command installs LompeOSSE and all of its required Python dependencies automatically.

LompeOSSE can be installed in any compatible Python environment. However, using a dedicated environment is recommended to avoid dependency conflicts with other packages.

## Getting started 



## Module usage

1. **Configure model input (user_input.py)**:
In user_input.py, the user provides two types of inputs. First, the standard Lompe inputs (event date, local grid, conductance model, and observational datasets). These are generic Lompe settings and are not implemented by LompeOSSE, but they are required to generate the baseline electric field model (see next point). Second, the LompeOSSE-specific inputs, which are handled by the LompeOSSE module. These include selecting the Gamera simulation snapshot (time step) to generate synthetic observations, as well as an optional magnetic local time (MLT) offset that allows exploration of multiple configurations from a single snapshot.

2. **Generate the electric field model**:
LompeOSSE calls the Lompe module to compute the baseline electric field model using the user-defined settings.

3. **Derive the OSSE model (lompeosse.py)**:
LompeOSSE scans the user-selected input datasets and extracts the corresponding synthetic quantities from the Gamera simulation at the correct locations and time. The initial Lompe model is then reset and populated with these synthetic datasets (including synthetic conductances), producing an electric field model that matches the user-defined grid and configuration, but with fully synthetic inputs.

4. **Run the inversion to solve for electrodynamic quantities and visualize the results**:
The Lompe technique is applied to the synthetic model to reconstruct the electrodynamic quantities (e.g., electric potential, electric field, and current systems). A figure with the Lompe outputs is generated.

5. **Validate the OSSE setup**:
The reconstructed fields are then compared with the corresponding “ground truth” values from the Gamera simulation, allowing the user to assess the performance and accuracy of their OSSE setup.

## Note regarding the Gamera simulation snapshots
LompeOSSE automatically extracts and prepares synthetic data without any additional user steps, as long as the [this simulation dataset](https://zenodo.org/records/16882035) is used. Ideally, users should therefore use the provided snapshots, since the spherical harmonics coefficients used to derive the synthetic magnetic field have been calculated specifically for those cases. Users wishing to apply LompeOSSE to a different simulation run must perform a new spherical harmonic analysis of the horizontal ionospheric currents to generate the corresponding synthetic magnetic field. 

## Example usage

from lompeosse import LompeOSSE  
lompeosse_obj = LompeOSSE(model, nstep=1, mlt_off=mlt_offset, epoch=2015)  
osse_model = lompeosse_obj.osse_model  
osse_model.run_inversion(l1 = 1, l2 = 1)

## Documentation 

The LompeOSSE module includes in-script documentation at the beginning of each file. 
In addition, a demo script is provided to illustrate end-to-end usage of the OSSE workflow. Specific OSSE examples can be found in the examples/ folder.

For detailed information about the underlying Lompe technique, please refer to the official Lompe documentation at the [Lompe GitHub repository](https://github.com/klaundal/lompe). That repository also includes several usage examples. 

## Funding

The development of LompeOSSE is funded by the ESA/PRODEX Experiment Arrangement.

