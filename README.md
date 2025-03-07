# LompeOSSE

LompeOSSE is a tool based on the **Local mapping of polar ionospheric electrodynamics (Lompe)** technique, designed for use in **Observation System Simulation Experiments (OSSEs)**. It is an extension of the Lompe technique, providing the framework for simulating ionospheric electrodynamics in various observational contexts.


## Overview

LompeOSSE builds upon the Lompe technique to simulate and analyze the dynamics of ionospheric electrodynamics, with a particular focus on polar regions. It is used in conjunction with OSSEs to enhance the understanding and prediction of space weather phenomena. The tool is particularly useful for testing and evaluating space observation systems in simulated environments before real-world deployment.

For more information on the original Lompe technique, visit the [Lompe GitHub repository](https://github.com/klaundal/lompe).


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

from lompeOSSE import lompeOSSE (???)
osse_model = lompeOSSE(real_model, Gstep=1, epoch=2015)
osse_model.run_inversion()


## Documentation 

LompeOSSE follows a similar structure and documentation as the Lompe tool. For detailed information on specific features and usage, refer to the Lompe documentation or check the examples provided in the repository.

### Lompe papers
============
- Main Lompe paper that describes the technique: `Local Mapping of Polar Ionospheric Electrodynamics <https://doi.org/10.1029/2022JA030356>`_
- Paper about the Lompe code: `The Lompe code: A Python toolbox for ionospheric data analysis <https://doi.org/10.3389/fspas.2022.1025823>`_



## Funding

The development of LompeOSSE is funded by the ESA/PRODEX Experiment Arrangement.

