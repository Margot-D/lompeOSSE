""" 
    This script:
        - performs spherical harmonic analysis of REMIX horizontal ionospheric currents
        - uses a lot of memory because of the large number of SH coefficients
        - is included for completion, it is not intended to be run by users
        - requires the PynaMIT package to run 


"""

import numpy as np
import h5py
import dipole
from sh_basis import SHBasis
from grid import Grid
from basis_evaluator import BasisEvaluator
#import pynamit # https://github.com/DynaMIT-uib/PynaMIT
import matplotlib.pyplot as plt
dp = dipole.Dipole(2020)

# copied from the remix code
def efield(x, y, Psi, returnDeltas=False, ri = 6.5*1e3):
        """
        Calculate the electric field at each point in the grid.
        Args:
            returnDeltas (bool, optional): Whether to return the differences in theta and phi along with the electric field.
            ri (float, optional): The value of Ri multiplied by 1e3.
        Returns:
            tuple: A tuple containing the electric field components (-etheta, -ephi) in V/m.
                   If `returnDeltas` is True, it also includes the differences in theta and phi (dtheta, dphi).
        Raises:
            SystemExit: If the variables have not been initialized for the specific hemisphere.
        Note:
            This method assumes that the variables have been initialized for the specific hemisphere
            by calling the `init_var` method prior to calculating the electric field.
        """
        Nt,Np = Psi.shape
        theta = np.arcsin(np.sqrt(x**2 + y**2))
        phi   = np.arctan2(y, x)
        # interpolate Psi to corners
        Psi_c = np.zeros(x.shape)
        Psi_c[1:-1,1:-1] = 0.25*(Psi[1:,1:]+Psi[:-1,1:]+Psi[1:,:-1]+Psi[:-1,:-1])
        # fix up periodic
        Psi_c[1:-1,0]  = 0.25*(Psi[1:,0]+Psi[:-1,0]+Psi[1:,-1]+Psi[:-1,-1])
        Psi_c[1:-1,-1] = Psi_c[1:-1,0]
        # fix up pole
        Psi_pole = Psi[0,:].mean()
        Psi_c[0,1:-1] = 0.25*(2.*Psi_pole + Psi[0,:-1]+Psi[0,1:])
        Psi_c[0,0]    = 0.25*(2.*Psi_pole + Psi[0,-1]+Psi[0,0])
        Psi_c[0,-1]   = 0.25*(2.*Psi_pole + Psi[0,-1]+Psi[0,0])
        # fix up low lat boundary
        # extrapolate linearly just like we did for the coordinates
        # (see genOutGrid in src/remix/mixio.F90)
        # note, neglecting the possibly non-uniform spacing (don't care)
        Psi_c[-1,:] = 2*Psi_c[-2,:]-Psi_c[-3,:]
        # now, do the differencing
        # for each cell corner on the original grid, I have the coordinates and Psi_c
        # need to find the gradient at cell center
        # the result is the same size as Psi
        # first etheta
        tmp    = 0.5*(Psi_c[:,1:]+Psi_c[:,:-1])  # move to edge center
        dPsi   = tmp[1:,:]-tmp[:-1,:]
        tmp    = 0.5*(theta[:,1:]+theta[:,:-1])
        dtheta = tmp[1:,:]-tmp[:-1,:]
        etheta = dPsi/dtheta/ri  # this is in V/m
        # now ephi
        tmp    = 0.5*(Psi_c[1:,:]+Psi_c[:-1,:])  # move to edge center
        dPsi   = tmp[:,1:]-tmp[:,:-1]
        tmp    = 0.5*(phi[1:,:]+phi[:-1,:])
        dphi   = tmp[:,1:]-tmp[:,:-1]
        tc = 0.25*(theta[:-1,:-1]+theta[1:,:-1]+theta[:-1,1:]+theta[1:,1:]) # need this additionally
        ephi = dPsi/dphi/np.sin(tc)/ri  # this is in V/m
        if returnDeltas:
            return (-etheta,-ephi,dtheta,dphi)  # E = -grad Psi
        else:
            return (-etheta,-ephi)  # E = -grad Psi
        
# --> same as my get_efield function in lompeosse
# could be replaced by
# from lompeosse import get_E
# etheta, ephi = get_E()


# datafile = '../data/msphere.mix.h5'
datafile = '../Gamera_data.h5'
step = 'Step#0'
# ['Step#0', 'Step#13', 'Step#19', 'Step#20', 'Step#21']

data = h5py.File(datafile, 'r')
# read coords and calculate angles
x = data['X'][:]
y = data['Y'][:]
theta = np.arcsin(np.sqrt(x**2 + y**2))
phi   = np.arctan2(y, x)
theta = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1]/2
phi   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :]/2

# Normalize azimuthal angles to [0 - 2pi] # !! check if useful with Kalle 
phi[phi < 0] = phi[phi < 0] + 2*np.pi
phi[:, 0] -= 2 * np.pi  # Adjust first column

# read data
data_step = data[step]
Hall_n = data_step['Hall conductance NORTH'][:]
Hall_s = data_step['Hall conductance SOUTH'][:]
Pedersen_n = data_step['Pedersen conductance NORTH'][:]
Pedersen_s = data_step['Pedersen conductance SOUTH'][:]
Phi_n = data_step['Potential NORTH'][:]
Phi_s = data_step['Potential SOUTH'][:]

# calculate the electric field
Eth_n, Eph_n = efield(x, y, Phi_n)
Eth_s, Eph_s = efield(x, y, Phi_s)
Eth = np.hstack((Eth_n.flatten(), Eth_s.flatten()))
Eph = np.hstack((Eph_n.flatten(), Eph_s.flatten()))
Eh  = np.vstack((Eth, Eph)) # 2 x N array with theta and phi components in the rows


# calculate the current - but first I need the main magnetic field unit vector (dipole field)
#   and the third component of E:
lat = np.hstack((90 - np.rad2deg(theta).flatten(), -90 + np.rad2deg(theta).flatten()))
lon = np.hstack((np.rad2deg(phi).flatten(), np.rad2deg(phi).flatten()))
B  = np.vstack(dp.B(lat, 1))
b  = B / np.linalg.norm(B, axis = 0)
b  = np.vstack((b[1], -b[0], np.zeros(b.shape[1]))) # r, theta phi
E = np.vstack((-(b[1] * Eh[0])/b[0], Eh[0], Eh[1])) # r, theta, phi

SP = np.hstack((Pedersen_n.flatten(), Pedersen_s.flatten())).reshape((1, -1)) # shape (1, N) 
SH = np.hstack((Hall_n.flatten()    , Hall_s.flatten())    ).reshape((1, -1)) # shape (1, N)
bxE = np.cross(b, E, axisa = 0, axisb = 0, axisc = 0)
j = SP * E[1:] * SH * bxE[1:] # horizontal components



# spherical harmonic analysis
N, M = 50, 50 # 150, 150 corresponds to 11475 n,m-pairs
shbasis  = SHBasis(N, M)
datagrid = Grid(lat = lat, lon = lon)
datagrid_evaluator = BasisEvaluator(shbasis, datagrid, reg_lambda = 0)# 1e-5)#1e0)# 10**1)
#gtg = datagrid_evaluator.least_squares_helmholtz.ATWA
j_coeffs = datagrid_evaluator.grid_to_basis(j, helmholtz = True)
j_m = datagrid_evaluator.basis_to_grid(j_coeffs, helmholtz = True)
j_coeff_cf, j_coeff_df = j_coeffs

# save coefficient for each time step
np.save(f'cfcoeff_{step}.npy', j_coeff_cf)
np.save(f'dfcoeff_{step}.npy', j_coeff_df)

# make some plots to test if it worked
fig, ax = plt.subplots(figsize = (8, 8))
ax.hist2d(j_m.flatten(), j.flatten(), bins = (100, 100), norm = 'log', range = ((-.3, .3), (-.3, .3)))
ax.set_aspect('equal')
ax.set_xlabel('j components from inversion')
ax.set_ylabel('j components from REMIX')
ax.plot([-.3, .3], [-.3, .3], 'k-')
plt.show()


