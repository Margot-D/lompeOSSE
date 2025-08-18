
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import h5py
import dipole
from magnetic_field_utils.sh_basis import SHBasis
from magnetic_field_utils.grid import Grid
from magnetic_field_utils.basis_evaluator import BasisEvaluator

mu0 = np.pi * 4e-7

# spherical harmonic analysis
N, M = 110, 110 # 150, 150 corresponds to 11475 n,m-pairs

# TODO: conversion from dipole to geographic
def get_B(r, theta, phi, RI, nstep, no_df_current = False):
    """ Calculate the magnetic field TODO in Tesla?
        
        RI is the ionosphere radius. r < RI is considered internal, r > RI is considered external
    
        theta, phi in degrees


        no_df_current: Set to True for 'space_mag_fac' data type (e.g, Iridium)

    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coeff_path = os.path.join(base_dir, 'B_coeffs')

    alpha_coeffs = np.load(coeff_path + f'/cfcoeff_Step#{nstep}.npy')
    psi_coeffs   = np.load(coeff_path + f'/dfcoeff_Step#{nstep}.npy')

    if no_df_current:
        if np.any(r < RI):
            print('Not a good idea to set no_df_current to True with r < RI')
        psi_coeffs *= 0

    # broadcast, get total shape, and flatten input arrays:
    radius, theta, phi = np.broadcast_arrays(r, theta, phi)
    shape = radius.shape
    radius, theta, phi = radius.flatten(), theta.flatten(), phi.flatten()
    B = np.full((3, radius.size), np.nan) # initialize array to hold magnetic field

    iii = radius < RI
    if np.sum(iii) > 0: # internal:
        r, th, ph = radius[iii], theta[iii], phi[iii]
        grid = Grid(theta = th, phi = ph)
        shbasis  = SHBasis(N, M)
        n = shbasis.n
        grid_evaluator = BasisEvaluator(shbasis, grid)
        
        kappa = psi_coeffs * (n + 1) / (2 * n + 1) * mu0
        Btheta, Bphi = (grid_evaluator.G_grad * np.expand_dims(r/RI, -1)**n).dot(kappa)
        Br = (grid_evaluator.G * np.expand_dims(r/RI, -1)**(n-1)).dot(kappa * n)

        B[0, iii] = Br
        B[1, iii] = Btheta
        B[2, iii] = Bphi


    if np.sum(~iii) > 0: # external:
        r, th, ph = radius[~iii], theta[~iii], phi[~iii]
        grid = Grid(theta = th, phi = ph)
        shbasis  = SHBasis(N, M)
        n = shbasis.n
        grid_evaluator = BasisEvaluator(shbasis, grid)

        # psi part
        kappa = -psi_coeffs * n / (2 * n + 1) * mu0
        Btheta_psi, Bphi_psi = (grid_evaluator.G_grad * np.expand_dims(RI/r, -1)**(n+1)).dot(kappa)
        Br = (grid_evaluator.G * np.expand_dims(RI/r, -1)**(n+2)).dot(-kappa * (n + 1))

        # alpha part
        alpha = -alpha_coeffs * mu0 / (n * (n + 1))
        Btheta_alpha, Bphi_alpha = grid_evaluator.G_rxgrad.dot(alpha)

        B[0, ~iii] = Br
        B[1, ~iii] = Btheta_psi + Btheta_alpha
        B[2, ~iii] = Bphi_psi + Bphi_alpha

    B = B.reshape((3, ) + shape)

    return(B * 1e9) # TODO in tesla?


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import polplot

    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 10))
    paxes = np.vectorize(polplot.Polarplot)(axes)

    # radii
    RI = (6371.2 + 300)*1e3 # ionosphere radius (CHANGE TO CORRECT GAMERA RADIUS)
    RI = 6500e3
    r = RI + 50e3

    # make scalargrid
    las, los = np.linspace(50, 90, 40), np.linspace(0, 360, 100)
    las, los = map(np.ravel, np.meshgrid(las, los))
    las, los = np.vstack((las, -las)), np.vstack((los, los))

    # make vectorgrid
    grid, _ = polplot.sdarngrid(dlat = 2, dlon = 2, latmin = 50)
    lav, lov = grid[0], grid[1] * 15    
    lav, lov = np.vstack((lav, -lav)), np.vstack((lov, lov))

    nstep= 0

    Bs = get_B(r, 90 - las, los, RI, nstep)

    alpha_coeffs = np.load(f'B_coeffs/cfcoeff_Step#{nstep}.npy')
    psi_coeffs   = np.load(f'B_coeffs/dfcoeff_Step#{nstep}.npy')
    j_coeffs = np.vstack((alpha_coeffs, psi_coeffs))
    N, M = 110, 110 # 150, 150 corresponds to 11475 n,m-pairs
    shbasis  = SHBasis(N, M)
    vgrid = Grid(lat = lav, lon = lov)
    vgrid_evaluator = BasisEvaluator(shbasis, vgrid, reg_lambda = 0)# 1e-5)#1e0)# 10**1)
    j_m = vgrid_evaluator.basis_to_grid(j_coeffs, helmholtz = True)

    MLT_ROT = 0
    for p in paxes[0]:
        j_ = np.split(j_m, 2, axis = 1)[0]
        p.quiver(lav[0], lov[0]/15 + MLT_ROT, -j_[0], j_[1], scale = 1)

    for p in paxes[1]:
        j_ = np.split(j_m, 2, axis = 1)[1]
        p.quiver(lav[1], lov[1]/15 + MLT_ROT,  j_[0], j_[1], scale = 1)

    for component in range(3):
        for hemisphere in range(2):
            paxes[hemisphere, component].contourf(las[0], los[0]/15 + MLT_ROT, Bs[component, hemisphere], cmap = plt.cm.bwr, levels = np.linspace(-100, 100, 20), zorder =0)

            if hemisphere == 0:
                paxes[hemisphere, 0].write(50, 12, r'$B_r$'     , ha = 'center', va = 'bottom', size = 16)
                paxes[hemisphere, 1].write(50, 12, r'$B_\theta$', ha = 'center', va = 'bottom', size = 16)
                paxes[hemisphere, 2].write(50, 12, r'$B_\phi$'  , ha = 'center', va = 'bottom', size = 16)
                paxes[hemisphere, 0].write(50, 18, 'North', ha = 'right', va = 'center', rotation = 90, size = 16)
            else:
                paxes[hemisphere, 0].write(50, 18, 'South', ha = 'right', va = 'center', rotation = 90, size = 16)


    plt.tight_layout()
    plt.show()

# def get_B_space(glat, glon, height, time, df_coeffs, cf_coeffs, epoch = 2015., chunksize = 15000):
#     """ Calculate model magnetic field in space 

#     This function uses dask to parallelize computations. That means that it is quite
#     fast and that the memory consumption will not explode unless `chunksize` is too large.

#     Parameters
#     ----------
#     glat : array_like
#         array of geographic latitudes (degrees)
#     glon : array_like
#         array of geographic longitudes (degrees)
#     height : array_like
#         array of geodetic heights (km)
#     time : array_like
#         list/array of datetimes, needed to calculate magnetic local time
#     cf_coeffs: array_like
#         array of SH coefficients for curl-free currents
#     df_coeffs: array_like
#         array of SH coefficients for divergence-free currents
#     epoch : float, optional
#         epoch (year) used in conversion to magnetic coordinates with the IGRF. Default = 2015.
#     chunksize : int, optional
#         the input arrays will be split in chunks in order to parallelize
#         computations. Larger chunks consumes more memory, but might be faster. Default is 15000.
#     coeff_fn: str, optional
#         file name of model coefficients - must be in format produced by model_vector_to_txt.py
#         (default is latest version)


#     Returns
#     -------
#     Be : array_like
#         array of model magnetic field (nT) in geodetic eastward direction 
#         (same dimension as input)
#     Bn : array_like
#         array of model magnetic field (nT) in geodetic northward direction 
#         (same dimension as input)
#     Bu : array_like
#         array of model magnetic field (nT) in geodetic upward direction 
#         (same dimension as input)


#     Note
#     ----
#     Array inputs should have the same dimensions.

#     """

#     # TODO: ADD CHECKS ON INPUT (?)

#     m = np.hstack((m_cf.flatten(), m_df.flatten())) # toroidal, poloidal coefficients 

#     NT, MT, NV, MV = N, M, N, M # spherical harmonic truncation levels


#     # turn coordinates/times into dask arrays
#     glat   = da.from_array(glat  , chunks = chunksize)
#     glon   = da.from_array(glon  , chunks = chunksize)
#     time   = da.from_array(time  , chunks = chunksize)
#     height = da.from_array(height, chunks = chunksize)

#     # get G0 matrix - but first make a wrapper that only takes dask arrays as input
#     _getG0 = lambda la, lo, t, h: getG0(la, lo, t, h, epoch = epoch, NT = NT, MT = MT, NV = NV, MV = MV)

#     # use that wrapper to calculate G0 for each block
#     G0 = da.map_blocks(_getG0, glat, glon, height, time, chunks = (3*chunksize, 1), new_axis = 1, dtype = np.float64)

#     # get a matrix with columns that are 19 unscaled magnetic field terms at the given coords:
#     B  = G0.dot( m ).compute()

#     # the resulting array will be stacked Be, Bn, Bu components. Return the partions
#     return np.split(B, 3)


# def get_B_ground(qdlat, mlt, height, current_height = 110, chunksize = 25000):
#     """ Calculate model magnetic field on ground 
    
#     This function uses dask to parallelize computations. That means that it is quite
#     fast and that the memory consumption will not explode unless `chunksize` is too large.

    
#     Parameters
#     ----------
#     qdlat : array_like or float
#         quasi-dipole latitude, in degrees. Can be either a scalar (float), or
#         an array with an equal number of elements as mlt
#     mlt : array_like
#         array of magnetic local times (hours)
#     height : float
#         geodetic height, in km (0 <= height <= current_height)
#     current_height : float, optional
#         height (km) of the current sheet. Default is 110.
#     chunksize : int
#         the input arrays will be split in chunks in order to parallelize
#         computations. Larger chunks consumes more memory, but might be faster. Default is 25000.
#     coeff_fn: str, optional
#         file name of model coefficients - must be in format produced by model_vector_to_txt.py
#         (default is latest version)


#     Returns
#     -------
#     Bqphi : array_like
#         magnetic field in quasi-dipole eastward direction
#     Bqlambda : array_like
#         magnetic field in quasi-dipole northward direction
#     Bqr : array_like
#         magnetic field in upward direction. See notes

#     Note
#     ----
#     We assume that there are no induced currents. The error in this assumption will be larger
#     for the radial component than for the horizontal components

#     Array inputs should have the same dimensions.
#     """

#     m =m_df.flatten() # poloidal coefficients 
#     N, M = N, M # stupid

#     # convert input to dask arrays - qdlat is converted to np.float32 to make sure it has the flatten function
#     qdlat = da.from_array(np.float32(qdlat).flatten(), chunks = chunksize)
#     mlt   = da.from_array(mlt.flatten()  , chunks = chunksize)

#     # get G0 matrix - but first make a wrapper that only takes dask arrays as input
#     _getG0 = lambda x, y: get_ground_field_G0(x, y, height, current_height, N = N, M = M)

#     # use that wrapper to calculate G0 for each block
#     G0 = da.map_blocks(_getG0, qdlat, mlt, chunks = 3*chunksize, new_axis = 1)

#     # get a matrix with columns that are 19 unscaled magnetic field terms at the given coords:
#     B_matrix  = G0.dot( m ).compute()


#     # the resulting array will be stacked Be, Bn, Bu components. Return the partions
#     return np.split(B, 3)


# if __name__ == '__main__':
    
#     # load SH coefficints -- this is supposed to go in the functions in the end:
#     j_coeff_cf = np.load('cfcoeff.npy')
#     j_coeff_df = np.load('dfcoeff.npy')
  

# shbasis  = pynamit.SHBasis(N, M)
# datagrid = pynamit.Grid(lat = lat, lon = lon)
# datagrid_evaluator = pynamit.BasisEvaluator(shbasis, datagrid, reg_lambda = 0)# 1e-5)#1e0)# 10**1)
# #gtg = datagrid_evaluator.least_squares_helmholtz.ATWA
# j_coeffs = datagrid_evaluator.grid_to_basis(j, helmholtz = True)
# j_m = datagrid_evaluator.basis_to_grid(j_coeffs, helmholtz = True)
# j_coeff_cf, j_coeff_df = j_coeffs

