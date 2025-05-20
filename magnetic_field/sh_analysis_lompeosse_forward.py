import numpy as np
import h5py
import dipole
import pynamit
from pyamps.sh_utils import getG0, get_ground_field_G0

# spherical harmonic analysis
N, M = 100, 50 # 150, 150 corresponds to 11475 n,m-pairs

j_coeff_cf = np.load('cfcoeff.npy')
j_coeff_df = np.load('dfcoeff.npy')


def get_B_space(glat, glon, height, time, df_coeffs, cf_coeffs, epoch = 2015., chunksize = 15000):
    """ Calculate model magnetic field in space 

    This function uses dask to parallelize computations. That means that it is quite
    fast and that the memory consumption will not explode unless `chunksize` is too large.

    Parameters
    ----------
    glat : array_like
        array of geographic latitudes (degrees)
    glon : array_like
        array of geographic longitudes (degrees)
    height : array_like
        array of geodetic heights (km)
    time : array_like
        list/array of datetimes, needed to calculate magnetic local time
    cf_coeffs: array_like
        array of SH coefficients for curl-free currents
    df_coeffs: array_like
        array of SH coefficients for divergence-free currents
    epoch : float, optional
        epoch (year) used in conversion to magnetic coordinates with the IGRF. Default = 2015.
    chunksize : int, optional
        the input arrays will be split in chunks in order to parallelize
        computations. Larger chunks consumes more memory, but might be faster. Default is 15000.
    coeff_fn: str, optional
        file name of model coefficients - must be in format produced by model_vector_to_txt.py
        (default is latest version)


    Returns
    -------
    Be : array_like
        array of model magnetic field (nT) in geodetic eastward direction 
        (same dimension as input)
    Bn : array_like
        array of model magnetic field (nT) in geodetic northward direction 
        (same dimension as input)
    Bu : array_like
        array of model magnetic field (nT) in geodetic upward direction 
        (same dimension as input)


    Note
    ----
    Array inputs should have the same dimensions.

    """

    # TODO: ADD CHECKS ON INPUT (?)

    m = np.hstack((m_cf.flatten(), m_df.flatten())) # toroidal, poloidal coefficients 

    NT, MT, NV, MV = N, M, N, M # spherical harmonic truncation levels


    # turn coordinates/times into dask arrays
    glat   = da.from_array(glat  , chunks = chunksize)
    glon   = da.from_array(glon  , chunks = chunksize)
    time   = da.from_array(time  , chunks = chunksize)
    height = da.from_array(height, chunks = chunksize)

    # get G0 matrix - but first make a wrapper that only takes dask arrays as input
    _getG0 = lambda la, lo, t, h: getG0(la, lo, t, h, epoch = epoch, NT = NT, MT = MT, NV = NV, MV = MV)

    # use that wrapper to calculate G0 for each block
    G0 = da.map_blocks(_getG0, glat, glon, height, time, chunks = (3*chunksize, 1), new_axis = 1, dtype = np.float64)

    # get a matrix with columns that are 19 unscaled magnetic field terms at the given coords:
    B  = G0.dot( m ).compute()

    # the resulting array will be stacked Be, Bn, Bu components. Return the partions
    return np.split(B, 3)


def get_B_ground(qdlat, mlt, height, current_height = 110, chunksize = 25000):
    """ Calculate model magnetic field on ground 
    
    This function uses dask to parallelize computations. That means that it is quite
    fast and that the memory consumption will not explode unless `chunksize` is too large.

    
    Parameters
    ----------
    qdlat : array_like or float
        quasi-dipole latitude, in degrees. Can be either a scalar (float), or
        an array with an equal number of elements as mlt
    mlt : array_like
        array of magnetic local times (hours)
    height : float
        geodetic height, in km (0 <= height <= current_height)
    current_height : float, optional
        height (km) of the current sheet. Default is 110.
    chunksize : int
        the input arrays will be split in chunks in order to parallelize
        computations. Larger chunks consumes more memory, but might be faster. Default is 25000.
    coeff_fn: str, optional
        file name of model coefficients - must be in format produced by model_vector_to_txt.py
        (default is latest version)


    Returns
    -------
    Bqphi : array_like
        magnetic field in quasi-dipole eastward direction
    Bqlambda : array_like
        magnetic field in quasi-dipole northward direction
    Bqr : array_like
        magnetic field in upward direction. See notes

    Note
    ----
    We assume that there are no induced currents. The error in this assumption will be larger
    for the radial component than for the horizontal components

    Array inputs should have the same dimensions.
    """

    m =m_df.flatten() # poloidal coefficients 
    N, M = N, M # stupid

    # convert input to dask arrays - qdlat is converted to np.float32 to make sure it has the flatten function
    qdlat = da.from_array(np.float32(qdlat).flatten(), chunks = chunksize)
    mlt   = da.from_array(mlt.flatten()  , chunks = chunksize)

    # get G0 matrix - but first make a wrapper that only takes dask arrays as input
    _getG0 = lambda x, y: get_ground_field_G0(x, y, height, current_height, N = N, M = M)

    # use that wrapper to calculate G0 for each block
    G0 = da.map_blocks(_getG0, qdlat, mlt, chunks = 3*chunksize, new_axis = 1)

    # get a matrix with columns that are 19 unscaled magnetic field terms at the given coords:
    B_matrix  = G0.dot( m ).compute()


    # the resulting array will be stacked Be, Bn, Bu components. Return the partions
    return np.split(B, 3)



shbasis  = pynamit.SHBasis(N, M)
datagrid = pynamit.Grid(lat = lat, lon = lon)
datagrid_evaluator = pynamit.BasisEvaluator(shbasis, datagrid, reg_lambda = 0)# 1e-5)#1e0)# 10**1)
#gtg = datagrid_evaluator.least_squares_helmholtz.ATWA
j_coeffs = datagrid_evaluator.grid_to_basis(j, helmholtz = True)
j_m = datagrid_evaluator.basis_to_grid(j_coeffs, helmholtz = True)
j_coeff_cf, j_coeff_df = j_coeffs



print(
"""
Next steps are to 
1) convert the coefficients to coefficients for magnetic potential, and 
2) use those coefficient to calculate the magnetic field at the desired locations, using the equations in Laundal et al. 2016
3) eventually, the calculation of the coefficients should be separated from the rest, so that we have them saved, and only step 2 should be done in the osse tool
""")








