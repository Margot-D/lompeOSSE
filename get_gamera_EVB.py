# Created 13 jan 2025 - Margot Decotte

################################################
################################################

import h5py
import matplotlib.pyplot as plt
import numpy as np
import apexpy
import secsy as cs
from scipy.interpolate import griddata, RectBivariateSpline
from ppigrf import igrf_gc, igrf
from lompe.utils.time import yearfrac_to_datetime
import copy
import lompe



################################################
################################################

def get_Gdata(h5file, step, hem='north', epoch=2015.):
        
    """
    Download and extract GAMERA coordinates.
    
    Parameters:

    	h5file (str): Path to Gamera HDF5 file
		step (int): Step number (Gamera simulation)
        hem: hemisphere ('north' or 'south')
        epoch: decimal year used for...

    Returns:

        dataG: Dictionary containing the retrieved GAMERA dataset 
        plus derived mlat, mlon, mlt, glon, glat columns


    """

    Gdata = {}

    # Open the HDF5 file in read mode
    with h5py.File(h5file, "r") as f:
        Gdata['X'] = f['X'][:]
        Gdata['Y'] = f['Y'][:]
        for h in f['Step#%d' % step].keys():
            Gdata[h] = f['Step#%d' % step][h][:]


    # Initialise dataset for specific hemisphere
    # keep only mlat >0 or <0? 
    h = hem.upper()
    
    hemi_Gdata = {} # Create a new dictionary to store the selected variables

    # Loop through all keys in Gdata
    for key in Gdata.keys():
        if "north" in key.lower() and h == "NORTH":
            new_key = key.replace("NORTH", "").strip()  # Standardized key name
            hemi_Gdata[new_key] = Gdata[key]

        elif "south" in key.lower() and h == "SOUTH":
            new_key = key.replace("SOUTH", "").strip()
            hemi_Gdata[new_key] = Gdata[key]

        elif "north" not in key and "south" not in key:
            # Keep variables that are not hemisphere-specific
            hemi_Gdata[key] = Gdata[key]

    # Replace Gdata with the updated version
    Gdata.clear()
    Gdata.update(hemi_Gdata)

    # Cartesian coordinates
    X = Gdata['X']
    Y = Gdata['Y']

    # Spherical coordinates 
    r = np.sqrt(X**2 + Y**2)
    theta = np.arcsin(r) # colatitude in radians (??)
    phi   = np.arctan2(Y, X) # azimuthal angle in radians (theta column in remix file)

    # Ensure all the azimuthal angles are in the range [0 - 2pi] instead of [-pi - pi]
    phi[phi < 0] = phi[phi < 0] + 2*np.pi
    phi[:, 0] -= 2 * np.pi  # fixing the first phi point to just below 0
    		
    Gdata['R'] = r/1e3 # in km (is it used anywhere?)
    Gdata['THETA'] = theta
    Gdata['PHI'] = phi

    # Correct r, theta and phi to match other variables in GAMERA data file
    # r_trim     = (r[:-1, :-1] + r[:-1, 1:] + r[1:, :-1] + r[1:, 1:]) /4 # take the center of each grid cell by averaging values from adjacent points
    r_trim     = r[:-1, :-1] + np.diff(r, axis=0)[:, :-1]/2 + np.diff(r, axis=1)[:-1, :] /2 # averaged over all four corner points of each grid cell
    theta_trim = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1] /2 # averaged over theta respective grid directions
    phi_trim   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :] /2 # averaged over phi respective grid directions

    Gdata['r'] = r_trim/1e3 # in km (is it used anywhere?)
    Gdata['theta'] = theta_trim
    Gdata['phi'] = phi_trim

    # Derive mlat, mlon, mlt from trimmed theta and phi
    mlatG = 90 - np.rad2deg(theta_trim) # in degrees
    mlonG = np.rad2deg(phi_trim) # in degrees
    mltG = phi_trim * (12/np.pi) # in hours
    mltOffset = 12 # offset (in hours) to get 0/24 MLT in the bottom of a polar plot
    mltG = mltG + mltOffset 

    Gdata['mlat'] = mlatG
    Gdata['mlon'] = mlonG
    Gdata['mlt'] = mltG

    # Define minimum latitude and replace values outside mask with NaN
    min_lat = 20  # Should be at least 11
    mask = Gdata['mlat'] > min_lat  # Boolean mask based on latitude

    for key in Gdata.keys():
        if Gdata[key].shape == Gdata['mlat'].shape:  
            Gdata[key] = np.where(mask, Gdata[key], np.nan) # Apply the mask only to columns that have the same shape as 'mlat'
        else:
            #print(f"Skipping {key}: shape {Gdata[key].shape} does not match mlat")
            pass

    if Gdata["mlat"].min() < min_lat:
        print('Gdata["mlat"].min(): ', Gdata['mlat'].min(), 'degrees')
        print(f"Low latitude GAMERA data (< {min_lat} deg) has been discarded")
    
    # Make apex object for magnetic coordinates
    time = yearfrac_to_datetime([epoch])
    apx = apexpy.Apex(time[0].year)

    # Convert Gamera coordinates from magnetic dipole to geographic
    glatG, glonG, _ = apx.apex2geo(Gdata['mlat'], Gdata['mlon'], height=120)

    Gdata['glon'] = glonG
    Gdata['glat'] = glatG

    return Gdata

def get_E(Lgrid, user_coords, Gdata, time, test=False):

    """
    This function extracts the electric field from GAMERA REMIX datafile, for given 
    lon,lat. It first converts from magnetic dipole coordinates to a geographic 
    geocentric system. Then it interpolates the electric field from the GAMERA grid 
    to the cubed sphere grid defined by the user.

    Lgrid: 
    Gdata:
    time:
    test:

    """

    # Use the not trimmed coords (109, 721)
    x = Gdata['X']
    theta = Gdata['THETA']
    phi = Gdata['PHI']

    Psi = Gdata['Potential']

    # ########
    # Derive GAMERA electric field (based on REMIX efield() function)
    # ########

    # EthG_mag, EphG_mag = data.efield() # gives only theta (polar angle) and phi (azimuthal) components in a magnetic dipole system

    RionE = 6.5  # Earth Ionosphere radius in 1000 km
    ri = RionE * 1e3  # Convert to meters

    # Interpolate electric potential (Psi Ψ) to the grid cell corners
    Psi_c = np.zeros(x.shape) # Initialize interpolated array with same shape as x
    Psi_c[1:-1,1:-1] = 0.25 * (Psi[1:,1:] + Psi[:-1,1:] + Psi[1:,:-1] + Psi[:-1,:-1])  # Average surrounding cell values

    # Handle periodic boundary conditions (ensuring continuity across the grid edges)
    Psi_c[1:-1,0]  = 0.25 * (Psi[1:,0] + Psi[:-1,0] + Psi[1:,-1] + Psi[:-1,-1]) # Wrap around left/right boundary
    Psi_c[1:-1,-1] = Psi_c[1:-1,0] # Ensure consistency at the last column

    # Handle the pole boundary condition (assume a mean potential at the pole)
    Psi_pole = Psi[0,:].mean() # Average potential at the pole
    Psi_c[0,1:-1] = 0.25 * (2.*Psi_pole + Psi[0,:-1] + Psi[0,1:]) # Interpolate using the pole value
    Psi_c[0,0]    = 0.25 * (2.*Psi_pole + Psi[0,-1] + Psi[0,0]) # Handle first column at the pole
    Psi_c[0,-1]   = 0.25 * (2.*Psi_pole + Psi[0,-1] + Psi[0,0])	 # Handle last column at the pole		

    # Handle the low-latitude boundary using linear extrapolation (assumes a uniform spacing)
    Psi_c[-1,:] = 2 * Psi_c[-2,:] - Psi_c[-3,:] # Linear extrapolation for the last row

    # Compute electric field components (E = -∇Ψ)
    # First, compute etheta (meridional component of E-field)
    tmp    = 0.5 * (Psi_c[:,1:] + Psi_c[:,:-1])  # Average potential to move to edge center
    dPsi   = tmp[1:,:] - tmp[:-1,:] # Compute difference along latitude
    tmp    = 0.5 * (theta[:,1:] + theta[:,:-1]) # Compute midpoints of theta grid
    dtheta = tmp[1:,:] - tmp[:-1,:] # Compute difference in theta
    etheta = dPsi/dtheta/ri  # Convert to electric field (V/m)

    # Now compute ephi (zonal component of E-field)
    tmp    = 0.5 * (Psi_c[1:,:] + Psi_c[:-1,:]) 
    dPsi   = tmp[:,1:] - tmp[:,:-1]
    tmp    = 0.5 * (phi[1:,:] + phi[:-1,:])
    dphi   = tmp[:,1:] - tmp[:,:-1]
    tc = 0.25 * (theta[:-1,:-1] + theta[1:,:-1] + theta[:-1,1:] + theta[1:,1:]) # Compute average theta for each grid cell
    ephi = dPsi/dphi/np.sin(tc)/ri  # V/m

    # Compute final electric field components (E = -grad Ψ)
    EphG_mag = -ephi   # Zonal (east-west) component
    EthG_mag = -etheta # Meridional (north-south) component

    # Convert phi,theta to east,north components
    EeG_mag = EphG_mag # east 
    EnG_mag = -EthG_mag # north


    # ########
    # Convert electric field from magnetic dipole coordinates to a geographic geodetic system
    # ########

    mlatG = Gdata['mlat']
    mlonG = Gdata['mlon']
    rG = Gdata['r'] # km IS THAT WHAT I NEED TO USE IN THE CONVERSION FUNCTION???

    # convert gamera (magnetic dipole) coordinates to lompe (geocentric) coordinates
    EeG_geo, EnG_geo, _ = efield_gamera2geo(EeG_mag, EnG_mag, mlonG, mlatG, rG, time)  

    EeG, EnG = EeG_geo, EnG_geo
    
    assert EeG.shape == mlonG.flatten().shape, f"Dimensions of E_east,E_north {EeG.shape} and flattened mlonG,mlatG {mlonG.flatten().shape} must match!"


    # ########
    # Interpolate the electric field to cubed sphere grid longitude and latitude 
    # ########

    glonG = Gdata['glon']
    glatG = Gdata['glat']

    Ee, En, glon, glat = interp_efield_2geogrid(Lgrid, user_coords, glonG, glatG, EeG, EnG, test=test)

    # Convert back to theta, phi components
    Eph = Ee # azimuthal
    Eth = -En # polar

    ################
    # Calculate Er using the IGRF magnetic field
    ################

    B0 = get_Bigrf(Lgrid, glon, glat, time)

    assert Eth.shape == B0[2].shape, f"Dimensions of Eth {Eth.shape} and Bth {B0[2].shape} must match!"

    #####
    # Derive Er
    #####

    Er = -(Eth*B0[2] + Eph*B0[1])/B0[0]
    
    geo_coords = {'glon': glon, 'glat': glat}
    return Er, Eph, Eth, geo_coords # radial, phi, theta

def get_Bigrf(Lgrid, glon, glat, time):

    """
    Input: geocentric coordinates (after interpolation)
    """

    r = np.full(glon.shape, Lgrid.R*1e-3) # radius in km
    theta = 90 - glat # polar angle/colatitude, in degrees (south on an Earth-centered sphere with radius r)
    phi = glon # azimuthal angle (same as geographic longitude), in degrees (positive east) 

    #####
    # Calculate main field values for all grid points (geocentric IGRF) 
    # using user r, theta, phi having dimensions of interpolated Ee, En
    #####

    # fix with input height (same as in getGdata)
    Be_nT, Bn_nT, Bup_nT = igrf(phi, 90-theta, 120, time) # returns radial, south, east in nT 
    # used igrf_gc before 

    Bth_nT = - Bn_nT
    Bph_nT = Be_nT
    Br_nT = Bup_nT

    # Convert from nT to T
    Bth = Bth_nT * 1e-9
    Bph = Bph_nT * 1e-9
    Br = Br_nT * 1e-9

    # Make sure Bth and Eth have the same direction (South I think), same with Er and Br
    B0 = np.vstack((Br, Bph, Bth)) # radial, east, south 

    return B0

def get_V(Lgrid, user_coords, Gdata, time, test=False):

    """
    """

    Er, Eph, Eth, interp_coords = get_E(Lgrid, user_coords, Gdata, time, test=test) # returns E radial, phi, theta, the corresponding coordinates and the main magnetic field
    B0 = get_Bigrf(Lgrid, interp_coords['glon'], interp_coords['glat'], time)

    Br = B0[0]
    Bph = B0[1]
    Bth = B0[2]

    B = np.sqrt(Br**2 + Bth**2 + Bph**2)

    Vr = (Eth*Bph - Eph*Bth)/B**2
    Vph = (Er*Bth - Eth*Br)/B**2
    Vth = (Eph*Br - Er*Bph)/B**2

    return Vr, Vph, Vth, interp_coords

def get_B(): # KALLE IS WORKING ON IT 

    """
    """

    return Br, Bph, Bth


def efield_gamera2geo(E_mag_east, E_mag_north, mlon, mlat, height, time):

    """
    Convert GAMERA electric field from magnetic dipole coordinates (original GAMERA data) 
    to geographic geodetic coodinates (used in Lompe).

    E_mag_east: east component of GAMERA electric field 
    E_mag_north: north component of GAMERA electric field 
    mlon: GAMERA magnetic longitude
    mlat: GAMERA magnetic latitude
    height: geodetic height of the GAMERA data points (in km)

    """

    E_mag_east, E_mag_north = E_mag_east.flatten(), E_mag_north.flatten()
    mlon, mlat, height = mlon.flatten(), mlat.flatten(), height.flatten()

    apx = apexpy.Apex(time, refh = 110)

    glat, glon, _ = apx.apex2geo(mlat, mlon, height)
    f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = apx.basevectors_apex(glat, glon, height, coords = 'geo')
    
    # magnetic field inclination
    sinIm = 2 * np.sin(np.deg2rad(mlat)) / np.sqrt(4 - 3 * np.cos(np.deg2rad(mlat))**2) 

    # h = hem.upper()
    # if h == 'NORTH':
    E_geo_east, E_geo_north, E_geo_up = E_mag_east * d1 - (E_mag_north/sinIm) * d2
    # elif h == 'SOUTH': 
    #     E_geo_east, E_geo_north, E_geo_up = E_mag_east * d1 + (E_mag_north/sinIm) * d2 # is that correct??

    return E_geo_east, E_geo_north, E_geo_up # geodetic 


def interp_efield_2geogrid(Lgrid, user_coords, glonG, glatG, EeG, EnG, test=False):

    """
    Interpolate the GAMERA electric field (east and north components) from the
    GAMERA grid to the cubed sphere grid defined by user (grid used in Lompe)

    Lgrid: cubed sphere grid defined by user
    lonG: GAMERA magnetic longitude (in degrees)
    latG: GAMERA magnetic latitude (in degrees)
    EeG: east component of the GAMERA electric field
    EnG: north component of the GAMERA electric field
    test: True if you want to verify interpolation with plots

    returns: 
    # North and east components of the electric field in the user cubed sphere grid
    # glon, glat are the new user lon,lat (after filtering + flattening)
    """

    # ########
    # Using the cubed sphere grid, interpolate the electric field to input longitude and latitude
    # ########

    grid = Lgrid
    projection = grid.projection 
    lon, lat = user_coords['lon'], user_coords['lat'] # user_coords.values()
    # lon, lat = grid.lon, grid.lat # glon, glat of Lompe grid points (in deg)
    glonG, glatG = glonG.flatten(), glatG.flatten()

    # keep gamera points that are in the cubed sphere grid
    iii = grid.ingrid(glonG, glatG) # lonG and latG must be in degrees

    glonG, glatG, EeG, EnG = glonG[iii], glatG[iii], EeG[iii], EnG[iii]

    # Convert spherical coordinates (GEOGRAPHIC) to cartesian (GAMERA points)
    xiG, etaG, E_xi, E_eta = projection.vector_cube_projection(EeG, EnG, glonG, glatG)

    # Convert spherical coordinates to cartesian (input lon, lat)
    xi, eta = projection.geo2cube(lon.flatten(), lat.flatten(), set_points_off_cube_to_nan=True)
    # xi, eta = grid.xi.flatten(), grid.eta.flatten()

    # Interpolate data from the Gamera lonG, latG points to any lon, lat point (the interpolation uses cartesian coordinates xi and eta)
    E_xi_interp = griddata((xiG, etaG), E_xi, (xi, eta), method='linear') # is that method ok?
    E_eta_interp = griddata((xiG, etaG), E_eta, (xi, eta), method='linear')

    # Now convert back to spherical coordinates to get E_theta_interp and E_phi_interp (should be the same vectors as previous plot)
    glon, glat, Ee_interp, En_interp = projection.vector_cube_to_geo(E_xi_interp, E_eta_interp, xi, eta)

    En, Ee = En_interp, Ee_interp

    # # Remove nan values (do that outside the function maybe) 
    # Ee_clear = Ee.flatten()[~np.isnan(Ee)]
    # En_clear = En.flatten()[~np.isnan(En)]
    # glon_clear = glon.flatten()[~np.isnan(Ee)]
    # glat_clear = glat.flatten()[~np.isnan(Ee)]

    # # checks whether all values in En and Ee are finite (i.e., not NaN, inf, or -inf)
    # assert (np.sum(~np.isfinite(En_clear)) + np.sum(~np.isfinite(Ee_clear))) == 0

    # Ee   = Ee_clear
    # En   = En_clear
    # glon = glon_clear
    # glat = glat_clear

    if test:

        # First plot the electric field vector in its original coordinates
        fig,axs = plt.subplots(2,2,figsize=(10,10))
        csax0 = cs.CSplot(axs[0][0],grid,gridtype='geo')
        csax0.add_coastlines(color='grey')
        csax0.scatter(glonG[19], glatG[19], s=10, color='red')
        csax0.quiver(EeG, EnG, glonG, glatG, color='k')
        axs[0][0].set_xlabel('Longitude')
        axs[0][0].set_ylabel('Latitude')
        axs[0][0].set_title(r"$E_{field}$ in GAMERA spherical coordinates ($E_\phi$, $E_\theta$)")

        # Then plot E_xi, E_eta and compare direction and magnitude to E_phi, E_theta
        csax1 = cs.CSplot(axs[0][1],grid,gridtype='cs')
        csax1.add_coastlines(color='grey')
        axs[0][1].scatter(xiG[19], etaG[19], s=10, color='red')
        axs[0][1].quiver(xiG, etaG, E_xi, E_eta, color='k') # use matplotlib quiver function when it comes to xi and eta coordinates
        axs[0][1].set_title(r"$E_{field}$ in GAMERA cube coordinates ($E_\xi$, $E_\eta$)")

        # Now plot the electric field vector for the input lon, lat values
        csax2 = cs.CSplot(axs[1][0],grid,gridtype='cs')
        csax2.add_coastlines(color='grey')
        # csax2.scatter(xi, eta, s=2, color='red')
        axs[1][0].scatter(xi[5], eta[5], s=10, color='green')
        axs[1][0].quiver(xi, eta, E_xi_interp, E_eta_interp, color='k') # scale???
        axs[1][0].set_title(r"Interpolated $E_{field}$ (cube coord. $E_\xi$, $E_\eta$)")

        # Finally, plot the electric field back in a spherical system
        csax3 = cs.CSplot(axs[1][1],grid,gridtype='geo')
        csax3.add_coastlines(color='grey')
        csax3.scatter(glon[5], glat[5], s=10, color='green')
        csax3.quiver(Ee, En, glon, glat) #, scale=900
        axs[1][1].set_xlabel('Longitude')
        axs[1][1].set_ylabel('Latitude')
        axs[1][1].set_title(r"Interpolated $E_{field}$ (spherical coord. $E_\phi$, $E_\theta$)")
        plt.tight_layout()
        plt.show()

    return Ee.reshape(lon.shape), En.reshape(lon.shape), glon.reshape(lon.shape), glat.reshape(lon.shape)

def interp2lompegrid(Lgrid, glonG, glatG, var):

    """
    Interpolate GAMERA variables (scalars) from the GAMERA grid to the 
    cubed sphere grid defined by user (grid used in Lompe)

    Lgrid: cubed sphere grid defined by user
    glonG: GAMERA geographic longitude (in degrees) 
    glatG: GAMERA geographic latitude (in degrees) 
    var: data to interpolate
    """

    grid = Lgrid
    projection = grid.projection

    # keep gamera points that are in the cubed sphere grid
    iii = grid.ingrid(glonG, glatG, ext_factor = 1.5)

    # define Gamera xi and eta coordinates (convert from spherical glon,glat to cartesian xi,eta)
    xiG, etaG = projection.geo2cube(glonG[iii],glatG[iii]) 
    
    # define Lompe grid xi and eta coordinates
    xi, eta = grid.xi, grid.eta

    # Interpolate data from the Gamera grid to the Lompe grid
    varinterp = griddata((xiG,etaG), var[iii], (xi.flatten(), eta.flatten()))

    # reshape (to orignal 2D shape)
    varinterp = varinterp.reshape(grid.shape)

    return varinterp

def get_conductance_functions(grid, gamera_data):
        
    """
    Compute interpolated Hall and Pedersen conductance functions for Lompe.

    Parameters:
        grid: Lompe grid object.
        glonG, glatG: Longitude and latitude of GAMERA data.
        data: Dictionary containing 'Pedersen conductance' and 'Hall conductance'.

    Returns:
        Tuple (SHfunc, SPfunc) — Interpolated conductance functions for Lompe.
    """

    glonG = gamera_data['glon']
    glatG = gamera_data['glat']

    # Extract conductances from GAMERA dataset
    SPG = gamera_data['Pedersen conductance']
    SHG = gamera_data['Hall conductance']

    # Interpolate to user cubed sphere grid
    SPinterp = interp2lompegrid(grid, glonG, glatG, SPG)
    SHinterp = interp2lompegrid(grid, glonG, glatG, SHG)

    SHinterp_clear = np.ma.masked_invalid(SHinterp)
    SPinterp_clear = np.ma.masked_invalid(SPinterp)

    # checks whether all values in SPinterp and SHinterp are finite (i.e., not NaN, inf, or -inf)
    assert (np.sum(~np.isfinite(SPinterp_clear)) + np.sum(~np.isfinite(SHinterp_clear))) == 0

    # problem if nans in the conductances... 

    # Functions to interpolate to any lon,lat (RectBivariateSpline creates a continuous interpolation function from a grid of values)
    SHfuncCS0 = RectBivariateSpline(grid.xi[0], grid.eta[:,0], SHinterp_clear.T)
    SPfuncCS0 = RectBivariateSpline(grid.xi[0], grid.eta[:,0], SPinterp_clear.T)
    SHfuncCS = lambda xi, eta : SHfuncCS0(xi, eta, grid = False)
    SPfuncCS = lambda xi, eta : SPfuncCS0(xi, eta, grid = False)

    def SPfunc(glon, glat):
        ''' Pedersen conductance on grid '''
        xit, etat = grid.projection.geo2cube(glon, glat)
        return SPfuncCS(xit, etat)

    def SHfunc(glon, glat):
        ''' Hall conductance on grid '''
        xit, etat = grid.projection.geo2cube(glon, glat)
        return SHfuncCS(xit, etat)
    
    return SPfunc, SHfunc


def filter_datasets_by_grid(model, grid):
    """
    Filters datasets in the model based on whether their coordinates are inside the grid.

    Parameters:
        model: The Lompe model containing datasets.
        grid: The grid object with an ingrid method.

    Returns:
        The model with filtered datasets.
    """

    for datatype, dataset_list in model.data.items():
        if not dataset_list:
            print(f'{datatype} dataset is empty')
            continue

        valid_list = []  # Store filtered datasets

        for ds in dataset_list:
            print(datatype, ds.values.shape)
            lon, lat = ds.coords['lon'], ds.coords['lat']
            indices = np.where(grid.ingrid(lon, lat))[0]
            filtered_ds = ds.subset(indices)
            print('Dataset filtered, new shape: ', filtered_ds.values.shape)
            valid_list.append(filtered_ds)

        model.data[datatype] = valid_list  # Replace dataset with filtered version

    return model

def make_OSSE_model(real_model, Gstep, epoch=2015.):

    """
    real_data: lompe object 
    step number in the GAMERA run #DO WE WANT TO CHANGE THAT AT SOME POINT?
    epoch used for IGRF calculation, decimal year
    """

    # Making sure the datasets used in the lompe object (real_model) have their coordinates inside the grid
    ingrid_model = filter_datasets_by_grid(real_model, real_model.grid_J)

    # Make a copy of the real dataset (same properties)
    fake_model = copy.copy(ingrid_model)
    grid = fake_model.grid_J

    # #######
    # Load GAMERA data
    # #######

    mixFiles = '/Users/margot/Downloads/msphere.mix.h5' # Kareem's dataset
    hem = 'NORTH' # or 'SOUTH'# initialise for specified hemisphere 
    gamera_data = get_Gdata(mixFiles, Gstep, hem) 

    # #######
    # Replace datasets in model by GAMERA datasets
    # #######

    # efield_Gdata, conv_Gdata, ground_mag_Gdata, space_mag_full_Gdata, space_mag_fac_Gdata, fac_Gdata = lompe.model.Data([], [], datatype=i for i in datatypes)
    
    for datatype, dataset in fake_model.data.items():

        if not dataset:
            print(f'{datatype} dataset is empty')
            continue
        else:
            ds = dataset[0]
            coords = ds.coords # glon, glat
            stacked_coords = np.vstack((coords['lon'], coords['lat']))

            if datatype == 'convection':
                
                _, Vph, Vth, _ = get_V(grid, coords, gamera_data, yearfrac_to_datetime([epoch])[0]) # after interpolation to superdarn coordinates glon, glat
                print('Gamera convection data extracted')

                Ve = Vph
                Vn = -Vth

                # Projection along LOS (Ve, Vn will be same shape as los)
                los = ds.los
                le, ln = los[0], los[1]
                vlos = Ve * le + Vn * ln
                
                # Convection data object
                conv_Gdata = lompe.Data(vlos, stacked_coords, LOS=los, datatype='convection', iweight=1.0, error=100)

            if datatype == 'efield':

                _, Eph, Eth, _ = get_E(grid, coords, gamera_data, yearfrac_to_datetime([epoch])[0])

                # Electric field data object 
                Ee = Eph
                En = -Eth
                E_values = np.vstack((Ee.flatten(), En.flatten()))
                efield_Gdata = lompe.model.Data(E_values, stacked_coords, datatype='Efield', iweight=1.0, error=1e-3)
        
        # Gamera conductances
        SHfunc, SPfunc = get_conductance_functions(grid, gamera_data)

        # Reset model (delete datasets and clear model vectors)
        fake_model.clear_model(Hall_Pedersen_conductance = (SHfunc, SPfunc))
        
        # Replace original datasets by GAMERA datasets
        fake_model.add_data(conv_Gdata) # this is adding under 'convection' data but not replacing 
        # fake_model.add_data(conv_Gdata, efield_Gdata, ground_mag_Gdata, space_mag_full_Gdata, space_mag_fac_Gdata, fac_Gdata) # this is adding under 'convection' data but not replacing 

    return fake_model
