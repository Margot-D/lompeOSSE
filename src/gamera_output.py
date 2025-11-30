import h5py
import os
import numpy as np
import apexpy
from scipy.interpolate import griddata
from ppigrf import igrf
import dipole # github.com/klaundal/dipole

RE = 6371.2 # Earth radius in km
RI = 6500 # Ionospheric radius in km (used in Gamera simulations)

class Gamera_output(object):

    """
    A helper class for loading, organizing, and working with Gamera simulation data in LompeOSSE.

    Parameters
    ----------
    filename : str
        Path to the Gamera HDF5 file
    hemisphere : str
        Hemisphere to load ("NORTH" or "SOUTH").
    time : datetime-like
        Time of event.
    timestep : int, optional
        Index of the snapshot within the Gamera file to load (default: 0).
        Use find-Gamera-snapshot.py to inspect available snapshots.


    mlt offset :
        Rotate the Gamera snapshot in magnetic local time (hours)


    """

    def __init__(self, time, timestep=0, mlt_offset=0, hemisphere='NORTH'):

        self.time = time
        self.refh = RI-RE # in km
        self.apex = apexpy.Apex(time, self.refh)

        self.timestep = timestep

        # Path to Gamera data file
        package_dir = os.path.dirname(__file__)                  # src/lompeosse
        root = os.path.abspath(os.path.join(package_dir, ".."))  # lompeosse/
        self.datapath = os.path.join(root, "data/Gamera_data.h5")

        if not os.path.exists(self.datapath):
            raise FileNotFoundError(
                f"Required file not found: {self.datapath}\n"
                "Please download it (https://zenodo.org/records/16882035) and place it in the 'data' folder."
            )

        # Load Gamera data at given time step
        print(f'Loading Gamera data from snapshot #{timestep} with {mlt_offset} hours MLT offset')
        self.gamera_data = self.get_Gamera_data(self.datapath, time, timestep, mlt_offset, hemisphere)

    # TODO remove mlt_off from gamera output, deal with it in Lompeosse by adding hours to time (ask Kalle again)
    def get_Gamera_data(self, fn, t, nstep, mlt_off, hem='NORTH'):

        """
        Reads Gamera data from an HDF5 file, extracts relevant variables based on 
        the specified timestep and hemisphere, converts magnetic dipole coordinates to geographic...

        Returns:
        --------
        Gdata: dict
            A dictionary containing Gamera data, ready for use in lompeOSSE
        """

        Gdata = {}

        # Open HDF5 file and load Gamera data
        with h5py.File(fn, "r") as f:
            Gdata['X'] = f['X'][:]
            Gdata['Y'] = f['Y'][:]
            for step in f['Step#%d' % nstep].keys():
                Gdata[step] = f['Step#%d' % nstep][step][:]


        # Filter data by hemisphere
        h = hem.upper()
        hemi_Gdata = {}

        for key in Gdata.keys():
            key_lower = key.lower()

            if h == "NORTH":
                if "north" in key_lower:
                    new_key = key.replace("NORTH", "").strip()
                    hemi_Gdata[new_key] = Gdata[key]

                elif "south" not in key_lower:
                    hemi_Gdata[key] = Gdata[key]

            elif h == "SOUTH":
                if "south" in key_lower:
                    new_key = key.replace("SOUTH", "").strip()
                    hemi_Gdata[new_key] = Gdata[key]

                elif "north" not in key_lower:
                    hemi_Gdata[key] = Gdata[key]


        # Update Gdata with hemisphere-filtered values
        Gdata.clear()
        Gdata.update(hemi_Gdata)

        # Convert Cartesian (X, Y) to spherical coordinates (R, THETA, PHI)
        X = Gdata['X'] # TODO in Earth's radius?
        Y = Gdata['Y']
        theta = np.arcsin(np.sqrt(X**2 + Y**2)) # colatitude in radians (??)
        phi = np.arctan2(Y, X) # azimuthal angle in radians (theta column in remix file)

        # TODO remove this when I get lompeosse to deal with it instead
        phi_offset = mlt_off*15 # offset in degrees
        phi = phi + phi_offset

        # # Normalize azimuthal angles to [0 - 2pi] # TODO check if useful with Kalle 
        # phi[phi < 0] = phi[phi < 0] + 2*np.pi
        # phi[:, 0] -= 2 * np.pi  # Adjust first column
        
        Gdata['THETA'] = theta
        Gdata['PHI'] = phi

        # Correct r, theta and phi to match other variables in Gamera data file
        # Compute grid-centered spherical coordinates
        theta_trim = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1] /2 # averaged over theta respective grid directions
        phi_trim   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :] /2 # averaged over phi respective grid directions

        # Compute magnetic latitude, longitude, and local time (USEFUL??)
        mlatG = 90 - np.rad2deg(theta_trim) # in degrees
        if hem == 'SOUTH': mlatG = (-1)*mlatG
        
        Gdata['mlat'] = mlatG
        mlt = phi_trim * (12/np.pi) +6 # in hours NOTE The +6 shift aligns MLT midnight 
                                                        # with the nightside as defined in 
                                                        # the original Gamera dataset.
        # mlt = (phi_trim * (12 / np.pi) + 6) % 24

        # Remove data below min_lat
        min_lat = 20  # Should be at least 11 deg
        mask = np.abs(Gdata['mlat']) > min_lat

        for key in Gdata.keys():
            if Gdata[key].shape == Gdata['mlat'].shape:  
                Gdata[key] = np.where(mask, Gdata[key], np.nan) # Apply NaN to out-of-bounds data

        if np.abs(Gdata["mlat"]).min() < min_lat:
            print('Gdata["mlat"].min(): ', np.abs(Gdata['mlat']).min(), 'degrees')
            print(f"Low latitude Gamera data (< {min_lat} deg) has been discarded")
        
        # Convert from dipole (magnetic) to geocentric (geographic) coordinates using apexpy
        dp = dipole.Dipole(t.year)
        mlon = dp.mlt2mlon(mlt, t)
        Gdata['glat'], Gdata['glon'], _ = self.apex.apex2geo(Gdata['mlat'], mlon, self.refh)

        return Gdata
    

    def get_conductance_functions(self, grid):
           
        """
        Generates interpolation functions for Gamera Hall and Pedersen conductances.

        Returns:
        --------
        tuple of functions
            - SPfunc(glon, glat): Function that interpolates Gamera Pedersen conductance at given (lon, lat).
            - SHfunc(glon, glat): Function that interpolates Gamera Hall conductance at given (lon, lat).
        
            
        Notes:
        ------        
        - It seems that a too large grid introduces NaNs in the conductances. TODO EXPLAIN WHY?
        """

        # Extract conductances from Gamera dataset
        SPG = self.gamera_data['Pedersen conductance']
        SHG = self.gamera_data['Hall conductance']

        # Interpolate Gamera conductances to lon, lat
        def SPfunc(lon,lat):
            ''' Gamera Pedersen conductance '''
            SP = self.interp_to_measurements(grid, lon, lat, var = SPG)
            return SP

        def SHfunc(lon,lat):
            ''' Gamera Hall conductance '''
            SH = self.interp_to_measurements(grid, lon, lat, var = SHG)
            return SH
        
        return SHfunc, SPfunc
    

    def get_E(self, grid, ds):

        """
        Compute the Gamera electric field at Gamera grid points, then transform it into geodetic coordinates 
        and finally interpolate at measurement glon, glat. 

        TODO write something like: This is largely based on the Kaipy module but also integrate the conversion from magnetic dipole to geographic coordinates.

        Returns:
        --------
        tuple: (E_east, E_north) 
            Electric field components in the geographic eastward and northward directions at measurement locations lon/lat.
        """

        #-----------
        # First, compute the Gamera electric field 

        # Extract Gamera grid coordinates
        x = self.gamera_data['X']
        theta = self.gamera_data['THETA']
        phi = self.gamera_data['PHI']

        # Extract Gamera electric potential (in kV)
        Psi = self.gamera_data['Potential']

        # Earth ionosphere reference radius (in km)
        ri = RI
        
        # Initialize interpolated potential (Psi Ψ) array
        Psi_c = np.zeros(x.shape)
        Psi_c[1:-1,1:-1] = 0.25 * (Psi[1:,1:] + Psi[:-1,1:] + Psi[1:,:-1] + Psi[:-1,:-1])  # Average surrounding cell values

        # Handle periodic boundary conditions in longitude
        Psi_c[1:-1,0]  = 0.25 * (Psi[1:,0] + Psi[:-1,0] + Psi[1:,-1] + Psi[:-1,-1])
        Psi_c[1:-1,-1] = Psi_c[1:-1,0] # Ensure consistency at the last column

        # Handle the pole boundary condition using mean potential
        Psi_pole = Psi[0,:].mean() # Average potential at the pole
        Psi_c[0,1:-1] = 0.25 * (2.*Psi_pole + Psi[0,:-1] + Psi[0,1:]) 
        Psi_c[0,0]    = 0.25 * (2.*Psi_pole + Psi[0,-1] + Psi[0,0]) 
        Psi_c[0,-1]   = 0.25 * (2.*Psi_pole + Psi[0,-1] + Psi[0,0])		

        # Handle lower latitude boundary with linear extrapolation
        Psi_c[-1,:] = 2 * Psi_c[-2,:] - Psi_c[-3,:]

        # Compute meridional (north-south) electric field component (E_theta)
        tmp    = 0.5 * (Psi_c[:,1:] + Psi_c[:,:-1])  
        dPsi   = tmp[1:,:] - tmp[:-1,:]
        tmp    = 0.5 * (theta[:,1:] + theta[:,:-1])
        dtheta = tmp[1:,:] - tmp[:-1,:]
        etheta = (-1)*dPsi/dtheta/ri  # E = -∇Ψ (V/m)

        # Compute zonal (east-west) electric field component (E_phi)
        tmp    = 0.5 * (Psi_c[1:,:] + Psi_c[:-1,:]) 
        dPsi   = tmp[:,1:] - tmp[:,:-1]
        tmp    = 0.5 * (phi[1:,:] + phi[:-1,:])
        dphi   = tmp[:,1:] - tmp[:,:-1]
        tc = 0.25 * (theta[:-1,:-1] + theta[1:,:-1] + theta[:-1,1:] + theta[1:,1:])
        ephi = (-1)*dPsi/dphi/np.sin(tc)/ri  # E = -grad Ψ (V/m)

        # TODO use these etheta and ephi in inverse code?

        #-----------
        # Then, convert from magnetic dipole coordinate to geographic 

        # Convert Gamera electric field from magnetic dipole to geocentric coordinates
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(self.gamera_data['glat'].flatten(), self.gamera_data['glon'].flatten(), height = grid.R*1e-3 - RE, coords = 'geo')

        # Compute magnetic field inclination
        mlat = self.gamera_data['mlat'].flatten()
        sinIm = 2 * np.sin(np.deg2rad(mlat)) / np.sqrt(4 - 3 * np.cos(np.deg2rad(mlat))**2) 

        # Convert to geographic geodetic coordinates using the d1 and d2 base vectors
        E_mag_east, E_mag_north = ephi, -etheta # Convert to east-north components
        E_geo_east, E_geo_north, E_geo_up = E_mag_east.flatten() * d1 - (E_mag_north.flatten()/sinIm) * d2 #TODO OK?
        
        #-----------
        # Finally, interpolate Gamera electric field (geographic components) from the Gamera grid to the measurement locations (glon, glat)
        
        E_east, E_north = self.interp_to_measurements(grid, ds.coords['lon'], ds.coords['lat'], vec=(E_geo_east, E_geo_north))

        return E_east, E_north # East, north Gamera electric field in geocentric coordinates at measurement locations


    def get_Bigrf(self, ds):

        """
        Compute the geodetic IGRF magnetic field components at the interpolated grid points.

        Returns:
        --------
        B0 : (3, N) ndarray
            Magnetic field components in testla (T), with:
            - B0[0] = Br  (radial component)
            - B0[1] = Bph (azimuthal/eastward component)
            - B0[2] = Bth (polar/southward component)
        """
        
        # Compute the IGRF main field components in nT
        Be, Bn, Bup = np.array(igrf(ds.coords['lon'], ds.coords['lat'], self.refh, self.time)) * 1e-9 # igrf returns east, north, up components in nT 

        return np.vstack((Bup, Be, -Bn)) # radial, east (phi), south (theta)


    def get_V(self, grid, ds):

        """
        Compute the ExB drift velocity (east and north components) 
        from the Gamera electric field and the IGRF magnetic field.

        Returns:
        --------
        tuple: (V_east, V_north)
            Plasma drift velocity (in m/s) in the eastward and northward directions
        """
        
        # Electric field
        E_east, E_north = self.get_E(grid, ds)
        Eph = E_east # azimuthal
        Eth = -E_north # polar

        # Geomagnetic field components and magnitude
        Br, Bph, Bth = self.get_Bigrf(ds)
        B2 = np.sqrt(Br**2 + Bth**2 + Bph**2)

        # Radial electric field component using divergence-free assumption
        Er = -(Eth * Bth + Eph * Bph) / Br

        # E×B drift velocity components in (r, ph, th)
        Vr  = (Eth * Bph - Eph * Bth) / B2**2
        Vph = (Er * Bth - Eth * Br)   / B2**2
        Vth = (Eph * Br - Er * Bph)   / B2**2

        # Convert back to east/north
        V_east = Vph
        V_north = -Vth

        return V_east, V_north


    def interp_to_usergrid(self, grid, var): 

        """
        Interpolates a variable from the Gamera grid to the Lompe grid.

        TODO write something like: this is useful for the validation metrics, 
        to compare the lompeOSSE outputs to original gamera variables 

        Parameters:
        -----------
        var : ndarray (scalar)
            The variable (e.g., conductance, electric field, or velocity) defined on the Gamera grid.

        Returns:
        --------
        varinterp : ndarray
            The interpolated variable on the Lompe grid, reshaped to match the grid dimensions.
        
        """

        # Identify Gamera grid points that fall inside the Lompe cubed sphere grid
        iii = grid.ingrid(self.gamera_data['glon'], self.gamera_data['glat'], ext_factor = 1.5)

        # Project valid Gamera coordinates (glon, glat) to cubed sphere coordinates (xi, eta)
        xiG, etaG = grid.projection.geo2cube(self.gamera_data['glon'][iii], self.gamera_data['glat'][iii]) 
        
        # Extract the xi, eta coordinates of the Lompe grid
        xi, eta = grid.xi, grid.eta

        # Interpolate Gamera variable values to user-defined grid
        varinterp = griddata((xiG,etaG), var[iii], (xi.flatten(), eta.flatten()))

        # Reshape the interpolated data to match the original 2D grid structure
        varinterp = varinterp.reshape(grid.shape)

        return varinterp
    

    def interp_to_measurements(self, grid, lon, lat, var=None, vec=None):
        """
        Interpolate a scalar or vector field from the Gamera grid to measurement locations.

        Parameters
        ----------
        grid : lompe.Grid
            Target cubed-sphere grid.
        lon, lat : ndarray
            Measurement coordinates (deg).
        var : ndarray, optional
            Scalar Gamera variable to interpolate (e.g. conductance).
        vec : tuple(ndarray, ndarray), optional
            Vector Gamera field given as (E_east, E_north) in geographic coordinates.

        Returns
        -------
        ndarray or tuple(ndarray, ndarray)
            - If scalar: var_interp
            - If vector A: (A_east_interp, A_north_interp)
        """

        # --- Project measurement coords to cubed sphere
        xi, eta = grid.projection.geo2cube(lon.flatten(), lat.flatten(), set_points_off_cube_to_nan=True) 

        # ---------------------------------------------------------------------
        # SCALAR INTERPOLATION
        # ---------------------------------------------------------------------
        if var is not None and vec is None:

            # --- Keep Gamera points inside grid only
            iii = grid.ingrid(self.gamera_data['glon'], self.gamera_data['glat'], ext_factor = 1.5) # requires lon, lat in degrees

            # Project Gamera coords glon/glat to cubed sphere
            xiG, etaG = grid.projection.geo2cube(self.gamera_data['glon'][iii], self.gamera_data['glat'][iii])

            # Interpolate from Gamera locations xiG/etaG to measurement locations xi/eta
            var_interp = griddata((xiG, etaG), var[iii], (xi, eta), method="linear")
            return var_interp.reshape(lon.shape)

        # ---------------------------------------------------------------------
        # VECTOR INTERPOLATION (east, north)
        # ---------------------------------------------------------------------
        if vec is not None and var is None:
            A_east, A_north = vec

            # --- Keep Gamera points inside grid only
            iii = grid.ingrid(self.gamera_data['glon'].flatten(), self.gamera_data['glat'].flatten()) # requires lon, lat in degrees

            # Project Gamera vector and coords to cubed sphere
            xiG, etaG, A_xi, A_eta = grid.projection.vector_cube_projection(A_east[iii], A_north[iii], self.gamera_data['glon'].flatten()[iii], self.gamera_data['glat'].flatten()[iii])

            # Interpolate both vector components
            A_xi_interp  = griddata((xiG, etaG), A_xi,  (xi, eta), method="linear")
            A_eta_interp = griddata((xiG, etaG), A_eta, (xi, eta), method="linear")

            # Convert back to geographic coordinates
            _, _, A_east_interp, A_north_interp = grid.projection.vector_cube_to_geo(A_xi_interp, A_eta_interp, xi, eta)

            return (A_east_interp.reshape(lon.shape),
                    A_north_interp.reshape(lon.shape))

        raise ValueError("Provide either var=<scalar> or vec=(Ee, En).")

