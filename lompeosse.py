import h5py
import os
import numpy as np
import apexpy
from scipy.interpolate import griddata
from ppigrf import igrf
import copy
import dipole # github.com/klaundal/dipole
import lompe
from magnetic_field_utils import get_B

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
        path = os.path.abspath(os.path.dirname(__file__))
        self.datapath = os.path.join(path, 'data/Gamera_data.h5') 

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


class LompeOSSE(object):

    """ 
    OSSE Model class 

    The OSSE model is a copy of user-defined Lompe model, but its datasets are replaced with  
    synthetic data from Gamera simulations, including Gamera-derived conductances. 
    All other model properties remain unchanged.

    """

    def __init__(self, input_model, gamera_object, mlt_offset=0):

        """
        Initializes the Lompe-OSSE electric field model.

        This class creates a copy of input_model, a Lompe object, and replaces the data 
        in its datasets with synthetic data from Gamera simulations.

        The synthetic data is extracted from a series of simulation snapshots at the 
        coordinates of the original datasets (measurement locations). 
        

        Example:
        -------
        grid = cs.CSgrid(*gridparams)

        input_model = lompe.Emodel(grid, (Hall_function, Pedersen_function))
        input_model.add_data(Efield_dataset, ground_B_dataset, etc...)

        synthetic_model = lompeOSSE(model, Gstep=1, epoch=epoch).synthetic_model

        synthetic_model.run_inversion()

        lompeplot(synthetic_model, include_data = True)

        
        Parameters:
        -------
        input_model: lompe object (user-defined with Emodel)
            The reference Lompe model containing the original datasets to be replaced with Gamera data.

        Gstep: int
            The snapshot index from the set of Gamera simulation snapshots.

        mlt_off: int, optional, default=0
            MLT offset. Rotates the final map by shifting the MLT coordinate system. 
            This allows sampling from a different MLT sector in the Gamera simulation.

        hem: str, optional, default='NORTH'
            Hemisphere indicator, either 'NORTH' or 'SOUTH' # USEFUL???

        epoch: float, optional, default=2015.0 
            Decimal year used for IGRF calculations AND OTHER STUFF...

        refh : float, optional, default=120
            Reference height in km for apex coordinates, the field lines are mapped to this height.

            
        Returns:
        -------
        A new lompe object with the same properties as input_model, but with synthetic Gamera data.
        """

        self.Gamera_object = gamera_object
        self.apex = self.Gamera_object.apex
        self.gamera_data = self.Gamera_object.gamera_data
        self.Gstep = self.Gamera_object.timestep

        self.mlt_off = mlt_offset

        self._input_model = input_model

        # Get OSSE model
        self.synthetic_model = self.make_OSSE_model()


    def make_OSSE_model(self):

        """
        Creates a synthetic OSSE model by replacing real observational datasets 
        with corresponding Gamera-generated datasets.

        This function:
        1. Filters the original model's datasets to ensure they are within the defined grid.
        2. Creates a copy of the filtered model.
        3. Iterates through available datasets and replaces known datatypes 
        (e.g., convection, efield, etc...) with synthetic data from Gamera simulations.
        4. Computes conductance functions from Gamera data.
        5. Clears the model and reintroduces processed Gamera datasets.

        Returns:
        --------
        lompe.Emodel object
            A copy of the original model but with synthetic Gamera data replacing real observations.
        """

        # Ensure input datasets are inside the user grid
        self.filter_datasets_by_grid()

        # Make a copy of input model
        print('\n Initializing synthetic model...')
        synthetic_model = copy.copy(self._input_model)
        grid = synthetic_model.grid_J

        print('\n Scanning user datasets and searching for corresponding Gamera data...')
        # Map known datatypes to their processing functions
        datatype_processors = {'convection': self.extract_synth_convection,
                               'efield': self.extract_synth_efield,
                               'space_mag_full': self.extract_synth_bfield,
                               'space_mag_fac': self.extract_synth_bfield, 
                               'ground_mag': self.extract_synth_bfield}

        # Replace datasets in model by Gamera datasets        
        processed_data = {}
        for datatype, dataset_list in synthetic_model.data.items():

            if not dataset_list:
                print(f'{datatype} dataset not found')
                continue

            processed_data[datatype] = []  # Store processed datasets for this datatype
            
            for ds in dataset_list:
                if datatype in datatype_processors:
                    print(f'{datatype} dataset found...')
                    gamera_ds = datatype_processors[datatype](grid, ds)
                    processed_data[datatype].append(gamera_ds)
                else:
                    print(f"Warning: No processing function for datatype '{datatype}'.")

        # Gamera conductances
        SHfunc, SPfunc = self.Gamera_object.get_conductance_functions(grid)
        print('\n Gamera conductances extracted')

        # Reset model (delete datasets and clear model vectors)
        print('\n Clearing input model...')
        print('Adding Gamera conductances')
        synthetic_model.clear_model(Hall_Pedersen_conductance = (SHfunc, SPfunc))
        
        # Add synthetic datasets to synthetic_model
        print('Adding Gamera datasets')
        for dataset_list in processed_data.values():
            for gamera_ds in dataset_list:
                synthetic_model.add_data(gamera_ds)
        
        print('\n ...Synthetic model generated')

        return synthetic_model


    # def filter_datasets_by_grid(model, grid):
    def filter_datasets_by_grid(self):

        """
        Filters the datasets in the original model (input_model) to retain only data points within the model grid.

        Returns:
        --------
        lompe.Emodel object
            The original model with its datasets filtered to include only points inside the grid.
        """
        
        # print('Shape before filtering: ', self._input_model.data['convection'][0].values.shape)

        for datatype, dataset_list in self._input_model.data.items():
            if not dataset_list:
                continue
            
            valid_list = []  # Store filtered datasets
            for ds in dataset_list:
                lon, lat = ds.coords['lon'], ds.coords['lat']
                indices = np.where(self._input_model.grid_J.ingrid(lon, lat))[0]
                filtered_ds = ds.subset(indices)
                valid_list.append(filtered_ds)

            self._input_model.data[datatype] = valid_list  # Replace original datasets with filtered version
        # print('Shape after filtering: ', self._input_model.data['convection'][0].values.shape)

        return self._input_model
    

    def extract_synth_convection(self, grid, ds):

        """
        Generates a synthetic convection dataset for synthetic_model integration.

        
        Parameters:
        -----------
        ds: lompe.Data object
            The original dataset containing measurement coordinates and line-of-sight (LOS) unit vectors.

        stacked_coords: ndarray
            A (2, N) array containing the dataset's geographic coordinates (longitude, latitude).


        Returns:
        --------
        lompe.Data object
            A synthetic convection dataset with Gamera-derived LOS velocities.
        """

        Ve, Vn = self.Gamera_object.get_V(grid, ds)
        print('Gamera convection data extracted')

        # Project Gamera velocity components onto the dataset's LOS direction
        vlos = Ve * ds.los[0] + Vn * ds.los[1]

        return lompe.Data(vlos, np.vstack((ds.coords['lon'], ds.coords['lat'])), LOS= ds.los, datatype='convection', iweight=ds.iweight, error=ds.error)


    def extract_synth_efield(self, grid, ds):  

        """
        Generates a synthetic electric field dataset for synthetic_model integration.
        

        Parameters:
        -----------
        ds: lompe.Data object
            The original dataset containing measurement coordinates.
        
        stacked_coords: ndarray
            A (2, N) array containing the dataset's geographic coordinates (longitude, latitude).

            
        Returns:
        --------
        lompe.Data object
            A synthetic electric field dataset with Gamera-derived values.
        """

        Ee, En = self.Gamera_object.get_E(grid, ds)
        print('Gamera electric field data extracted')

        E_values = np.vstack((Ee.flatten(), En.flatten()))

        return lompe.Data(E_values, np.vstack((ds.coords['lon'], ds.coords['lat'])), datatype='Efield', iweight=ds.iweight, error=ds.error)


    def extract_synth_bfield(self, grid, ds):

        """
        Generates a synthetic magnetic field dataset for synthetic_model integration.


        Parameters:
        -----------
        ds: lompe.Data object
            The original dataset containing measurement coordinates.
        
        stacked_coords: ndarray
            A (2, N) array containing the dataset's geographic coordinates (longitude, latitude).

        r: float
            With RI the ionosphere radius; r < RI is considered internal, r > RI is considered external relative to the ionosphere. 
            Ground magnetic field is obtained for r = RE.

        no_df_current: bool
            Flag indicating whether to exclude horizontal divergence-free current contributions. 
            Set to True for 'space_mag_fac' data types (e.g., Iridium) and False for 'space_mag_full' and 'ground_mag'.

        Returns:
        --------
        lompe.Data object
            A synthetic magnetic field dataset with Gamera-derived values.
        """

        if ds.datatype == 'ground_mag': r = np.full_like(ds.coords['lon'], RE*1e3) # assume perfectly circular Earth
        else: r = ds.coords['r']

        if ds.datatype == "space_mag_fac": no_df_current=True 
        else: no_df_current=False

        # Height of the ionosphere used for magnetic field calculations
        height = r*1e-3-RE

        # Convert measurement geocentric coordinates to magnetic dipole coordinates (Gamera) 
        mlat, mlon = self.apex.geo2apex(ds.coords['lat'], ds.coords['lon'], height) #lat, lon, height of the data points

        # Calculate magnetic field at measurement coordinates r (in meters), theta, phi
        Br, Bth, Bph = get_B(r, 90 - mlat, mlon + self.mlt_off*15, self.Gstep, no_df_current=no_df_current) 
        print(f'Gamera {ds.datatype} data extracted')

        # then convert to geo

        # Compute APEX base vectors at given geographic coordinates and heights
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(ds.coords['lat'], ds.coords['lon'], height=height, coords = 'geo')
        
        # Convert magnetic field in magnetic dipole coordinates to geographic coords
        B_geo_east, B_geo_north = Bph*f1 - Bth*f2
        B_geo_up = Br

        # Lompe requires east, north, up components
        Benu = np.vstack((B_geo_east, B_geo_north, B_geo_up))

        # TODO what should r be here?? is it r or height? 
        return lompe.Data(Benu* 1e-9, np.vstack((ds.coords['lon'], ds.coords['lat'], r)), datatype=ds.datatype, iweight=ds.iweight, error=ds.error)
    
