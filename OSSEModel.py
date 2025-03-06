""" OSSE Model clas """

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

RE = 6371.2 # Earth radius in kilometers

class lompeOSSE():
    def __init__(self, real_model, Gstep, hem='NORTH', epoch=2015., refh=120):

        """
        Initializes the Lompe-OSSE electric field model.

        This class creates a copy of real_model, a Lompe object, and replaces the data 
        in its datasets with synthetic data from Gamera simulations.

        The synthetic data is extracted from a series of simulation snapshots at the 
        coordinates of the original datasets. 
        

        Example:
        -------
        grid = cs.CSgrid(*gridparams)

        real_model = lompe.Emodel(grid, (Hall_function, Pedersen_function))
        real_model.add_data(Efield_dataset, ground_B_dataset, etc...)

        osse_model = lompeOSSE(model, Gstep=1, epoch=epoch).osse_model

        osse_model.run_inversion()

        lompeplot(osse_model, include_data = True) 

        
        Parameters:
        -------
        real_model: lompe object (user-defined with Emodel)
            The reference Lompe model containing the original datasets to be replaced with Gamera data.

        Gstep: int
            The snapshot index from the set of Gamera simulation snapshots.

        hem: str, optional, default='NORTH'
            Hemisphere indicator, either 'NORTH' or 'SOUTH' # USEFUL???

        epoch: float, optional, default=2015.0 
            Decimal year used for IGRF calculations AND OTHER STUFF...

        refh : float, optional, default=120
            Reference height in km for apex coordinates, the field lines are mapped to this height.

            
        Returns:
        -------
        A new lompe object with the same properties as real_model, but with synthetic Gamera data.
        """

        # Call the Emodel constructor
        # super().__init__(real_model.grid_J, 
        #                 #  real_model.hall_conductance(real_model.grid_J.lon, real_model.grid_J.lat), 
        #                 #  real_model.pedersen_conductance(real_model.grid_J.lon, real_model.grid_J.lat),
        #                 (lambda lon, lat: real_model.hall_conductance(lon, lat), 
        #                 lambda lon, lat: real_model.pedersen_conductance(lon, lat)),
        #                  real_model.epoch,
        #                  real_model.dipole, 
        #                  real_model.perfect_conductor_radius)

        self.real_model = real_model
        self.Gstep = Gstep
        self.hem = hem # useful?
        self.epoch = epoch # useful?
        self.t = yearfrac_to_datetime([self.epoch])[0]
        self.refh = 120
        self.apex = apexpy.Apex(self.t, self.refh) # OK?


        # Load GAMERA data
        self.mixFile = '/Users/margot/Downloads/msphere.mix.h5'  # update path--> replace by datafile containing smthg like 10 snapshots
        self.gamera_data = self.get_Gdata() 


        # Get OSSE model
        self.osse_model = self.make_OSSE_model()

        # self.__dict__ = self.make_OSSE_model().__dict__

    
    # def make_OSSE_model(self, real_model, Gstep, epoch=2015.):
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

        # Ensure datasets are inside the grid
        ingrid_model = self.filter_datasets_by_grid() # self.real_model, self.real_model.grid_J

        # Make a copy of the real dataset
        self.osse_model = copy.copy(ingrid_model)
        self.grid = self.osse_model.grid_J

        # Map known datatypes to their processing functions
        datatype_processors = {'convection': self.Gprocess_convection,
                               'efield': self.Gprocess_efield}
        # ADD MORE DATATYPES AND PROCESSING FUNCTIONS

        # Replace datasets in model by Gamera datasets        
        processed_data = {}
        for datatype, dataset_list in self.osse_model.data.items():

            if not dataset_list:
                print(f'{datatype} dataset is empty')
                continue

            processed_data[datatype] = []  # Store processed datasets for this datatype
            for ds in dataset_list:
                self.coords = ds.coords # glon, glat
                stacked_coords = np.vstack((self.coords['lon'], self.coords['lat']))

                if datatype in datatype_processors:
                    gamera_ds = datatype_processors[datatype](ds, stacked_coords)
                    processed_data[datatype].append(gamera_ds)

                else:
                    print(f"Warning: No processing function for datatype '{datatype}'.")

            # Gamera conductances
            SHfunc, SPfunc = self.get_conductance_functions() # self.grid, self.gamera_data

            # Reset model (delete datasets and clear model vectors)
            self.osse_model.clear_model(Hall_Pedersen_conductance = (SHfunc, SPfunc))
            
            # Add synthetic datasets to osse_model
            for dataset_list in processed_data.values():
                for gamera_ds in dataset_list:
                    self.osse_model.add_data(gamera_ds)

        return self.osse_model


    # def filter_datasets_by_grid(model, grid):
    def filter_datasets_by_grid(self):

        """
        Filters the datasets in the original model (real_model) to retain only data points within the model grid.

        Returns:
        --------
        lompe.Emodel object
            The original model with its datasets filtered to include only points inside the grid.
        """
        
        # print('Shape before filtering: ', self.real_model.data['convection'][0].values.shape)

        for datatype, dataset_list in self.real_model.data.items():
            if not dataset_list:
                continue
            
            valid_list = []  # Store filtered datasets
            for ds in dataset_list:
                lon, lat = ds.coords['lon'], ds.coords['lat']
                indices = np.where(self.real_model.grid_J.ingrid(lon, lat))[0]
                filtered_ds = ds.subset(indices)
                valid_list.append(filtered_ds)

            self.real_model.data[datatype] = valid_list  # Replace original datasets with filtered version
        
        # print('Shape after filtering: ', self.real_model.data['convection'][0].values.shape)

        return self.real_model
    

    def Gprocess_convection(self, ds, stacked_coords):

        """
        Generates a synthetic convection dataset for osse_model integration.

        
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
        
        Ve, Vn = self.get_V()
        print('Gamera convection data extracted')

        # Project Gamera velocity components onto the (real) dataset's LOS direction
        vlos = Ve * ds.los[0] + Vn * ds.los[1]

        return lompe.Data(vlos, stacked_coords, LOS= ds.los, datatype='convection', iweight=1.0, error=100)


    def Gprocess_efield(self, ds, stacked_coords):

        """
        Generates a synthetic electric field dataset for osse_model integration.
        

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

        Ee, En = self.get_E()
        print('Gamera electric field data extracted')

        E_values = np.vstack((Ee.flatten(), En.flatten()))

        return lompe.Data(E_values, stacked_coords, datatype='Efield', iweight=1.0, error=1e-3)


    # def Gprocess_Bfield(self, ds, stacked_coords):

    #     """
    #     Generates a synthetic magnetic field dataset for osse_model integration.


    #     Parameters:
    #     -----------
    #     ds: lompe.Data object
    #         The original dataset containing measurement coordinates.
        
    #     stacked_coords: ndarray
    #         A (2, N) array containing the dataset's geographic coordinates (longitude, latitude).


    #     Returns:
    #     --------
    #     lompe.Data object
    #         A synthetic magnetic field dataset with Gamera-derived values.
    #     """

    #     Be, Bn = self.get_B(test=False)
    #     print('Gamera magnetic field data extracted')

    #     B_values = np.vstack((Be.flatten(), Bn.flatten()))

    #     return lompe.Data(B_values, stacked_coords, datatype='', iweight=, error=)
    

    # def get_Gdata(mixFile, step, hem='north', epoch=2015.):
    def get_Gdata(self):

        """
        Reads Gamera data from an HDF5 file, extracts relevant variables based on 
        the specified hemisphere, convert magnetic dipole coordinates to geographic, 
        and processes them for use in Lompe analysis.

        Returns:
        --------
        Gdata: dict
            A dictionary containing Gamera data
        """

        Gdata = {}

        # Open HDF5 file and load Gamera data
        with h5py.File(self.mixFile, "r") as f:
            Gdata['X'] = f['X'][:]
            Gdata['Y'] = f['Y'][:]
            for h in f['Step#%d' % self.Gstep].keys():
                Gdata[h] = f['Step#%d' % self.Gstep][h][:]


        # Filter data by hemisphere
        h = self.hem.upper()
        hemi_Gdata = {}

        for key in Gdata.keys():
            if "north" in key.lower() and h == "NORTH":
                new_key = key.replace("NORTH", "").strip() 
                hemi_Gdata[new_key] = Gdata[key]

            elif "south" in key.lower() and h == "SOUTH":
                new_key = key.replace("SOUTH", "").strip()
                hemi_Gdata[new_key] = Gdata[key]

            # Keep non-hemisphere-specific variables
            elif "north" not in key and "south" not in key:
                hemi_Gdata[key] = Gdata[key]

        # Update Gdata with hemisphere-filtered values
        Gdata.clear()
        Gdata.update(hemi_Gdata)

        # Convert Cartesian (X, Y) to spherical coordinates (R, THETA, PHI)
        X = Gdata['X'] # in Earth's radius?
        Y = Gdata['Y']
        r = np.sqrt(X**2 + Y**2) # ???
        theta = np.arcsin(r) # colatitude in radians (??)
        phi = np.arctan2(Y, X) # azimuthal angle in radians (theta column in remix file)

        # Normalize azimuthal angles to [0 - 2pi]
        phi[phi < 0] = phi[phi < 0] + 2*np.pi
        phi[:, 0] -= 2 * np.pi  # Adjust first column
        
        Gdata['R'] = r*RE # in km??
        Gdata['THETA'] = theta
        Gdata['PHI'] = phi

        # Correct r, theta and phi to match other variables in GAMERA data file
        # Compute grid-centered spherical coordinates
        # r_trim     = (r[:-1, :-1] + r[:-1, 1:] + r[1:, :-1] + r[1:, 1:]) /4 # take the center of each grid cell by averaging values from adjacent points
        r_trim     = r[:-1, :-1] + np.diff(r, axis=0)[:, :-1]/2 + np.diff(r, axis=1)[:-1, :] /2 # averaged over all four corner points of each grid cell
        theta_trim = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1] /2 # averaged over theta respective grid directions
        phi_trim   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :] /2 # averaged over phi respective grid directions

        Gdata['r'] = r_trim*RE # in km??
        Gdata['theta'] = theta_trim
        Gdata['phi'] = phi_trim

        # Compute magnetic latitude, longitude, and local time (USEFUL??)
        mlatG = 90 - np.rad2deg(theta_trim) # in degrees
        mlonG = np.rad2deg(phi_trim) # in degrees
        mltG = phi_trim * (12/np.pi) # in hours
        mltOffset = 12 # offset (in hours) to center MLT at 0/24 in polar plots
        mltG += mltOffset 

        Gdata['mlat'] = mlatG
        Gdata['mlon'] = mlonG
        Gdata['mlt'] = mltG # (USEFUL??)

        # Apply latitude mask to remove data below min_lat
        min_lat = 20  # Should be at least 11 deg
        mask = Gdata['mlat'] > min_lat  # Boolean mask based on latitude

        for key in Gdata.keys():
            if Gdata[key].shape == Gdata['mlat'].shape:  
                Gdata[key] = np.where(mask, Gdata[key], np.nan) # Apply NaN to out-of-bounds data

        if Gdata["mlat"].min() < min_lat:
            print('Gdata["mlat"].min(): ', Gdata['mlat'].min(), 'degrees')
            print(f"Low latitude GAMERA data (< {min_lat} deg) has been discarded")
        
        # Convert from magnetic to geographic coordinates using apexpy
        self.glatG, self.glonG, _ = self.apex.apex2geo(Gdata['mlat'], Gdata['mlon'], self.refh)

        Gdata['glon'] = self.glonG
        Gdata['glat'] = self.glatG

        return Gdata
    

    # def get_conductance_functions(grid, gamera_data):
    def get_conductance_functions(self):
           
        """
        Generates interpolation functions for Hall and Pedersen conductances from Gamera data.

        
        Returns:
        --------
        tuple of functions
            - SPfunc(glon, glat): Function that interpolates Pedersen conductance at given (lon, lat).
            - SHfunc(glon, glat): Function that interpolates Hall conductance at given (lon, lat).
        
            
        Notes:
        ------        
        - It seems that a too large grid introduces NaNs in the conductances. EXPLAIN WHY?
        """

        # Extract conductances from GAMERA dataset
        SPG = self.gamera_data['Pedersen conductance']
        SHG = self.gamera_data['Hall conductance']

        # Interpolate to user cubed sphere grid
        SPinterp = self.interp2lompegrid(SPG) # self.grid, self.glonG, self.glatG, SPG
        SHinterp = self.interp2lompegrid(SHG)

        # # Ensures no NaN or invalid values remain in the interpolated data
        # SHinterp_clear = np.ma.masked_invalid(SHinterp)
        # SPinterp_clear = np.ma.masked_invalid(SPinterp)

        # checks whether all values in SPinterp and SHinterp are finite (i.e., not NaN, inf, or -inf)
        assert (np.sum(~np.isfinite(SPinterp)) + np.sum(~np.isfinite(SHinterp))) == 0, \
            f"There are NaNs in the conductances (SPinterp: {np.sum(~np.isfinite(SPinterp))}, SHinterp: {np.sum(~np.isfinite(SHinterp))}). This might be caused by a grid that is too large; try reducing grid size"

        # Functions to interpolate to any lon,lat (RectBivariateSpline creates a continuous interpolation function from a grid of values)
        SHfuncCS0 = RectBivariateSpline(self.grid.xi[0], self.grid.eta[:,0], SHinterp.T)
        SPfuncCS0 = RectBivariateSpline(self.grid.xi[0], self.grid.eta[:,0], SPinterp.T)
        SHfuncCS = lambda xi, eta : SHfuncCS0(xi, eta, grid = False)
        SPfuncCS = lambda xi, eta : SPfuncCS0(xi, eta, grid = False)

        def SPfunc(glon, glat):
            ''' Pedersen conductance on grid '''
            xit, etat = self.grid.projection.geo2cube(glon, glat)
            return SPfuncCS(xit, etat)

        def SHfunc(glon, glat):
            ''' Hall conductance on grid '''
            xit, etat = self.grid.projection.geo2cube(glon, glat)
            return SHfuncCS(xit, etat)
        
        return SPfunc, SHfunc
    

    # def get_E(Lgrid, user_coords, Gdata, time, test=False):
    def get_E(self):

        """
        Compute the electric field from Gamera potential data and transform it into geodetic coordinates.

        Returns:
        --------
        tuple: (Eph, -Eth) 
            Electric field components in the geographic eastward and northward directions.
        """

        # Extract Gamera grid coordinates
        x = self.gamera_data['X']
        theta = self.gamera_data['THETA']
        phi = self.gamera_data['PHI']

        # Extract Gamera electric potential
        Psi = self.gamera_data['Potential']

        # Earth ionosphere reference radius (in km)
        ri = 6.5e3 
        
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

        # Convert to east-north components
        self.EeG_mag = ephi # Eastward component
        self.EnG_mag = -etheta # Northward component

        # Convert electric field from Gamera (magnetic dipole) to geocentric coordinates
        self.EeG_geo, self.EnG_geo, _ = self.efield_gamera2geo() # self.EeG_mag, self.EnG_mag, self.mlonG, self.mlatG, self.rG, self.t  
        
        # Interpolate electric field to cubed sphere grid longitude and latitude 
        self.Ee, self.En, self.glon, self.glat = self.interp_efield_2geogrid() # self.grid, self.coords, self.glonG, self.glatG, self.EeG, self.EnG, test=test

        return self.Ee, self.En # East, north components


    def get_V(self):

        """
        Compute the plasma drift velocity components from the Gamera electric field and the IGRF magnetic field 
        using the E×B drift formula.

        Returns:
        --------
        tuple: (Ve, Vn)
            Plasma drift velocity (in m/s) in the eastward and northward directions
        """
        
        # Retrieve the electric field components in azimuthal (Eph) and polar (Eth) directions
        self.Ee, self.En = self.get_E()
        Eph, Eth = self.Ee, -self.En 

        # Retrieve the geomagnetic field components 
        B0 = self.get_Bigrf() # self.grid, self.glon, self.glat, self.t
        Br, Bph, Bth = B0[0], B0[1], B0[2]
        B = np.sqrt(Br**2 + Bth**2 + Bph**2) # total magnetic field strength

        # Compute radial electric field component (E_r) using the divergence-free assumption
        Er = -(Eth * Bth + Eph * Bph) / Br

        # Compute E × B drift velocity components
        Vr  = (Eth * Bph - Eph * Bth) / B**2
        Vph = (Er * Bth - Eth * Br)   / B**2
        Vth = (Eph * Br - Er * Bph)   / B**2

        # Store eastward and northward velocity components
        self.Ve, self.Vn = Vph, -Vth

        return self.Ve, self.Vn


    def get_B(): 

        """
        """

        return Br, Bph, Bth
    

    # def get_Bigrf(Lgrid, glon, glat, time):
    def get_Bigrf(self):

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
        
        # Set constant radius for all points in km (assuming surface of the Earth + altitude)
        r = np.full(self.glon.shape, self.grid.R*1e-3) # radius in km

        # Compute the IGRF main field components in nT
        Be_nT, Bn_nT, Bup_nT = igrf(self.glon, self.glat, self.refh, self.t) # returns radial, south, east in nT 
        # Br_nT, Bth_nT, Bph_nT = igrf_gc(r, 90 - self.glat, self.glon, self.t) 

        # Convert from nT to T
        Bn  = Bn_nT  * 1e-9
        Be  = Be_nT  * 1e-9
        Bup = Bup_nT * 1e-9

        # Convert to polar, azimuthal, radial components
        Bth = -Bn
        Bph = Be
        Br = Bup # is that right?

        # Br  = Br_nT  * 1e-9
        # Bth  = Bth_nT  * 1e-9
        # Bph = Bph_nT * 1e-9

        return np.vstack((Br, Bph, Bth)) # radial, east, south 


    # def efield_gamera2geo(E_mag_east, E_mag_north, mlon, mlat, height, time):
    def efield_gamera2geo(self):

        """
        Convert Gamera electric field components from magnetic dipole coordinates 
        to geographic geodetic coordinates (which are used in Lompe).

        Returns:
        --------
        tuple of ndarray: (E_geo_east, E_geo_north, E_geo_up)
            Electric field components in the geographic eastward, northward and upward (radial) directions.
        """

        # Compute APEX base vectors at given geographic coordinates and heights
        glonG, glatG, rG = self.glonG.flatten(), self.glatG.flatten(), self.gamera_data['r'].flatten()
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(glatG, glonG, height=rG, coords = 'geo')
        
        # Compute magnetic field inclination
        mlat = self.gamera_data['mlat'].flatten()
        sinIm = 2 * np.sin(np.deg2rad(mlat)) / np.sqrt(4 - 3 * np.cos(np.deg2rad(mlat))**2) 

        # Convert to geographic geodetic coordinates using the d1 and d2 base vectors
        E_mag_east, E_mag_north = self.EeG_mag.flatten(), self.EnG_mag.flatten()
        E_geo_east, E_geo_north, E_geo_up = E_mag_east * d1 - (E_mag_north/sinIm) * d2

        # h = hem.upper()
        # if h == 'NORTH':
        # elif h == 'SOUTH': 
        #     E_geo_east, E_geo_north, E_geo_up = E_mag_east * d1 + (E_mag_north/sinIm) * d2 # is that correct??

        return (E_geo_east.reshape(self.glonG.shape), 
                E_geo_north.reshape(self.glonG.shape), 
                E_geo_up.reshape(self.glonG.shape))


    # def interp_efield_2geogrid(self, EeG, EnG, test=False):
    def interp_efield_2geogrid(self):

        """
        Interpolate the Gamera electric field (east and north components) from the 
        Gamera grid to the user-defined cubed sphere grid used in Lompe.

        Returns:
        --------
        tuple of ndarray: (Ee_interp, En_interp, glon, glat)
            - Ee_interp: Interpolated electric field (eastward component) on the cubed sphere grid
            - En_interp: Interpolated electric field (northward component) on the cubed sphere grid
            - glon: Geographic longitude of the cubed sphere grid points
            - glat: Geographic latitude of the cubed sphere grid points

        """

        # Select GAMERA points inside the cubed sphere grid  
        iii = self.grid.ingrid(self.glonG, self.glatG)  # requires lonG/latG in degrees  

        # Project GAMERA electric field and coordinates onto the cubed sphere  
        self.projection = self.grid.projection 
        xiG, etaG, E_xi, E_eta = self.projection.vector_cube_projection(  
            self.EeG_geo[iii], self.EnG_geo[iii], self.glonG[iii], self.glatG[iii])

        # Convert user-defined geographic grid (lon, lat) to cubed sphere coordinates (xi, eta)
        xi, eta = self.projection.geo2cube(self.coords['lon'].flatten(), self.coords['lat'].flatten(), set_points_off_cube_to_nan=True)
        # xi, eta = grid.xi.flatten(), grid.eta.flatten()

        # Interpolate the electric field from Gamera grid points to the cubed sphere grid
        E_xi_interp = griddata((xiG, etaG), E_xi, (xi, eta), method='linear') # method ok?
        E_eta_interp = griddata((xiG, etaG), E_eta, (xi, eta), method='linear')

        # Convert interpolated electric field back to geographic coordinates
        glon, glat, Ee_interp, En_interp = self.projection.vector_cube_to_geo(E_xi_interp, E_eta_interp, xi, eta)

        return (Ee_interp.reshape(self.coords['lon'].shape), 
                En_interp.reshape(self.coords['lon'].shape), 
                glon.reshape(self.coords['lon'].shape), 
                glat.reshape(self.coords['lon'].shape))


    def interp2lompegrid(self, var):

        """
        Interpolates a variable from the Gamera grid to the Lompe grid.

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
        iii = self.grid.ingrid(self.glonG, self.glatG, ext_factor = 1.5)

        # Convert valid Gamera (glon, glat) coordinates to Lompe's cubed sphere coordinates (xi, eta)
        xiG, etaG = self.projection.geo2cube(self.glonG[iii],self.glatG[iii]) 
        
        # Extract the xi, eta coordinates of the Lompe grid
        xi, eta = self.grid.xi, self.grid.eta

        # Interpolate Gamera variable values to the Lompe grid
        varinterp = griddata((xiG,etaG), var[iii], (xi.flatten(), eta.flatten()))

        # Reshape the interpolated data to match the original 2D grid structure
        varinterp = varinterp.reshape(self.grid.shape)

        return varinterp