""" OSSE Model class 

The OSSE model is a copy of user-defined Lompe model, but its datasets are replaced with  
synthetic data from Gamera simulations, including Gamera-derived conductances. 
All other model properties remain unchanged.

"""

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
from magnetic_field_utils import get_B
import os
from functools import partial

RE = 6371.2 # Earth radius in km
RI = 6500 # Ionospheric radius in km (used in Gamera simulations)

class LompeOSSE(object):
    # def __new__(cls, user_model, Gstep, mlt_off=0, hem='NORTH', epoch=2015., refh=122):
    #     instance = super().__new__(cls)
    #     instance.__init__(user_model, Gstep, mlt_off, hem, epoch, refh)
    #     return instance.osse_model  # This ensures that when you instantiate lompeOSSE, it directly returns osse_model


    def __init__(self, user_model, nstep=0, mlt_off=0, epoch=2015.):

        """
        Initializes the Lompe-OSSE electric field model.

        This class creates a copy of user_model, a Lompe object, and replaces the data 
        in its datasets with synthetic data from Gamera simulations.

        The synthetic data is extracted from a series of simulation snapshots at the 
        coordinates of the original datasets (measurement locations). 
        

        Example:
        -------
        grid = cs.CSgrid(*gridparams)

        user_model = lompe.Emodel(grid, (Hall_function, Pedersen_function))
        user_model.add_data(Efield_dataset, ground_B_dataset, etc...)

        osse_model = lompeOSSE(model, Gstep=1, epoch=epoch).osse_model

        osse_model.run_inversion()

        lompeplot(osse_model, include_data = True)

        
        Parameters:
        -------
        user_model: lompe object (user-defined with Emodel)
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
        A new lompe object with the same properties as user_model, but with synthetic Gamera data.
        """

        # # Call the Emodel constructor
        # super().__init__(user_model.grid_J, 
        #                 #  user_model.hall_conductance(user_model.grid_J.lon, user_model.grid_J.lat), 
        #                 #  user_model.pedersen_conductance(user_model.grid_J.lon, user_model.grid_J.lat),
        #                 (lambda lon, lat: user_model.hall_conductance(lon, lat), 
        #                 lambda lon, lat: user_model.pedersen_conductance(lon, lat)),
        #                  user_model.epoch,
        #                  user_model.dipole, 
        #                  user_model.perfect_conductor_radius)

        self._input_model = user_model
        self.Gstep = nstep
        print(f'Step#{self.Gstep}')

        hem = 'NORTH' if self._input_model.lat_J.min() > 0 else 'SOUTH'
        print('Hemisphere:', hem)

        self.epoch = epoch
        self.t = yearfrac_to_datetime([self.epoch])[0]
        self.refh = RI-RE # in km
        self.apex = apexpy.Apex(self.t, self.refh) # OK?


        self.osse_model = copy.copy(self._input_model)
        self.grid = self.osse_model.grid_J
        self.projection = self.grid.projection 


        # Build path to Gamera data file
        path = os.path.abspath(os.path.dirname(__file__))
        self.datapath = os.path.join(path, 'data/Gamera_data.h5') 
        print(self.datapath)

        # Check if data file exists
        if not os.path.exists(self.datapath):
            raise FileNotFoundError(
                f"Required file not found: {self.datapath}\n"
                "Please download it (https://zenodo.org/records/16882035) and place it in the 'data' folder." # TODO add zenodo link
            )

        # Load Gamera data
        self.gamera_data = self.get_Gdata(hem, mlt_off) 


        # Get OSSE model
        # self.osse_model = self.make_OSSE_model()
        self.make_OSSE_model()


    # def make_OSSE_model(self, user_model, Gstep, epoch=2015.):
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
        self.filter_datasets_by_grid() # self.user_model, self.user_model.grid_J

        # Make a copy of input model
        print('\n Initializing OSSE model...')
        self.osse_model = copy.copy(self._input_model)
        self.grid = self.osse_model.grid_J
        self.grid_E = self.osse_model.grid_E

        print('\n Scanning user datasets and searching for corresponding Gamera data...')
        # Map known datatypes to their processing functions
        datatype_processors = {'convection': self.Gprocess_convection,
                               'efield': self.Gprocess_efield, #TODO useful??
                               'space_mag_full': partial(self.Gprocess_Bfield, datatype='space_mag_full'),
                               'space_mag_fac': partial(self.Gprocess_Bfield, datatype='space_mag_fac', no_df_current=True), 
                               'ground_mag': partial(self.Gprocess_Bfield, datatype='ground_mag', r=RE)} # assume perfectly circular Earth # or r = RI - 110 in km ?

        # Replace datasets in model by Gamera datasets        
        processed_data = {}
        for datatype, dataset_list in self.osse_model.data.items():

            if not dataset_list:
                print(f'{datatype} dataset is empty')
                continue

            processed_data[datatype] = []  # Store processed datasets for this datatype
            for ds in dataset_list:
                self.coords = ds.coords # geocentric lon, lat
                self.stacked_coords = np.vstack((self.coords['lon'], self.coords['lat']))

                if datatype in datatype_processors:
                    gamera_ds = datatype_processors[datatype](ds, self.stacked_coords)
                    processed_data[datatype].append(gamera_ds)

                else:
                    print(f"Warning: No processing function for datatype '{datatype}'.")

        # Gamera conductances
        print('\n Extracting Gamera conductances')
        SHfunc, SPfunc = self.get_conductance_functions() #self.coords['lon'], self.coords['lat']

        # Reset model (delete datasets and clear model vectors)
        print('\n Clearing Emodel...')
        print('Adding Gamera conductances')
        self.osse_model.clear_model(Hall_Pedersen_conductance = (SHfunc, SPfunc))
        
        # Add synthetic datasets to osse_model
        print('Adding Gamera datasets')
        for dataset_list in processed_data.values():
            for gamera_ds in dataset_list:
                self.osse_model.add_data(gamera_ds)

        # return self.osse_model


    # def filter_datasets_by_grid(model, grid):
    def filter_datasets_by_grid(self):

        """
        Filters the datasets in the original model (user_model) to retain only data points within the model grid.

        Returns:
        --------
        lompe.Emodel object
            The original model with its datasets filtered to include only points inside the grid.
        """
        
        # print('Shape before filtering: ', self.user_model.data['convection'][0].values.shape)

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
        # print('Shape after filtering: ', self.user_model.data['convection'][0].values.shape)

        return self._input_model
    

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

        return lompe.Data(vlos, self.stacked_coords, LOS= ds.los, datatype='convection', iweight=1.0, error=100)


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

        # TODO useful at all? 
         
        Ee, En = self.get_E()
        print('Gamera electric field data extracted')

        E_values = np.vstack((Ee.flatten(), En.flatten()))

        #TODO are E_values and stacked_coords the same dim? 
        print('hey!')
        print('E_values dim:', np.shape(E_values))
        print('stacked_coords dim:', np.shape(self.stacked_coords))

        return lompe.Data(E_values, self.stacked_coords, datatype='Efield', iweight=1.0, error=1e-3)


    def Gprocess_Bfield(self, ds, stacked_coords, datatype, r, no_df_current=False):

        """
        Generates a synthetic magnetic field dataset for osse_model integration.


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

        if datatype != "ground_mag": r== ds.coords['height']+RE
        print('radius:', r)

        # Convert measurement geocentric coordinates to magnetic dipole coordinates (Gamera) 
        lat,lon = self.apex.geo2apex(self.stacked_coords[1], self.stacked_coords[0], r-RE) #lat, lon, height of the data points

        theta = 90 - lat
        phi = lon

        # Calculate magnetic field at measurement coordinates r (in meters), theta, phi
        B = get_B(r*1e3, theta, phi, self.Gstep, no_df_current=no_df_current) 

        Br, Bth, Bph = B[0], B[1], B[2] # in Tesla

        # then convert to geo

        # Compute APEX base vectors at given geographic coordinates and heights
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(self.stacked_coords[1], self.stacked_coords[0], height=r-RE, coords = 'geo')
        
        # Convert magnetic field in magnetic dipole coordinates to geographic coords
        B_geo_east, B_geo_north = Bph*f1 - Bth*f2
        B_geo_up = Br

        print(f'Gamera magnetic field data extracted ({datatype})')

        # Lompe requires east, north, up components
        Benu = np.vstack((B_geo_east, B_geo_north, B_geo_up))
        B_values = np.vstack((Benu[0].flatten(), Benu[1].flatten(), Benu[2].flatten()))

        return lompe.Data(B_values* 1e-9, self.stacked_coords, datatype=datatype, iweight=1.0, error=1e-9)
    

    # def get_Gdata(mixFile, step, hem='north', epoch=2015.):
    def get_Gdata(self, hem, mlt_off):

        """
        Reads Gamera data from an HDF5 file, extracts relevant variables based on 
        the specified hemisphere, converts magnetic dipole coordinates to geographic, 
        and processes them for use in Lompe analysis.

        Returns:
        --------
        Gdata: dict
            A dictionary containing Gamera data, ready for use in lompeOSSE
        """

        Gdata = {}

        # Open HDF5 file and load Gamera data
        with h5py.File(self.datapath, "r") as f:
            Gdata['X'] = f['X'][:]
            Gdata['Y'] = f['Y'][:]
            for step in f['Step#%d' % self.Gstep].keys():
                Gdata[step] = f['Step#%d' % self.Gstep][step][:]


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
        print(Gdata.keys()) # remove later

        # Convert Cartesian (X, Y) to spherical coordinates (R, THETA, PHI)
        X = Gdata['X'] # TODO in Earth's radius?
        Y = Gdata['Y']
        r = np.sqrt(X**2 + Y**2) # ???
        theta = np.arcsin(r) # colatitude in radians (??)
        phi = np.arctan2(Y, X) # azimuthal angle in radians (theta column in remix file)

        phi_offset = mlt_off*15 # offset in degrees
        phi = phi + phi_offset

        # # Normalize azimuthal angles to [0 - 2pi] # TODO check if useful with Kalle 
        # phi[phi < 0] = phi[phi < 0] + 2*np.pi
        # phi[:, 0] -= 2 * np.pi  # Adjust first column
        
        Gdata['R'] = r*RE # in km
        Gdata['THETA'] = theta
        Gdata['PHI'] = phi

        # Correct r, theta and phi to match other variables in GAMERA data file
        # Compute grid-centered spherical coordinates
        # r_trim     = (r[:-1, :-1] + r[:-1, 1:] + r[1:, :-1] + r[1:, 1:]) /4 # take the center of each grid cell by averaging values from adjacent points
        r_trim     = r[:-1, :-1] + np.diff(r, axis=0)[:, :-1]/2 + np.diff(r, axis=1)[:-1, :] /2 # averaged over all four corner points of each grid cell
        theta_trim = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1] /2 # averaged over theta respective grid directions
        phi_trim   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :] /2 # averaged over phi respective grid directions

        Gdata['r'] = r_trim*RE # in km
        Gdata['theta'] = theta_trim
        Gdata['phi'] = phi_trim

        # Compute magnetic latitude, longitude, and local time (USEFUL??)
        mlatG = 90 - np.rad2deg(theta_trim) # in degrees
        if hem == 'SOUTH': mlatG = (-1)*mlatG

        mlonG = np.rad2deg(phi_trim) # in degrees
        mltG = phi_trim * (12/np.pi) # in hours
        # mltOffset = 12 # offset (in hours) to center MLT at 0/24 in polar plots
        # mltG += mltOffset 

        Gdata['mlat'] = mlatG
        Gdata['mlon'] = mlonG
        Gdata['mlt'] = mltG # (USEFUL??)

        # Apply latitude mask to remove data below min_lat
        min_lat = 20  # Should be at least 11 deg
        mask = np.abs(Gdata['mlat']) > min_lat  # Boolean mask based on latitude

        for key in Gdata.keys():
            if Gdata[key].shape == Gdata['mlat'].shape:  
                Gdata[key] = np.where(mask, Gdata[key], np.nan) # Apply NaN to out-of-bounds data

        if np.abs(Gdata["mlat"]).min() < min_lat:
            print('Gdata["mlat"].min(): ', np.abs(Gdata['mlat']).min(), 'degrees')
            print(f"Low latitude GAMERA data (< {min_lat} deg) has been discarded")
        
        # Convert from magnetic to geographic coordinates using apexpy
        self.glatG, self.glonG, _ = self.apex.apex2geo(Gdata['mlat'], Gdata['mlon']+6, self.refh)

        Gdata['glon'] = self.glonG
        Gdata['glat'] = self.glatG

        return Gdata
    

    # def get_conductance_functions(grid, gamera_data):
    def get_conductance_functions(self):
           
        """
        Generates interpolation functions for Gamera Hall and Pedersen conductances.

        Returns:
        --------
        tuple of functions
            - SPfunc(glon, glat): Function that interpolates Gamera Pedersen conductance at given (lon, lat).
            - SHfunc(glon, glat): Function that interpolates Gamera Hall conductance at given (lon, lat).
        
            
        Notes:
        ------        
        - It seems that a too large grid introduces NaNs in the conductances. EXPLAIN WHY?
        """

        # Extract conductances from Gamera dataset
        SPG = self.gamera_data['Pedersen conductance']
        SHG = self.gamera_data['Hall conductance']

        # Interpolate Gamera conductances to lon, lat
        def SPfunc(lon,lat):
            ''' Gamera Pedersen conductance '''
            SP = self.interp_to_measurements(SPG, lon, lat)
            return SP

        def SHfunc(lon,lat):
            ''' Gamera Hall conductance '''
            SH = self.interp_to_measurements(SHG, lon, lat)
            return SH
        
        return SHfunc, SPfunc
    

    # def get_E(Lgrid, user_coords, Gdata, time, test=False):
    def get_E(self):

        """
        Compute the Gamera electric field at Gamera grid points, then transform it into geodetic coordinates.

        Returns:
        --------
        tuple: (Eph, -Eth) 
            Electric field components in the geographic eastward and northward directions at measurement locations lon/lat.
        """

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

        # TODO use etheta and ephi in inverse code?

        # Convert to east-north components
        self.EeG_mag = ephi # Eastward component
        self.EnG_mag = -etheta # Northward component

        # Convert Gamera electric field from magnetic dipole to geocentric coordinates
        self.EeG_geo, self.EnG_geo, _ = self.efield_mag2geo() # self.EeG_mag, self.EnG_mag, self.mlonG, self.mlatG, self.rG, self.t  
        
        # Interpolate Gamera E-field to measurement positions (glon/glat)
        self.Ee, self.En = self.efield_interp_to_measurements() # self.grid, self.coords, self.glonG, self.glatG, self.EeG, self.EnG, test=test

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
        # r = np.full(self.glon.shape, self.grid.R*1e-3) # radius in km
        r = np.full(self.coords['lon'].shape, self.grid.R*1e-3) # radius in km

        # Compute the IGRF main field components in nT
        Be_nT, Bn_nT, Bup_nT = igrf(self.coords['lon'], self.coords['lat'], self.refh, self.t) # returns radial, south, east in nT 
        # Br_nT, Bth_nT, Bph_nT = igrf_gc(r, 90 - self.glat, self.glon, self.t) 

        # Convert from nT to T
        Bn  = Bn_nT  * 1e-9
        Be  = Be_nT  * 1e-9
        Bup = Bup_nT * 1e-9

        # Convert to polar, azimuthal, radial components
        Bth = -Bn
        Bph = Be
        Br = Bup # is that right?

        return np.vstack((Br, Bph, Bth)) # radial, east, south 


    # def efield_gamera2geo(E_mag_east, E_mag_north, mlon, mlat, height, time):
    def efield_mag2geo(self):

        """
        Convert Gamera electric field components from magnetic dipole coordinates 
        to geographic geodetic coordinates (which are used in Lompe).

        Returns:
        --------
        tuple of ndarray: (E_geo_east, E_geo_north, E_geo_up)
            Electric field components in the geographic eastward, northward and upward (radial) directions.
        """

        # Compute APEX base vectors at Gamera geographic coordinates and heights
        glonG, glatG, rG = self.glonG.flatten(), self.glatG.flatten(), self.gamera_data['r'].flatten()
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(glatG, glonG, height=rG, coords = 'geo')
        
        # Compute magnetic field inclination
        mlat = self.gamera_data['mlat'].flatten()
        sinIm = 2 * np.sin(np.deg2rad(mlat)) / np.sqrt(4 - 3 * np.cos(np.deg2rad(mlat))**2) 

        # Convert to geographic geodetic coordinates using the d1 and d2 base vectors
        E_mag_east, E_mag_north = self.EeG_mag.flatten(), self.EnG_mag.flatten()
        E_geo_east, E_geo_north, E_geo_up = E_mag_east * d1 - (E_mag_north/sinIm) * d2 #TODO OK?

        # h = hem.upper()
        # if h == 'NORTH':
        # elif h == 'SOUTH': 
        #     E_geo_east, E_geo_north, E_geo_up = E_mag_east * d1 + (E_mag_north/sinIm) * d2 # is that correct??

        return (E_geo_east.reshape(self.glonG.shape), 
                E_geo_north.reshape(self.glonG.shape), 
                E_geo_up.reshape(self.glonG.shape))

    # def interp_efield_2geogrid(self, EeG, EnG, test=False):
    def efield_interp_to_measurements(self):

        """
        Interpolate the Gamera electric field (east and north components) from the 
        Gamera grid to the measurement locations (lon/lat).

        Returns:
        --------
        tuple of ndarray: (Ee_interp, En_interp, glon, glat)
            - Ee_interp: Interpolated electric field (eastward component) at measurement locations
            - En_interp: Interpolated electric field (northward component) at measurement locations
            - glon: Geographic longitude of the measurement points # TODO why return this?
            - glat: Geographic latitude of the measurement points

        """

        # Select Gsmera points inside the cubed sphere grid #TODO useful/correct? 
        iii = self.grid.ingrid(self.glonG, self.glatG)  # requires lon/lat in degrees  

        # Project Gamera E-field and coordinates (glonG, glatG) onto the cubed sphere (xiG, etaG) #TODO is that what we want? 
        self.projection = self.grid.projection 
        xiG, etaG, ExiG, EetaG = self.projection.vector_cube_projection(  
            self.EeG_geo[iii], self.EnG_geo[iii], self.glonG[iii], self.glatG[iii])

        # Project measurement coordinates (lon, lat) onto cubed sphere (xi, eta)
        xi, eta = self.projection.geo2cube(self.coords['lon'].flatten(), self.coords['lat'].flatten(), set_points_off_cube_to_nan=True)

        # Interpolate Gamera E-field to the measurement locations (xi, eta)
        Exi_interp = griddata((xiG, etaG), ExiG, (xi, eta), method='linear')
        Eeta_interp = griddata((xiG, etaG), EetaG, (xi, eta), method='linear')

        # Project back from the cubed sphere grid (xi, eta) to geographic coordinates (glon, glat) #TODO should I use index "G" for Efield???
        _, _, Ee_interp, En_interp = self.projection.vector_cube_to_geo(Exi_interp, Eeta_interp, xi, eta) # glon, glat are the same as self.coords['lon], self.coords['lat']

        # # First plot the electric field vector in its original Gamera coordinates E_phi, E_theta
        # fig,axs = plt.subplots(2,2,figsize=(10,10))
        # csax0 = cs.CSplot(axs[0][0], self.grid, gridtype='geo')
        # csax0.add_coastlines(color='grey')
        # csax0.scatter(self.glonG[0,19], self.glatG[0,19], s=35, color='red')
        # csax0.quiver(self.EeG_geo, self.EnG_geo, self.glonG, self.glatG, color='k')
        # axs[0][0].set_xlabel('Longitude')
        # axs[0][0].set_ylabel('Latitude')
        # axs[0][0].set_title(r"$E_{field}$ in GAMERA spherical coordinates ($E_\phi$, $E_\theta$)")

        # # Then plot E_xi, E_eta and compare direction and magnitude to E_phi, E_theta
        # csax1 = cs.CSplot(axs[0][1], self.grid, gridtype='cs')
        # csax1.add_coastlines(color='grey')
        # axs[0][1].scatter(xiG[19], etaG[19], s=35, color='red')
        # axs[0][1].quiver(xiG, etaG, Exi, Eeta, color='k') # use matplotlib quiver function when it comes to xi and eta coordinates
        # axs[0][1].set_title(r"$E_{field}$ in GAMERA cube coordinates ($E_\xi$, $E_\eta$)")

        # # Now plot the electric field vector interpolated to input lon, lat values
        # csax2 = cs.CSplot(axs[1][0], self.grid, gridtype='cs')
        # csax2.add_coastlines(color='grey')
        # axs[1][0].scatter(xi[5], eta[5], s=40, color='green')
        # axs[1][0].quiver(xi, eta, Exi_interp, Eeta_interp, color='k') # scale???
        # axs[1][0].set_title(r"Interpolated $E_{field}$ (cube coord. $E_\xi$, $E_\eta$)")

        # # Finally, plot the interpolated electric field back in a spherical system
        # csax3 = cs.CSplot(axs[1][1], self.grid, gridtype='geo')
        # csax3.add_coastlines(color='grey')
        # csax3.scatter(glon[5], glat[5], s=40, color='green')
        # csax3.quiver(Ee_interp, En_interp, glon, glat) #, scale=900
        # axs[1][1].set_xlabel('Longitude')
        # axs[1][1].set_ylabel('Latitude')
        # axs[1][1].set_title(r"Interpolated $E_{field}$ (spherical coord. $E_\phi$, $E_\theta$)")
        # plt.tight_layout()
        # plt.show()

        return (Ee_interp.reshape(self.coords['lon'].shape), 
                En_interp.reshape(self.coords['lon'].shape))


    def interp2lompegrid(self, var): #TODO delete

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
    
    def interp_to_measurements(self, var, lon, lat): # TODO should it be self.coords['lon'], self.coords['lat'] here?

        """
        Interpolate a variable (scalar) from the Gamera grid to the measurement locations (lon/lat).

        Parameters:
        -----------
        var : ndarray (scalar)
            Gamera scalar variable (e.g., conductance) defined at glonG, glatG. 

        Returns:
        --------
        varinterp : ndarray
            Interpolated variable at the measurement locations, reshaped to match coordinate dimensions.
        
        """
        
        # Identify Gamera grid points that fall inside the cubed sphere grid #TODO useful??
        iii = self.grid.ingrid(self.glonG, self.glatG, ext_factor = 1.5)

        # Convert Gamera geocentric coordinates (glonG, glatG) to cubed sphere coordinates (xiG, etaG)
        xiG, etaG = self.projection.geo2cube(self.glonG[iii],self.glatG[iii]) # Gamera data points coordinates

        # Convert measurement coordinates (lon, lat) to cubed sphere coordinates (xi, eta) #TODO lon,lat are glon,glat in reality
        # xi, eta = self.projection.geo2cube(self.coords['lon'].flatten(), self.coords['lat'].flatten(), set_points_off_cube_to_nan=True)
        xi, eta = self.projection.geo2cube(lon.flatten(), lat.flatten(), set_points_off_cube_to_nan=True) # points at which to interpolate the Gamera data

        # Interpolate Gamera variable to the measurement locations (cubed sphere coords)
        varinterp = griddata((xiG, etaG), var[iii], (xi, eta), method='linear')

        return varinterp.reshape(lon.shape)
    
    
# ### **Create a Factory Function**
# # function that instantiates the object and returns osse_model while still keeping the full lompeOSSE object accessible
# def create_lompeOSSE(*args, **kwargs):
#     obj = osseEmodel(*args, **kwargs)  # Create the object
#     return obj.osse_model, obj  # Return both osse_model and the full object