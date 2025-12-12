import numpy as np
import copy
import lompe
# from magnetic_field_utils import get_B
from gamera_output import Gamera_output
import datetime as dt

RE = 6371.2 # Earth radius in km
RI = 6500 # Ionospheric radius in km (used in Gamera simulations)

class LompeOSSE(object):

    """ 
    OSSE Model class 

    The OSSE model is a copy of user-defined Lompe model, but its datasets are replaced with  
    synthetic data from Gamera simulations, including Gamera-derived conductances. 
    All other model properties remain unchanged.

    Regarding the observational datasets, suported datasets by Lompe/LompeOSSE include:
    - Magnetic field perturbations on ground
    - Magnetic field perturbations in space associated with field-aligned currents
    - Magnetic field perturbations in space associated with both field-aligned currents 
    and horizontal divergence-free currents below the satellite
    - Ionospheric convection velocity (perpendicular to the magnetic field and 
    mapped to the ionospheric radius)
    - Ionospheric convection electric field 
    TODO OK?

    """

    def __init__(self, input_model, synthetic_object):

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
            This allows sampling multiple configurations (MLT sectorS) from a single Gamera snapshot.

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

        self.Gamera_object = synthetic_object
        self.apex = self.Gamera_object.apex
        self.gamera_data = self.Gamera_object.gamera_data
        self.Gstep = self.Gamera_object.timestep
        
        self.timestamp = self.Gamera_object.time #TODO fix that, i think we wamt time to be input in lompeosse instead

        self._input_model = input_model
        
        # Get synthetic model
        # self.synthetic_model = self.make_OSSE_model(time_offset)


    def make_OSSE_model(self, time_offset=0):

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

        TODO write something about  # Optional: add MLT offset (rotates the Gamera snapshot in magnetic local time)


        Returns:
        --------
        lompe.Emodel object
            A copy of the original model but with synthetic Gamera data replacing real observations.
        """

        self.time_offset = time_offset

        print('time offset:', self.time_offset, 'hours')
        ntime = self.timestamp + dt.timedelta(hours=self.time_offset)
        print(ntime)

        # Ensure input datasets are inside the user grid

        # print('Shape before filtering: ', self._input_model.data['convection'][0].values.shape)
        self.filter_datasets_by_grid()
        # print('Shape after filtering: ', self._input_model.data['convection'][0].values.shape)


        # Make a copy of input model
        print('\n Initializing synthetic model...')
        synthetic_model = copy.copy(self._input_model)
        # grid = synthetic_model.grid_J

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
                    print(f'{datatype} dataset found..')
                    gamera_ds = datatype_processors[datatype](ds, ntime)
                    processed_data[datatype].append(gamera_ds)
                else:
                    print(f"Warning: No processing function for datatype '{datatype}'.")

        # Gamera conductances
        # SHfunc, SPfunc = self.Gamera_object.get_conductance_functions(grid)
        SHfunc, SPfunc = self.extract_synth_conductances(ntime)
        # print('\n Gamera conductances extracted')

        # Reset model (delete datasets and clear model vectors)
        print('\n Clearing input model...')
        # print('Adding Gamera conductances')
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

        return self._input_model
    

    def extract_synth_conductances(self, time):
           
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
        # chatgpt suggestion 
        # SH_data = self.Gamera_object.get_Hall(grid.lon, grid.lat, time)
        # SP_data = self.Gamera_object.get_Pedersen(grid.lon, grid.lat, time)
        # def SHfunc(glon, glat):
        #     return Gamera_obj.interp_to_measurements(grid, glon, glat, var=SH_data)

        # Interpolate Gamera conductances to lon, lat
        def SPfunc(lon,lat):
            ''' Gamera Pedersen conductance '''
            SP = self.Gamera_object.get_Pedersen(lon, lat, time)
            return SP

        def SHfunc(lon,lat):
            ''' Gamera Hall conductance '''
            SH = self.Gamera_object.get_Hall(lon, lat, time)
            return SH
        
        return SHfunc, SPfunc
    

    def extract_synth_convection(self, ds, time):

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

        V_geo_east, V_geo_north = self.Gamera_object.get_V(ds.coords['lon'], ds.coords['lat'], time) #TODO add something about radius?
        print('..Gamera convection data extracted')

        # print(V_geo_east)
        # print(V_geo_north)

        # Project Gamera velocity components onto the dataset's LOS direction
        vlos = V_geo_east * ds.los[0] + V_geo_north * ds.los[1]

        return lompe.Data(vlos, np.vstack((ds.coords['lon'], ds.coords['lat'])), LOS= ds.los, datatype='convection', iweight=ds.iweight, error=ds.error)


    def extract_synth_efield(self, ds, time):  

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

        E_geo_east, E_geo_north = self.Gamera_object.get_E(ds.coords['lon'], ds.coords['lat'], time)
        print('..Gamera electric field data extracted')

        E_values = np.vstack((E_geo_east.flatten(), E_geo_north.flatten()))

        return lompe.Data(E_values, np.vstack((ds.coords['lon'], ds.coords['lat'])), datatype='Efield', iweight=ds.iweight, error=ds.error)


    def extract_synth_bfield(self, ds, time):

        #TODO time not taken into account
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

        # Radius used for magnetic field calculations (in meters)
        if ds.datatype == 'ground_mag': r = np.full_like(ds.coords['lon'], RE*1e3) # assumes perfectly circular Earth TODO fix?
        else: r = ds.coords['r']

        if ds.datatype == "space_mag_fac": no_df_current=True 
        else: no_df_current=False

        B_geo_east, B_geo_north, B_geo_up = self.Gamera_object.get_B(ds.coords['lon'], ds.coords['lat'], r, no_df_current)
        print(f'..Gamera {ds.datatype} data extracted')

        # Lompe requires east, north, up components
        B_values = np.vstack((B_geo_east, B_geo_north, B_geo_up))

        # TODO what should r be here?? is it r or height? 
        return lompe.Data(B_values* 1e-9, np.vstack((ds.coords['lon'], ds.coords['lat'], r)), datatype=ds.datatype, iweight=ds.iweight, error=ds.error)
    
