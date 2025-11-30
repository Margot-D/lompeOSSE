import numpy as np
import copy
import lompe
from magnetic_field_utils import get_B

RE = 6371.2 # Earth radius in km
RI = 6500 # Ionospheric radius in km (used in Gamera simulations)

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
    
