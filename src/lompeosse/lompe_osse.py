import numpy as np
import datetime as dt
import copy
import lompe
from .gamera_data import GameraData

RE = 6371.2 # Earth radius in km

class LompeOSSE(object):
    """
    Lompe model for Observation System Simulation Experiments (OSSEs).

    Creates a copy of a user-defined Lompe model and replaces its observational datasets 
    with synthetic data derived from Gamera simulations. The synthetic data are evaluated 
    at the same locations as the original Lompe datasets. 

    Parameters
    ----------
    input_model : lompe.Emodel
        User-defined Lompe model whose configuration and dataset locations are used as the 
        basis for the synthetic model.

    timestep : int
        Index of the Gamera simulation snapshot to use.
        Available timesteps: #0 #2 #3 #12 #13 #14 #16 #19 #20 #21 #22 (use ``find-simulation-snapshot.py`` 
        to inspect the available snapshots)

    event_time : datetime 
        Reference time associated with the user-defined event.
        
    time_offset : float, optional 
        Time offset in hours relative to ``event_time``. Changes the orientation of the Gamera simulation 
        pattern relative to the Lompe analysis region, allowing different parts of a given snapshot to be sampled. 
        Default is 0.

    Notes
    -----
    The synthetic model is reconstructed using the same model configuration and data geometry
    as the original model. The original Lompe model is not modified when constructing the synthetic 
    model. 
   
    Supported dataset types include:
        - Ground magnetic field perturbations
        - Space magnetic field perturbations associated with field-aligned currents
        - Space magnetic field perturbations associated with both field-aligned currents and divergence-free horizontal currents
        - Field-aligned current (FAC) density
        - Ionospheric convection velocity
        - Ionospheric convection electric field
    """

    def __init__(self, input_model, event_time, timestep=0, time_offset=0):   

        self._input_model = input_model

        if np.min(np.abs(input_model.grid_J.lat)) < 48:
            raise ValueError("LompeOSSE requires a grid extending no further equatorward than |48|°.")

        # Determine the time at which Gamera data are sampled
        self.analysis_time = event_time + dt.timedelta(hours=time_offset)

        self.timestep = timestep

        # Load the Gamera simulation corresponding to the hemisphere of the Lompe model grid
        hem = 'NORTH' if input_model.grid_J.projection.lat0 > 0 else 'SOUTH' 
        self.Gamera_object = GameraData(self.analysis_time, self.timestep, hemisphere = hem)
        
        # Construct the synthetic model TODO if the simulation file itself becomes too big and is expensive to load, maybe make_OSSE_model() could be called once the lompeosse object initialised. Not an issue for now.
        self.synthetic_model = self.make_OSSE_model()

    def make_OSSE_model(self):
        """
        Create a synthetic Lompe model using Gamera data.

        The original model provides the model configuration, grid, dataset locations, weights, 
        and errors. The values of the observational datasets are replaced by corresponding 
        quantities extracted from Gamera.

        The conductance functions are also replaced by functions that
        evaluate the Hall and Pedersen conductances from Gamera.

        Returns
        -------
        lompe.Emodel
            A copy of the input Lompe model containing synthetic Gamera datasets and 
            Gamera-derived conductances.
        """

        print(f'Initializing synthetic model... \n Gamera timestep #{self.timestep} \n Central time: {self.analysis_time}')

        # Make a shallow copy of the input model
        synthetic_model = copy.copy(self._input_model)

        # ``matrix_func`` contains bound methods used to construct the model matrices. 
        # Because this is a shallow copy, the methods would otherwise remain bound to 
        # the original model. Rebind them to the synthetic model so that the inversion 
        # uses its Gamera-derived conductances.
        synthetic_model.matrix_func = {'ground_mag':     synthetic_model._B_df_matrix,
                                       'convection':     synthetic_model._v_matrix,
                                       'efield':         synthetic_model._E_matrix,
                                       'space_mag_fac':  synthetic_model._B_cf_matrix,
                                       'space_mag_full': synthetic_model._B_cf_df_matrix,
                                       'fac':            synthetic_model.FAC_matrix}

        # print('Scanning user datasets and searching for corresponding Gamera data...')

        # Map each Lompe datatype with the function used to generate its synthetic Gamera counterpart
        datatype_processors = {'convection': self.extract_synthetic_convection,
                               'efield': self.extract_synthetic_efield,
                               'ground_mag': self.extract_synthetic_bfield,
                               'space_mag_fac': self.extract_synthetic_bfield, 
                               'space_mag_full': self.extract_synthetic_bfield,
                               'fac': self.extract_synthetic_fac}

        # Generate synthetic data for each dataset
        processed_data = {}
        for datatype, dataset_list in synthetic_model.data.items():

            if not dataset_list:
                continue

            if datatype not in datatype_processors:
                print(f"Warning: No processing function for datatype '{datatype}'")
                continue

            processed_data[datatype] = []
            for ds in dataset_list:

                    if ds.values.size == 0:
                        print(f"Skipping empty {datatype} dataset")
                        continue

                    # Extract the corresponding Gamera quantity at the measurement locations of the original dataset
                    synthetic_ds = datatype_processors[datatype](ds)
                    processed_data[datatype].append(synthetic_ds)

        # Use Gamera Hall and Pedersen conductances in the synthetic model
        SHfunc, SPfunc = self.extract_synthetic_conductances()

        # Clear the original datasets and model vectors, and replace the conductances 
        # with the Gamera-derived Hall and Pedersen conductances 
        synthetic_model.clear_model(Hall_Pedersen_conductance = (SHfunc, SPfunc))

        # Add the Gamera datasets to the synthetic model
        for dataset_list in processed_data.values():
            for synthetic_ds in dataset_list:
                synthetic_model.add_data(synthetic_ds)

        print(f'Synthetic model generated.')

        return synthetic_model

    # def extract_synthetic_conductances(self, time): # original version, without cache
    #     """
    #     Create functions for evaluating Gamera Hall and Pedersen conductances.

    #     Parameters
    #     ----------
    #     time : datetime
    #         Time at which the Gamera conductances are evaluated.

    #     Returns
    #     -------
    #     tuple of callable
    #         SHfunc and SPfunc, which return the Gamera Hall and Pedersen conductances 
    #         at given geographic coordinates.
    #     """

    #     def SHfunc(lon,lat):
    #         """Return Gamera Hall conductance at the given coordinates"""
    #         SH = self.Gamera_object.get_Hall(lon, lat, time)
    #         print('Gamera Hall conductance extracted')
    #         return SH
        
    #     def SPfunc(lon,lat):
    #         """Return Gamera Pedersen conductance at the given coordinates"""
    #         SP = self.Gamera_object.get_Pedersen(lon, lat, time)
    #         print('Gamera Pedersen conductance extracted')
    #         return SP
 
    #     return SHfunc, SPfunc
    
    def extract_synthetic_conductances(self):
        """
        Create functions for evaluating Gamera Hall and Pedersen conductances.

        The returned functions accept arbitrary geographic coordinates and return the corresponding 
        Gamera conductances at the specified time. Conductances are cached for each unique set of 
        coordinates to avoid repeating the expensive Gamera data interpolation when the same coordinates 
        are requested multiple times.

        Returns
        -------
        tuple of callable
            SHfunc and SPfunc, which return the Gamera Hall and Pedersen conductances at given 
            geographic coordinates.
        """

        conductance_cache = {}

        def get_conductances(lon, lat):

            key = (lon.shape, lat.shape, lon.dtype.str, lat.dtype.str, np.ascontiguousarray(lon).tobytes(), np.ascontiguousarray(lat).tobytes())

            if key not in conductance_cache:
                # Calculate and cache new conductances
                SH = self.Gamera_object.get_Hall(lon, lat)
                SP = self.Gamera_object.get_Pedersen(lon, lat)
                conductance_cache[key] = (SH, SP) # Storing in cache
            
            else:
                # Use cached conductances
                SH, SP = conductance_cache[key]

            return SH, SP

        def SHfunc(lon, lat):
            """Return Gamera Hall conductance at the given coordinates"""
            SH, _ = get_conductances(lon, lat)
            return SH

        def SPfunc(lon, lat):
            """Return Gamera Pedersen conductance at the given coordinates"""
            _, SP = get_conductances(lon, lat)
            return SP

        return SHfunc, SPfunc


    def extract_synthetic_convection(self, ds):
        """
        Generate a synthetic convection dataset from Gamera.

        The Gamera velocity components are evaluated at the measurement locations of the original Lompe dataset 
        and projected onto its line-of-sight directions.

        Parameters
        ----------
        ds : lompe.Data
            Original Lompe convection dataset. Its coordinates and line-of-sight vectors 
            define where and how the synthetic measurements are sampled.

        Returns
        -------
        lompe.Data
            Synthetic convection dataset containing Gamera-derived
            line-of-sight velocities.
        """

        V_geo_east, V_geo_north = self.Gamera_object.get_V(ds.coords['lon'], ds.coords['lat'])

        # Velocity in line-of-sight direction
        vlos = V_geo_east * ds.los[0] + V_geo_north * ds.los[1]

        return lompe.Data(vlos, np.vstack((ds.coords['lon'], ds.coords['lat'])), LOS=ds.los, datatype='convection', iweight=ds.iweight, error=ds.error)


    def extract_synthetic_efield(self, ds):  
        """
        Generate a synthetic electric field dataset from Gamera.

        Parameters
        ----------
        ds : lompe.Data
            Original Lompe electric field dataset. Its coordinates determine where the 
            synthetic electric field is sampled.

        Returns
        -------
        lompe.Data
            Synthetic electric field dataset containing Gamera-derived eastward and 
            northward electric field components.
        """

        E_geo_east, E_geo_north = self.Gamera_object.get_E(ds.coords['lon'], ds.coords['lat'])

        E_values = np.vstack((E_geo_east.flatten(), E_geo_north.flatten()))

        return lompe.Data(E_values, np.vstack((ds.coords['lon'], ds.coords['lat'])), datatype='Efield', iweight=ds.iweight, error=ds.error)


    def extract_synthetic_bfield(self, ds):
        """
        Generate a synthetic magnetic field dataset from Gamera.

        The magnetic field is evaluated at the coordinates of the original dataset. 

        Ground magnetic data are evaluated at the Earth's surface, while space magnetic data 
        retain their original radial coordinates.

        For ``space_mag_fac`` data, only the magnetic field contribution from field-aligned currents 
        is retained. For ``space_mag_full`` and ``ground_mag`` data, the divergence-free horizontal 
        current contribution is also included.

        Parameters
        ----------
        ds : lompe.Data
            Original Lompe magnetic field dataset. Its geographic coordinates and, for 
            space magnetic data, radial coordinates define where the synthetic magnetic field is sampled.

        Returns
        -------
        lompe.Data
            Synthetic magnetic field dataset containing Gamera-derived eastward, northward, and upward components.
        """

        # Radius used for magnetic field calculations in [m]
        if ds.datatype == 'ground_mag':
            r = np.full_like(ds.coords['lon'], RE * 1e3) # Earth's surface
        else:
            r = ds.coords['r']

        # Don't include the divergence-free current for ``space_mag_fac`` 
        no_df_current = ds.datatype == 'space_mag_fac'

        B_geo_east, B_geo_north, B_geo_up = self.Gamera_object.get_B(ds.coords['lon'], ds.coords['lat'], r, no_df_current=no_df_current)

        B_values = np.vstack((B_geo_east, B_geo_north, B_geo_up))

        return lompe.Data(B_values, np.vstack((ds.coords['lon'], ds.coords['lat'], r)), datatype=ds.datatype, iweight=ds.iweight, error=ds.error)


    def extract_synthetic_fac(self, ds):  
        """
        Generate a synthetic field-aligned current (FAC) dataset from Gamera.

        Parameters
        ----------
        ds : lompe.Data
            Original Lompe FAC dataset. Its coordinates determine where the synthetic FAC is sampled.

        Returns
        -------
        lompe.Data
            Synthetic FAC dataset containing Gamera-derived field-aligned current values.
        """

        fac = self.Gamera_object.get_FAC(ds.coords['lon'], ds.coords['lat'])

        return lompe.Data(fac.flatten(), np.vstack((ds.coords['lon'], ds.coords['lat'])), datatype='fac', iweight=ds.iweight, error=ds.error)

