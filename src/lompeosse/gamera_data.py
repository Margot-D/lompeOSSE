import h5py
import os
import numpy as np
import datetime as dt
import apexpy
from scipy.interpolate import griddata
from ppigrf import igrf
import dipole # github.com/klaundal/dipole
from secsy import cubedsphere as cs
# from pathlib import Path
from importlib.resources import files, as_file

from .magnetic_field_utils.sh_basis import SHBasis
from .magnetic_field_utils.grid import Grid
from .magnetic_field_utils.basis_evaluator import BasisEvaluator

mu0 = np.pi * 4e-7
N, M = 110, 110 # coefficients for the spherical harmonic analysis (in get_B)
RE = 6371.2 # Earth radius [km]
RI_GAMERA = 6500 # Ionospheric radius used in Gamera simulations [km]

class GameraData(object):
    """
    Provide Gamera simulation data for use in LompeOSSE.

    The Gamera simulation data are defined on a centered-dipole coordinate system, while 
    the Lompe model uses geographic coordinates. This class handles the coordinate transformations 
    between these systems and provides the methods needed by LompeOSSE to obtain Gamera quantities 
    - including conductances, electric and magnetic fields, field-aligned currents, and ionospheric 
    convection velocity - at the observation locations. 

    Parameters
    ----------
    gamera_time : datetime
        Time associated with the Gamera simulation.
    timestep : int, optional
        Index of the Gamera simulation snapshot to use. Default is 0.
    hemisphere : {'NORTH', 'SOUTH'}, optional
        Hemisphere of the Gamera data to use. Default is 'NORTH'.

    Notes
    -----
    The returned quantities have the same shape as the input observation coordinates.
    """

    def __init__(self, gamera_time, timestep=0, hemisphere='NORTH'):

        self.analysis_time = gamera_time
        self.timestep = timestep    
        self.hemisphere = hemisphere

        self.refh = RI_GAMERA - RE # reference height of the Gamera output [km]
        self.apex = apexpy.Apex(gamera_time, self.refh)
        self.dp = dipole.Dipole(gamera_time.year)

        # # Support both a source checkout and data bundled with an installed package
        # package_root = Path(__file__).resolve().parent
        # candidates = [package_root / "data" / "Gamera_data.h5"]
        # if package_root.parent.name == "src":
        #     candidates.insert(0, package_root.parent.parent / "data" / "Gamera_data.h5")
        # datapath = next((path for path in candidates if path.is_file()), None)

        # if datapath is None:
        #     raise FileNotFoundError(
        #         "Gamera_data.h5 was not found. Searched:\n"
        #         + "\n".join(str(path) for path in candidates)
        #         + "\nDownload it from https://zenodo.org/records/16882035 and place it at one of these paths.")

        datapath = files("lompeosse").joinpath("data", "Gamera_data.h5")

        if not datapath.is_file():
            raise FileNotFoundError("Gamera_data.h5 was not found. "
                                    "Download it from https://zenodo.org/records/16882035 "
                                    "and place it in the lompeosse/data directory.")

        # Load Gamera data
        with as_file(datapath) as filepath:
            self.gamera_data = self._load_Gamera_data(filepath)

    def _load_Gamera_data(self, fn):
        """
        Load Gamera data from an HDF5 file and prepare its coordinates.

        Parameters
        ----------
        fn : str or path-like
            Path to the Gamera HDF5 file.

        Returns
        -------
        gamera_data : dict
            Dictionary containing the Gamera variables for the requested timestep and hemisphere.
        """

        gamera_data = {}

        # ------------------------#
        # Open HDF5 file
        # ------------------------#

        with h5py.File(fn, "r") as f:

            # Get the snapshot indices available in the Gamera data file
            available_steps = sorted(int(k.split('#')[1]) for k in f.keys() if k.startswith("Step#"))

            if self.timestep not in available_steps:
                raise ValueError(f"Gamera snapshot #{self.timestep} is not available. Available snapshots: {available_steps}")

            # Load the Gamera cartesian grid coordinates
            gamera_data['X'] = f['X'][:]
            gamera_data['Y'] = f['Y'][:]

            # Load all variables from the requested snapshot
            for step in f['Step#%d' % self.timestep].keys():
                gamera_data[step] = f['Step#%d' % self.timestep][step][:]

        # ------------------------#
        # Filter Gamera dataset by hemisphere
        # ------------------------#

        hemi_gamera_data = {}

        for key, value in gamera_data.items():

            if self.hemisphere == "NORTH":
                if "SOUTH" in key:
                    continue
                if "NORTH" in key:
                    key = key.replace("NORTH", "").strip()

            elif self.hemisphere == "SOUTH":
                if "NORTH" in key:
                    continue
                if "SOUTH" in key:
                    key = key.replace("SOUTH", "").strip()

            hemi_gamera_data[key] = value

        # Replace the full dataset with the hemisphere-specific data
        gamera_data.clear()
        gamera_data.update(hemi_gamera_data)

        # ------------------------#
        # Derive coordinates
        # ------------------------#

        # Normalized Cartesian coordinates from the Gamera grid (dipole-based coordinate system)
        x = gamera_data['X'] 
        y = gamera_data['Y']

        # Convert Cartesian coordinates to spherical coordinates (following the Remix convention)
        self.theta = np.arcsin(np.sqrt(x**2 + y**2)) # polar angle [rad]
        self.phi = np.arctan2(y, x) # azimuthal angle [rad]

        # Shift negative azimuths to [0, 2π) and shift the first column by -2π to keep the grid continuous across the periodic boundary
        self.phi[self.phi < 0] += 2 * np.pi
        self.phi[:, 0] -= 2 * np.pi
      
        # Calculate cell-centered coordinates in the Gamera dipole coordinate system to match the shape of the Gamera variables
        self.theta_center = 0.25 * (self.theta[:-1, :-1] + self.theta[1:, :-1] + self.theta[:-1, 1:] + self.theta[1:, 1:])
        self.phi_center = 0.25 * (self.phi[:-1, :-1] + self.phi[1:, :-1] + self.phi[:-1, 1:] + self.phi[1:, 1:])

        # Convert azimuthal angle [rad] to magnetic local time [hours]
        self.mlt0 = 12 # Assuming phi=0 corresponds to magnetic noon in the Gamera coordinate system https://doi.org/10.1029/2021JA029738
        self.mlt = (self.phi_center * (12 / np.pi) + self.mlt0) % 24

        # Convert polar angle to signed magnetic latitude [degrees]
        self.mlat = 90 - np.rad2deg(self.theta_center)
        if self.hemisphere == 'SOUTH': self.mlat *= -1

        # Calculate geographic coordinates from the dipole coordinates at the analysis time
        self.gamera_glat, self.gamera_glon = self.gamera_dipole_to_geo(self.analysis_time)

        return gamera_data

    def _get_scalar_parameter(self, glon, glat, gamera_variable):
        """
        Interpolate a scalar Gamera quantity to geographic coordinates.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].
        gamera_variable : str
            Name of the Gamera quantity to interpolate.

        Returns
        -------
        array
            Interpolated Gamera quantity at the measurement locations.
        """

        projection = self.centered_csprojection(glon, glat)

        return self.interp_to_measurements(projection, glon, glat, self.gamera_glon, self.gamera_glat, var=self.gamera_data[gamera_variable])

    def get_potential(self, glon, glat):
        """Return the Gamera electric potential at the measurement locations"""
        return(self._get_scalar_parameter(glon, glat, 'Potential') *1e3) # Convert from [kV] to [V]

    def get_FAC(self, glon, glat):
        """Return the Gamera field-aligned current density at the measurement locations"""
        fac_parallel = self._get_scalar_parameter(glon, glat, 'Field-aligned current') *1e-6 # Convert from [µA/m²] to [A/m²]

        # The Gamera/REMIX quantity is positive along the background magnetic field. Lompe defines FAC as positive upward. 
        # The dipole field points into Earth in the north and out of Earth in the south.
        upward_sign = -1 if self.hemisphere.upper() == 'NORTH' else 1

        return upward_sign * fac_parallel # [A/m²], positive upward

    def get_Hall(self, glon, glat):
        """Return the Gamera Hall conductance at the measurement locations"""
        return(self._get_scalar_parameter(glon, glat, 'Hall conductance')) # [S]

    def get_Pedersen(self, glon, glat):
        """Return the Gamera Pedersen conductance at the measurement locations"""
        return(self._get_scalar_parameter(glon, glat, 'Pedersen conductance')) # [S]

    def get_hCurrents(self, glon, glat):
        """Calculate the Gamera horizontal ionospheric surface current density at measurement locations.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].

        Returns
        -------
        je : array
            Eastward horizontal surface current density [A/m].
        jn : array
            Northward horizontal surface current density [A/m].
        """

        # Electric field
        Ee, En = self.get_E(glon, glat)

        # Conductances
        SP = self.get_Pedersen(glon, glat)
        SH = self.get_Hall(glon, glat)

        hemisphere_sign = 1 if self.hemisphere == 'NORTH' else -1

        # Horizontal current density
        je = Ee * SP + SH * En * hemisphere_sign
        jn = En * SP - SH * Ee * hemisphere_sign

        return je, jn

    def get_E(self, glon, glat, RI=RI_GAMERA):
        """
        Calculate the Gamera electric field at measurement locations.

        The electric field is first calculated from the Gamera electric potential on the Gamera grid. 
        The field is then transformed from the Gamera magnetic coordinate system to geographic eastward 
        and northward components and interpolated to the requested geographic measurement locations.

        This implementation is largely based on the electric-field calculation in Kaipy/REMIX, with the additional transformation
        from the Gamera magnetic coordinate system to geographic coordinates.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].
        RI : float, optional
            Reference radius of the Gamera ionosphere [km].

        Returns
        -------
        E_east : array
            Eastward electric field at the measurement locations [V/m].
        E_north : array
            Northward electric field at the measurement locations [V/m].
        """
           
        theta = self.theta
        phi = self.phi
        Psi = self.gamera_data['Potential'] # [kV]

        # ------------------------#
        # Calculate electric field on the Gamera grid
        # ------------------------#  
                
        # Calculate potential at the grid corners by averaging surrounding cell values
        Psi_c = np.zeros(theta.shape)
        Psi_c[1:-1,1:-1] = 0.25 * (Psi[1:,1:] + Psi[:-1,1:] + Psi[1:,:-1] + Psi[:-1,:-1])

        # Handle periodic boundary conditions in longitude
        Psi_c[1:-1,0]  = 0.25 * (Psi[1:,0] + Psi[:-1,0] + Psi[1:,-1] + Psi[:-1,-1])
        Psi_c[1:-1,-1] = Psi_c[1:-1,0]

        # Handle the pole boundary condition using the mean potential at the pole
        Psi_pole = Psi[0,:].mean()
        Psi_c[0,1:-1] = 0.25 * (2.*Psi_pole + Psi[0,:-1] + Psi[0,1:]) 
        Psi_c[0,0]    = 0.25 * (2.*Psi_pole + Psi[0,-1] + Psi[0,0]) 
        Psi_c[0,-1]   = 0.25 * (2.*Psi_pole + Psi[0,-1] + Psi[0,0])     

        # Extrapolate the potential at the lower-latitude boundary
        Psi_c[-1,:] = 2 * Psi_c[-2,:] - Psi_c[-3,:]

        # Meridional (north-south) electric field component
        tmp    = 0.5 * (Psi_c[:,1:] + Psi_c[:,:-1])  
        dPsi   = tmp[1:,:] - tmp[:-1,:]
        tmp    = 0.5 * (theta[:,1:] + theta[:,:-1])
        dtheta = tmp[1:,:] - tmp[:-1,:]
        etheta = -dPsi/dtheta/RI  # E = -∇Ψ [kV/km]
        # etheta = etheta * 1e3  # [V/m]

        # Zonal (east-west) electric field component
        tmp    = 0.5 * (Psi_c[1:,:] + Psi_c[:-1,:]) 
        dPsi   = tmp[:,1:] - tmp[:,:-1]
        tmp    = 0.5 * (phi[1:,:] + phi[:-1,:])
        dphi   = tmp[:,1:] - tmp[:,:-1]
        tc = 0.25 * (theta[:-1,:-1] + theta[1:,:-1] + theta[:-1,1:] + theta[1:,1:]) # polar angle at cell centers
        ephi = -dPsi/dphi/np.sin(tc)/RI  # E = -∇Ψ [kV/km]
        # ephi = ephi * 1e3 # [V/m]

        # ------------------------#
        # Transform to geographic coordinates
        # ------------------------#

        gamera_glon, gamera_glat = self.gamera_glon.flatten(), self.gamera_glat.flatten()
        mlat = self.mlat.flatten()

        # Compute APEX base vectors at the Gamera grid locations
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(gamera_glat, gamera_glon, height=RI-RE, coords='geo')

        # Magnetic field inclination
        sinIm = 2 * np.sin(np.deg2rad(mlat)) / np.sqrt(4 - 3 * np.cos(np.deg2rad(mlat))**2) 

        # Convert magnetic components to geographic eastward and northward components uing the APEX base vectors
        E_mag_east, E_mag_north = ephi, -etheta
        E_geo_east, E_geo_north, E_geo_up = E_mag_east.flatten() * d1 - (E_mag_north.flatten()/sinIm) * d2
        
        # ------------------------#
        # Interpolate to measurement locations (glon, glat)
        # ------------------------#
        
        projection = self.centered_csprojection(glon, glat)
        E_east, E_north = self.interp_to_measurements(projection, glon, glat, gamera_glon, gamera_glat, vec=(E_geo_east, E_geo_north))

        return E_east, E_north

    def get_V(self, glon, glat, RI=RI_GAMERA):
        """
        Calculate the ExB plasma drift velocity at measurement locations.

        The electric field is obtained from the Gamera potential and the geomagnetic field from IGRF. 
        The radial electric-field component is calculated by assuming that the electric field is perpendicular to
        the magnetic field, E · B = 0.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].
        RI : float, optional
            Reference radius of the Gamera ionosphere [km].

        Returns
        -------
        V_east : array
            Eastward ExB drift velocity [m/s].
        V_north : array
            Northward ExB drift velocity [m/s].
        """

        # Electric field in geographic eastward and northward components
        E_east, E_north = self.get_E(glon, glat, RI)

        # Convert to spherical eastward (phi) and polar (theta) components
        Eph = E_east # azimuthal
        Eth = -E_north # polar

        # Geomagnetic field components and magnitude
        B_east, B_north, B_up = self.get_Bigrf(glon, glat)

        Br = B_up
        Bph = B_east
        Bth = -B_north

        B2 = np.sqrt(Br**2 + Bth**2 + Bph**2)

        # Calculate the radial electric field from E · B = 0
        Er = -(Eth * Bth + Eph * Bph) / Br

        # Calculate ExB drift velocity in spherical coordinates
        Vr  = (Eth * Bph - Eph * Bth) / B2**2
        Vph = (Er * Bth - Eth * Br)   / B2**2
        Vth = (Eph * Br - Er * Bph)   / B2**2

        # Convert back to geographic eastward and northward components
        V_east = Vph
        V_north = -Vth

        return V_east, V_north

    def get_B(self, glon, glat, r, no_df_current, RI=RI_GAMERA*1e3):
        """
        Calculate the Gamera magnetic field at measurement locations.

        The magnetic field is calculated from spherical harmonic coefficients for the selected Gamera timestep. 
        The calculation first converts the geographic measurement coordinates to magnetic Apex coordinates, 
        then evaluates the Gamera spherical-harmonic representation of the magnetic field at those locations, 
        and finally converts the resulting field components back to geographic eastward, northward, and upward components.

        The calculation uses different spherical-harmonic expressions depending on the measurement radius. For locations 
        inside RI, only the poloidal contribution is evaluated. For locations at or outside RI, both poloidal and toroidal 
        contributions are evaluated.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].
        r : array
            Geocentric radial distance [m].
        no_df_current : bool
            Whether to exclude the divergence-free current contribution.
            If False, the magnetic field includes both the field-aligned current and divergence-free 
            current contributions. If True, only the field-aligned current contribution is included 
            (as appropriate for FAC-only data such as ``space_mag_fac``).
        RI : float, optional
            Reference radius of the Gamera ionosphere [m].
            This radius separate the internal and external magnetic field calculations
        
        Returns
        -------
        B_east : array
            Eastward magnetic field component [T].
        B_north : array
            Northward magnetic field component [T].
        B_up : array
            Upward magnetic field component [T].

        # TODO: Account for elliptical Earth at some point
        """
    
        glon, glat, r = np.broadcast_arrays(np.asarray(glon, dtype=float), np.asarray(glat, dtype=float), np.asarray(r, dtype=float))
        shape = r.shape
        glon, glat, radius = glon.flatten(), glat.flatten(), r.flatten()

        # ------------------------#
        # Load spherical harmonic coefficients for the selected Gamera simulation timestep
        # ------------------------#

        base_dir = os.path.dirname(os.path.abspath(__file__))
        coeff_path = os.path.join(base_dir, 'magnetic_field_utils', 'B_coeffs')

        alpha_coeffs = np.load(coeff_path + f'/cfcoeff_Step#{self.timestep}.npy') 
        psi_coeffs   = np.load(coeff_path + f'/dfcoeff_Step#{self.timestep}.npy')

        if no_df_current:
            if np.any(radius < RI):
                print('Not a good idea to set no_df_current to True with r < RI')
            psi_coeffs *= 0

        # ------------------------#
        # Convert the measurement coordinates from geographic to magnetic coordinates
        # ------------------------#

        # Concert radial distance to altitude above Earth's reference radius
        height = radius*1e-3 - RE # [km]

        # Convert geographic coordinates to QD and modified Apex coordinates
        qdlat, qdlon = self.apex.geo2qd(glat, glon, height)
        alat, alon = self.apex.geo2apex(glat, glon, height)

        # Convert magnetic longitude to MLT at the analysis time
        qd_mlt = self.dp.mlon2mlt(qdlon, self.analysis_time)
        apex_mlt = self.dp.mlon2mlt(alon, self.analysis_time)

        # Shift MLT to match the Gamera coordinate convention, where phi=0 is at MLT=mlt0 #TODO KALLE OK???
        qd_gamera_mlt = (qd_mlt - self.mlt0) % 24
        apex_gamera_mlt = (apex_mlt - self.mlt0) % 24
        
        # ------------------------#
        # Evaluate the spherical-harmonic representation at the measurement coordinates
        # ------------------------#

        # Initialize the magnetic field array
        B = np.full((3, radius.size), np.nan) 

        # Separate locations inside and outside the Gamera ionosphere radius
        internal = radius < RI

        # Calculate the internal magnetic field (r < RI)
        if np.any(internal) > 0:

            r_ = radius[internal]

            # Set up the spherical-harmonic basis and evaluator at the measurement locations
            # grid = Grid(lat = qdlat[internal], lon = qd_mlt[internal] * 15)
            grid = Grid(lat = qdlat[internal], lon = qd_gamera_mlt[internal] * 15) #TODO KALLE OK???
            shbasis  = SHBasis(N, M)
            n = shbasis.n
            grid_evaluator = BasisEvaluator(shbasis, grid)

            # Evaluate the horizontal and radial components of the poloidal field
            kappa = psi_coeffs * (n + 1) / (2 * n + 1) * mu0
            Btheta, Bphi = (grid_evaluator.G_grad * np.expand_dims(r_/RI, -1)**(n-1)).dot(kappa)
            Br = (grid_evaluator.G * np.expand_dims(r_/RI, -1)**(n-1)).dot(kappa * n)

            B[0, internal] = Br
            B[1, internal] = Btheta
            B[2, internal] = Bphi

        # Calculate the external magnetic field (r >= RI)
        if np.any(~internal) > 0:

            r_ = radius[~internal]

            # Set up the spherical-harmonic basis and evaluator at the measurement locations
            # qd_grid = Grid(lat = qdlat[~internal], lon = qd_mlt[~internal] * 15)
            # apex_grid = Grid(lat = alat[~internal], lon = apex_mlt[~internal] * 15)
            qd_grid = Grid(lat = qdlat[~internal], lon = qd_gamera_mlt[~internal] * 15) #TODO KALLE OK???
            apex_grid = Grid(lat = alat[~internal], lon = apex_gamera_mlt[~internal] * 15) #TODO KALLE OK???
            shbasis  = SHBasis(N, M)
            n = shbasis.n
            qd_evaluator = BasisEvaluator(shbasis, qd_grid)
            apex_evaluator = BasisEvaluator(shbasis, apex_grid)

            # Evaluate the poloidal contribution
            kappa = -psi_coeffs * n / (2 * n + 1) * mu0
            Btheta_psi, Bphi_psi = (qd_evaluator.G_grad * np.expand_dims(RI/r_, -1)**(n+2)).dot(kappa)
            Br = (qd_evaluator.G * np.expand_dims(RI/r_, -1)**(n+2)).dot(-kappa * (n + 1))

            # Evaluate the toroidal contribution
            alpha = -alpha_coeffs * mu0 #/ (n * (n + 1))
            Btheta_alpha, Bphi_alpha = (apex_evaluator.G_rxgrad * np.expand_dims(RI / r_, -1)).dot(alpha)
            
            B[0, ~internal] = Br
            B[1, ~internal] = Btheta_psi
            B[2, ~internal] = Bphi_psi

        # The signs follow the coefficient convention used by the standalone forward calculation 
        # and the current-sheet boundary condition
        Br, Btheta, Bphi = -B

        # ------------------------#
        # Convert the magnetic field from magnetic to geographic coordinates
        # ------------------------#
        
        # Get the base vectors needed to transform the magnetic field to geographic components
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(glat, glon, height=height, coords='geo')
        f1, f2, d1, d2 = (np.asarray(v).reshape((-1, radius.size)) for v in (f1, f2, d1, d2))

        F = f1[0] * f2[1] - f1[1] * f2[0]
        B_east = Bphi * f2[1] + Btheta * f1[1]
        B_north = -Bphi * f2[0] - Btheta * f1[0]
        B_up = Br * np.sqrt(F)

        # Add the toroidal contribution to the geographic horizontal components at locations outside the ionosphere
        if np.any(~internal):
            sinI = 2 * np.sin(np.deg2rad(alat[~internal])) / np.sqrt(4 - 3 * np.cos(np.deg2rad(alat[~internal]))**2)
            bt = -Btheta_alpha
            bp = -Bphi_alpha
            B_east[~internal] += bt * d1[1, ~internal] - bp * sinI * d2[1, ~internal]
            B_north[~internal] += -bt * d1[0, ~internal] + bp * sinI * d2[0, ~internal]

        return B_east.reshape(shape), B_north.reshape(shape), B_up.reshape(shape)

    def get_Bigrf(self, glon, glat):
        """
        Calculate the IGRF magnetic field at measurement locations.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].

        Returns
        -------
        Bigrf_east : array
            Eastward magnetic field component [T].
        Bigrf_north : array
            Northward magnetic field component [T].
        Bigrf_up : array
            Upward magnetic field component [T].
        """

        # IGRF returns east, north, up components in nT 
        Bigrf_east, Bigrf_north, Bigrf_up = np.array(igrf(glon, glat, self.refh, self.analysis_time)) * 1e-9 # convert from [nT] to [T]

        return Bigrf_east, Bigrf_north, Bigrf_up

    def gamera_dipole_to_geo(self, time):
        """
        Convert Gamera magnetic coordinates to geographic coordinates.

        Parameters
        ----------
        time : datetime
            Time used to convert magnetic local time to magnetic longitude.

        Returns
        -------
        gamera_glat : array
            Geographic latitudes [degrees].
        gamera_glon : array
            Geographic longitudes [degrees].
        """

        mlon = self.dp.mlt2mlon(self.mlt, time)
        gamera_glat, gamera_glon, _ = self.apex.apex2geo(self.mlat, mlon, self.refh)

        return gamera_glat, gamera_glon
    
    def centered_csprojection(self, glon, glat):
        """
        Create a cubed-sphere projection centered on the input coordinates.

        Parameters
        ----------
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].

        Returns
        -------
        projection : lompe.cs.CSprojection
            Cubed-sphere projection centered on the mean direction of the input coordinates.
        """

        th = np.deg2rad(90 - glat).flatten()
        ph = np.deg2rad(glon).flatten()
        
        _r = np.mean(np.vstack((np.sin(th) * np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th))), axis = 1)
        _r = _r/np.linalg.norm(_r)

        mid_lon = np.rad2deg(np.arctan2(_r[1], _r[0]))
        mid_lat = np.rad2deg(np.arcsin(_r[2]))

        projection = cs.CSprojection((mid_lon, mid_lat), (1, 0))

        return projection

    def interp_to_measurements(self, projection, glon, glat, gamera_glon, gamera_glat, var=None, vec=None):
        """
        Interpolate a Gamera scalar or vector field to measurement locations.

        Parameters
        ----------
        projection : lompe.cs.CSprojection
            Cubed-sphere projection used for the interpolation.
        glon, glat : array
            Geographic longitudes and latitudes of the measurement locations [degrees].
        gamera_glon, gamera_glat : array
            Geographic coordinates of the Gamera grid [degrees].
        var : array, optional
            Scalar Gamera field to interpolate.
        vec : tuple of array, optional
            Vector Gamera field given as geographic eastward and northward compoments ``(A_east, A_north)``.

        Returns
        -------
        array or tuple of array
            Interpolated scalar field, or interpolated eastward and northward vector components.
        """

        # Project measurement coordinates to the cubed sphere
        xi, eta = projection.geo2cube(glon.flatten(), glat.flatten(), set_points_off_cube_to_nan=True) 
        
        # ---------------------------------------------------------------------
        # SCALAR INTERPOLATION
        # ---------------------------------------------------------------------
        if var is not None and vec is None:

            # Keep valid Gamera source points (remove NaNs if any)
            mask = (np.isfinite(var.flatten()))

            gamlon_valid = gamera_glon.flatten()[mask]
            gamlat_valid = gamera_glat.flatten()[mask]
            var_valid = var.flatten()[mask]

            # Project Gamera coordinates to the cubed sphere
            xiG, etaG = projection.geo2cube(gamlon_valid, gamlat_valid)
            
            # Remove NaNs (in case of projection failures)
            good = np.isfinite(xiG) & np.isfinite(etaG)  

            # Interpolate from Gamera locations xiG/etaG to measurement locations xi/eta
            var_interp = griddata((xiG[good], etaG[good]), var_valid[good], (xi, eta), method="linear")

            return var_interp.reshape(glon.shape)

        # ---------------------------------------------------------------------
        # VECTOR INTERPOLATION (east, north)
        # ---------------------------------------------------------------------
        if vec is not None and var is None:

            A_east, A_north = vec

            # Keep valid Gamera source points (remove NaNs if any)
            mask = (np.isfinite(A_east.flatten()) & np.isfinite(A_north.flatten()))

            gamlon_valid = gamera_glon.flatten()[mask]
            gamlat_valid = gamera_glat.flatten()[mask]
            Ae_valid = A_east.flatten()[mask]
            An_valid = A_north.flatten()[mask]

            # Project Gamera vector components and coordinates to the cubed sphere
            xiG, etaG, A_xi, A_eta = projection.vector_cube_projection(Ae_valid, An_valid, gamlon_valid, gamlat_valid)

            # Remove NaNs (in case of projection failures)
            good = (np.isfinite(xiG) & np.isfinite(etaG) &
                    np.isfinite(A_xi) & np.isfinite(A_eta))

            # Interpolate vector components
            A_xi_interp = griddata((xiG[good], etaG[good]), A_xi[good], (xi, eta), method="linear")
            A_eta_interp = griddata((xiG[good], etaG[good]), A_eta[good], (xi, eta), method="linear")

            # Convert interpolated vector components back to geographic coordinates
            _, _, A_east_interp, A_north_interp = projection.vector_cube_to_geo(A_xi_interp, A_eta_interp, xi, eta)

            return (A_east_interp.reshape(glon.shape), A_north_interp.reshape(glon.shape))

        raise ValueError("Provide either var=<scalar> or vec=(Ae, An).")





if __name__ == '__main__':
    # Validate the Gamera potential interpolation by comparing it with the original Gamera field

    import datetime as dt
    import matplotlib.pyplot as plt
    from polplot import Polarplot
    import numpy as np

    from lompeosse.gamera_data import GameraData

    event_time = dt.datetime(2012, 4, 5, 1, 19)
    analysis_time = event_time

    go = GameraData(analysis_time, timestep=0, hemisphere='NORTH')

    # Validate that interpolating the Gamera field to its own geographic grid recovers the original field
    psi = go.get_potential(go.gamera_glon, go.gamera_glat) / 1e3

    # Original Gamera potential
    opsi = go.gamera_data['Potential']

    # Gamera grid coordinates
    mlat = go.mlat
    mlt = go.mlt

    fig, axes = plt.subplots(ncols=2)
    paxes = list(map(Polarplot, axes))
    paxes[0].contour(mlat, mlt, psi, levels=np.r_[-300:300:30])
    paxes[1].contour(mlat, mlt, opsi, levels=np.r_[-300:300:30])
    plt.show()