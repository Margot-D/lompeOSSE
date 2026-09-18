import h5py
import os
import numpy as np
import datetime as dt
import apexpy
from scipy.interpolate import griddata
from ppigrf import igrf
import dipole # github.com/klaundal/dipole
from secsy import cubedsphere as cs
from pathlib import Path

from .magnetic_field_utils.sh_basis import SHBasis
from .magnetic_field_utils.grid import Grid
from .magnetic_field_utils.basis_evaluator import BasisEvaluator

mu0 = np.pi * 4e-7
RI_GAMERA = 6500*1e3 #  Ionospheric radius in [m] ???

# spherical harmonic analysis
N, M = 110, 110

RE = 6371.2 # Earth radius in km
RI = 6500 # Ionospheric radius in km (used in Gamera simulations)

class GameraData(object):

    """
    A helper class for loading, organizing, and working with Gamera simulation data in LompeOSSE.

    The GameraData module enables the automatic acquisition of synthetic data from 
Gamera simulation, 

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

    def __init__(self, time, timestep=0, hemisphere='NORTH'):

        self.time = time # (used in lompeosse)
        self.timestep = timestep
        self.hemisphere = hemisphere

        self.dp = dipole.Dipole(time.year)
        self.refh = RI-RE # reference height of the Gamera output [km]
        self.apex = apexpy.Apex(time, self.refh)

        # Support both a source checkout and data bundled with an installed package.
        package_root = Path(__file__).resolve().parent
        candidates = [package_root / "data" / "Gamera_data.h5"]
        if package_root.parent.name == "src":
            candidates.insert(0, package_root.parent.parent / "data" / "Gamera_data.h5")
        datapath = next((path for path in candidates if path.is_file()), None)

        if datapath is None:
            raise FileNotFoundError(
                "Gamera_data.h5 was not found. Searched:\n"
                + "\n".join(str(path) for path in candidates)
                + "\nDownload it from https://zenodo.org/records/16882035 and place it at one of these paths.")

        # Load Gamera data
        self.gamera_data = self._load_Gamera_data(datapath)

    def _load_Gamera_data(self, fn):
        """
        TODO fix text
        Reads Gamera data from an HDF5 file, extracts relevant variables based on 
        the specified timestep and hemisphere, converts magnetic dipole coordinates to geographic...

        Returns:
        --------
        gamera_data: dict
            A dictionary containing Gamera data, ready for use in lompeOSSE
        """

        print(f'Loading Gamera data/snapshot #{self.timestep}')

        gamera_data = {}

        # ----------
        # Open HDF5 file and load Gamera dataset

        with h5py.File(fn, "r") as f:
            gamera_data['X'] = f['X'][:]
            gamera_data['Y'] = f['Y'][:]
            for step in f['Step#%d' % self.timestep].keys():
                gamera_data[step] = f['Step#%d' % self.timestep][step][:]

        # ----------
        # Filter Gamera dataset by hemisphere

        h = self.hemisphere.upper()
        hemi_gamera_data = {}

        for key in gamera_data.keys():
            key_lower = key.lower()

            if h == "NORTH":
                if "north" in key_lower:
                    new_key = key.replace("NORTH", "").strip()
                    hemi_gamera_data[new_key] = gamera_data[key]

                elif "south" not in key_lower:
                    hemi_gamera_data[key] = gamera_data[key]

            elif h == "SOUTH":
                if "south" in key_lower:
                    new_key = key.replace("SOUTH", "").strip()
                    hemi_gamera_data[new_key] = gamera_data[key]

                elif "north" not in key_lower:
                    hemi_gamera_data[key] = gamera_data[key]

        # Update gamera_data with hemisphere-filtered values
        gamera_data.clear()
        gamera_data.update(hemi_gamera_data)

        # ----------
        # Derive useful variables

        # Convert Cartesian (X, Y) to spherical coordinates (R, THETA, PHI)
        x = gamera_data['X'] # TODO in Earth's radius?
        y = gamera_data['Y']
        theta = np.arcsin(np.sqrt(x**2 + y**2)) # colatitude in radians
        phi = np.arctan2(y, x) # azimuthal angle in radians (theta column in remix file)

        # # Normalize azimuthal angles to [0 - 2pi] # TODO check if useful with Kalle 
        # phi[phi < 0] = phi[phi < 0] + 2*np.pi
        # phi[:, 0] -= 2 * np.pi  # Adjust first column
        
        # TODO discuss that with Kalle (compare with get_E), and find better names?
        # Correct r, theta and phi to match other variables in Gamera data file
        # Compute grid-centered spherical coordinates
        self.theta_trim = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1] /2 # averaged over theta respective grid directions
        self.phi_trim   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :] /2 # averaged over phi respective grid directions

        self.mlt = self.phi_trim * (12/np.pi) +0 #TODO fix.... # convert azimuthal angle in radians to MLT hours

        # mlt = phi_trim * (12/np.pi) #+ (12 - mlt_off) # in hours NOTE The (12-mlt_off) shift ensures a total of 12 hour shift that aligns MLT midnight 
        #                                         # with the nightside as defined in 
        #                                         # the original Gamera dataset.

        # ----------
        # Exclude low latitude points in Gamera data #TODO not useful?

        mlatG = 90 - np.rad2deg(self.theta_trim) # in degrees
        if self.hemisphere == 'SOUTH': mlatG = (-1)*mlatG #TODO ok?

        # min_lat = 50 # 20  # Should be at least 11 deg
        # in_bounds = np.abs(mlatG) > min_lat

        # for key in gamera_data.keys():
        #     if gamera_data[key].shape == mlatG.shape:  
        #         gamera_data[key] = np.where(in_bounds, gamera_data[key], np.nan) # Apply NaN to out-of-bounds data

        # if np.abs(mlatG).min() < min_lat:
        #     print('gamera mlat.min(): ', np.abs(mlatG).min(), 'degrees')
        #     print(f"Low latitude Gamera data (< {min_lat} deg) has been discarded")
        
        # self.mlat = np.where(in_bounds, mlatG, np.nan) #TODO check if problem that it has nan in it...
        self.mlat = mlatG
        self.theta = theta # [radians]
        self.phi = phi # [radians]

        return gamera_data


    def _get_scalar_parameter(self, glon, glat, time, param):
        """
        glat, glon are the measurement coordinates
        param: str name of the quantity (potential, Hall conductance...)

        """
        print('Retrieving Gamera', param)

        z = self.gamera_data[param]

        gamera_glat, gamera_glon = self.gamera_dipole_to_geo(time)
        projection = self.centered_csprojection(glon, glat)

        return self.interp_to_measurements(projection, glon, glat, gamera_glon, gamera_glat, var=z)


    def get_potential(self, glon, glat, time):
        return(self._get_scalar_parameter(glon, glat, time, 'Potential') *1e3) # Convert [kV] to [V]

    def get_FAC(self, glon, glat, time):
        fac_parallel = self._get_scalar_parameter(glon, glat, time, 'Field-aligned current') * 1e-6

        # The Gamera/REMIX quantity is positive along the background magnetic
        # field. Lompe defines FAC as positive upward. The dipole field points
        # into Earth in the north and out of Earth in the south.
        upward_sign = -1 if self.hemisphere.upper() == 'NORTH' else 1

        return upward_sign * fac_parallel # [A/m²], positive upward

    def get_Hall(self, glon, glat, time):
        return(self._get_scalar_parameter(glon, glat, time, 'Hall conductance')) # in [S]

    def get_Pedersen(self, glon, glat, time):
        return(self._get_scalar_parameter(glon, glat, time, 'Pedersen conductance')) # in [S]
    

    def get_E(self, glon, glat, time, hI = 110): #TODO where should hI be used?
        """
        Compute the Gamera electric field at Gamera grid points, then transform it into geodetic coordinates 
        and finally interpolate at measurement glon, glat. 

        glon, glat: measurement geographic locations
        hI: ionospheric height in km

        TODO write something like: This is largely based on the Kaipy module but also integrate the conversion from magnetic dipole to geographic coordinates.

        Returns:
        --------
        tuple: (E_east, E_north) 
            Electric field components in [V/m] in the geographic eastward and northward directions at measurement locations lon/lat.

        """
        
        print('Retrieving Gamera electric field')

        x = self.gamera_data['X']
        theta_trim = self.theta_trim
        phi_trim = self.phi_trim
        theta = self.theta
        phi = self.phi
        Psi = self.gamera_data['Potential'] # [kV]

        #-----------
        # First, compute Gamera electric field 

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

        # TODO use these etheta and ephi in get_B?

        #-----------
        # Then, convert from magnetic dipole to geographic geodetic coordinates

        # gamera_glat, gamera_glon = self.gamera_dipole_to_geo(phi_trim, theta_trim, time)
        gamera_glat, gamera_glon = self.gamera_dipole_to_geo(time)
        
        # Compute APEX base vectors #TODO fix calculations?
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(gamera_glat.flatten(), gamera_glon.flatten(), height = 6500-6371.2, coords = 'geo')

        # Compute magnetic field inclination
        sinIm = 2 * np.sin(np.deg2rad(self.mlat.flatten())) / np.sqrt(4 - 3 * np.cos(np.deg2rad(self.mlat.flatten()))**2) 

        # Convert to geographic geodetic coordinates using the d1 and d2 base vectors
        E_mag_east, E_mag_north = ephi, -etheta # Convert to east-north components
        E_geo_east, E_geo_north, E_geo_up = E_mag_east.flatten() * d1 - (E_mag_north.flatten()/sinIm) * d2 #TODO OK?
        
        # TODO check if flatten() could be removed everywhere?

        #-----------
        # Finally, interpolate Gamera electric field (geographic components) from the Gamera grid to the measurement locations (glon, glat)
        
        projection = self.centered_csprojection(glon, glat)

        E_east, E_north = self.interp_to_measurements(projection, glon, glat, gamera_glon, gamera_glat, vec=(E_geo_east, E_geo_north))

        return E_east, E_north # East, north Gamera electric field in geocentric coordinates at measurement locations in [V/m]


    def get_V(self, glon, glat, time): #TODO should it take hI too?
        """
        Compute the ExB drift velocity (east and north components) 
        from the Gamera electric field and the IGRF magnetic field.

        Returns:
        --------
        tuple: (V_east, V_north)
            Plasma drift velocity in [m/s] in the eastward and northward directions
        """
        print('Retrieving Gamera convection data')

        # Electric field 
        E_east, E_north = self.get_E(glon, glat, time) #TODO do we need a radius or height as input here?
        Eph = E_east # azimuthal
        Eth = -E_north # polar

        # Geomagnetic field components and magnitude
        Br, Bph, Bth = self.get_Bigrf(glon, glat, time)
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

        return V_east, V_north # in [m/s]

    def get_B(self, glon, glat, r, no_df_current = False, RI = RI_GAMERA, time = None):
        """
        Compute the magnetic field...
            
            RI_GAMERA is the ionosphere radius. r < RI_GAMERA is considered internal, r > RI_GAMERA is considered external
        
            theta, phi in degrees

            RI is the modeled current-sheet radius in meters. The stored coefficients
            and the Gamera/Lompe current grid use 6500 km.
            time sets the MLT orientation and defaults to the object's time.

            no_df_current: Set to True for 'space_mag_fac' data type (e.g, Iridium)

            # logic (different from E):
            1) convert input to magnetic
            2) interpolate gamera to input coords (using SH) in magnetic coords
            3) convert output to geographic and return

        # TODO: Account for elliptical Earth at some point

        Returns:
        --------
        tuple: (B_east, B_north, B_up)
            Magnetic field in [T] in the eastward, northward and upward directions,
            with the broadcast shape of glon, glat and r. Scalars are supported.
        """

        nstep = self.timestep

        print('Retrieving Gamera magnetic field data')

        glon, glat, r = np.broadcast_arrays(np.asarray(glon, dtype=float), np.asarray(glat, dtype=float), np.asarray(r, dtype=float))
        shape = r.shape
        glon, glat, r = glon.flatten(), glat.flatten(), r.flatten()

        #-----------
        # Load spherical harmonic coefficients for given Gamera simulation timestep

        base_dir = os.path.dirname(os.path.abspath(__file__))
        coeff_path = os.path.join(base_dir, 'magnetic_field_utils', 'B_coeffs')

        alpha_coeffs = np.load(coeff_path + f'/cfcoeff_Step#{nstep}.npy') 
        psi_coeffs   = np.load(coeff_path + f'/dfcoeff_Step#{nstep}.npy')

        if no_df_current:
            if np.any(r < RI):
                print('Not a good idea to set no_df_current to True with r < RI')
            psi_coeffs *= 0

        #-----------
        # Calculate magnetic field at measurement coordinates

        # Interpret Gamera's dipole coordinates as magnetic Apex coordinates. The
        # poloidal potential uses QD latitude, while the toroidal potential uses MA.
        height = r - 6371.2e3 # height of the ionosphere [meters]
        qdlat, qdlon = self.apex.geo2qd(glat, glon, height * 1e-3)
        alat, alon = self.apex.geo2apex(glat, glon, height * 1e-3)

        # Invert the MLT-to-longitude conversion used by gamera_dipole_to_geo.
        if time is None: time = self.time
        qd_mlt = self.dp.mlon2mlt(qdlon, time)
        apex_mlt = self.dp.mlon2mlt(alon, time)

        radius = r

        # Initialize array to hold magnetic field
        B = np.full((3, radius.size), np.nan) 

        # Use spherical harmonic analysis to compute B 
        iii = radius < RI
        if np.sum(iii) > 0: # internal:
            print('(ground)')
            r_ = radius[iii]

            grid = Grid(lat = qdlat[iii], lon = qd_mlt[iii] * 15)
            shbasis  = SHBasis(N, M)
            n = shbasis.n
            grid_evaluator = BasisEvaluator(shbasis, grid)

            kappa = psi_coeffs * (n + 1) / (2 * n + 1) * mu0
            # Both radial and horizontal derivatives of the potential scale as r**(n-1).
            Btheta, Bphi = (grid_evaluator.G_grad * np.expand_dims(r_/RI, -1)**(n-1)).dot(kappa)
            Br = (grid_evaluator.G * np.expand_dims(r_/RI, -1)**(n-1)).dot(kappa * n)

            # print(Btheta.min(), Btheta.max(), Bphi.min(), Bphi.max())

            B[0, iii] = Br
            B[1, iii] = Btheta
            B[2, iii] = Bphi

        # print('coeffs: ', np.linalg.norm(psi_coeffs), np.linalg.norm(alpha_coeffs))

        if np.sum(~iii) > 0: # external:
            print('(space)')
            r_ = radius[~iii]

            qd_grid = Grid(lat = qdlat[~iii], lon = qd_mlt[~iii] * 15)
            apex_grid = Grid(lat = alat[~iii], lon = apex_mlt[~iii] * 15)
            shbasis  = SHBasis(N, M)
            n = shbasis.n
            qd_evaluator = BasisEvaluator(shbasis, qd_grid)
            apex_evaluator = BasisEvaluator(shbasis, apex_grid)

            kappa = -psi_coeffs * n / (2 * n + 1) * mu0
            Btheta_psi, Bphi_psi = (qd_evaluator.G_grad * np.expand_dims(RI/r_, -1)**(n+2)).dot(kappa)
            Br = (qd_evaluator.G * np.expand_dims(RI/r_, -1)**(n+2)).dot(-kappa * (n + 1))

            alpha = -alpha_coeffs * mu0 #/ (n * (n + 1))
            Btheta_alpha, Bphi_alpha = (apex_evaluator.G_rxgrad * np.expand_dims(RI / r_, -1)).dot(alpha)
            
            # print(Btheta_psi.min(), Btheta_psi.max(), Bphi_psi.min(), Bphi_psi.max(), Br.min(), Br.max())
            # print(Btheta_alpha.min(), Btheta_alpha.max(), Bphi_alpha.min(), Bphi_alpha.max())

            B[0, ~iii] = (Br)
            B[1, ~iii] = Btheta_psi
            B[2, ~iii] = Bphi_psi

        # The signs follow the coefficient convention used by the standalone
        # forward calculation and the current-sheet boundary condition.
        Br, Btheta, Bphi = -B

        #-----------
        # Convert output from magnetic dipole to geographic geocentric coordinates

        # Apply the magnetic-Apex vector formulas. The poloidal gradient is
        # represented in QD coordinates by f1/f2. The toroidal field is
        # represented in modified Apex coordinates by d1/d2.
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.apex.basevectors_apex(glat, glon, height=height* 1e-3, coords = 'geo') #TODO 8 feb just added the 1e-3 (height in km), correct?
        f1, f2, d1, d2 = (np.asarray(v).reshape((-1, radius.size)) for v in (f1, f2, d1, d2))

        F = f1[0] * f2[1] - f1[1] * f2[0]
        B_east = Bphi * f2[1] + Btheta * f1[1]
        B_north = -Bphi * f2[0] - Btheta * f1[0]
        B_up = Br * np.sqrt(F)

        if np.any(~iii):
            sinI = 2 * np.sin(np.deg2rad(alat[~iii])) / np.sqrt(4 - 3 * np.cos(np.deg2rad(alat[~iii]))**2)
            bt = -Btheta_alpha
            bp = -Bphi_alpha
            B_east[~iii] += bt * d1[1, ~iii] - bp * sinI * d2[1, ~iii]
            B_north[~iii] += -bt * d1[0, ~iii] + bp * sinI * d2[0, ~iii]

        return B_east.reshape(shape), B_north.reshape(shape), B_up.reshape(shape) # in [T]


    def get_Bigrf(self, glon, glat, time):
        """
        Compute the geodetic IGRF magnetic field components at the measurement locations.

        Returns:
        --------
        B0 : (3, N) ndarray
            Magnetic field components in testla (T), with:
            - B0[0] = Br  (radial component)
            - B0[1] = Bph (azimuthal/eastward component)
            - B0[2] = Bth (polar/southward component)
        """
        
        # Compute the IGRF main field components in nT
        Be, Bn, Bup = np.array(igrf(glon, glat, self.refh, time)) * 1e-9 # igrf returns east, north, up components in nT 

        return np.vstack((Bup, Be, -Bn)) # radial, east (phi), south (theta)


    def gamera_dipole_to_geo(self, time):
    # def gamera_dipole_to_geo(self, phi, theta, time):

        """
        Convert Gamera coordinates from from centered-dipole (magnetic) to geocentric (geographic),
        taking the specified time into account.

        phi: longitude ish
        theta: colatitude 
        """

        # mlt = phi * (12/np.pi) # convert phi in radians to MLT hours
        mlon = self.dp.mlt2mlon(self.mlt, time) # convert MLT hours to dipole longitude at the given time
        # mlat = 90 - np.rad2deg(theta) # convert theta in radians to dipole latitude in degrees
        # if self.hemisphere == 'SOUTH': mlat = (-1)*mlat 
        gamera_glat, gamera_glon, _ = self.apex.apex2geo(self.mlat, mlon, self.refh)

        return gamera_glat, gamera_glon
    

    def centered_csprojection(self, glon, glat):
        
        # find the mid point of the input coordinates to make a projection centered at that point

        th = np.deg2rad(90 - glat).flatten()
        ph = np.deg2rad(glon).flatten()
        
        _r = np.mean(np.vstack((np.sin(th) * np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th))), axis = 1)
        _r = _r/np.linalg.norm(_r)

        mid_lon = np.rad2deg(np.arctan2(_r[1], _r[0]))
        mid_lat = np.rad2deg(np.arcsin(_r[2]))

        projection = cs.CSprojection((mid_lon, mid_lat), (1, 0))

        return projection


    def interp_to_measurements(self, projection, lon, lat, gamera_lon, gamera_lat, var=None, vec=None):
        """
        Interpolate a scalar or vector field from the Gamera grid to measurement locations (lon, lat).

        Parameters
        ----------
        grid : lompe.Grid
            Target cubed-sphere grid.
        lon, lat : ndarray
            Measurement coordinates (deg).
        gamera_lon, gamer_lat: 
            Gamera coordinates
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

        # Valid source region above latlim
        latlim = 40

        # --- Project measurement coords to cubed sphere
        xi, eta = projection.geo2cube(lon.flatten(), lat.flatten(), set_points_off_cube_to_nan=True) 
        
        # ---------------------------------------------------------------------
        # SCALAR INTERPOLATION
        # ---------------------------------------------------------------------
        if var is not None and vec is None:

            # Keep valid source points
            mask = ((np.abs(gamera_lat.flatten()) >= latlim) & # latitude filter
                    np.isfinite(var.flatten())) # remove NaNs if any

            gamlon_valid = gamera_lon.flatten()[mask]
            gamlat_valid = gamera_lat.flatten()[mask]
            var_valid = var.flatten()[mask]

            # Project Gamera coords glon/glat to cubed sphere
            xiG, etaG = projection.geo2cube(gamlon_valid, gamlat_valid)
            
            # Remove NaNs (in case of projection failures)
            good = np.isfinite(xiG) & np.isfinite(etaG)  

            # Interpolate from Gamera locations xiG/etaG to measurement locations xi/eta
            var_interp = griddata((xiG[good], etaG[good]), var_valid[good], (xi, eta), method="linear")

            return var_interp.reshape(lon.shape)

        # ---------------------------------------------------------------------
        # VECTOR INTERPOLATION (east, north)
        # ---------------------------------------------------------------------
        if vec is not None and var is None:

            A_east, A_north = vec

            # Keep valid source points
            mask = ((np.abs(gamera_lat.flatten()) >= latlim) & # latitude filter
                np.isfinite(A_east.flatten()) & # remove NaNs if any
                np.isfinite(A_north.flatten()))

            gamlon_valid = gamera_lon.flatten()[mask]
            gamlat_valid = gamera_lat.flatten()[mask]
            Ae_valid = A_east.flatten()[mask]
            An_valid = A_north.flatten()[mask]

            # Project Gamera vector and coords to cubed sphere
            xiG, etaG, A_xi, A_eta = projection.vector_cube_projection(Ae_valid, An_valid, gamlon_valid, gamlat_valid)

            # Remove NaNs (in case of projection failures)
            good = (np.isfinite(xiG) &
                    np.isfinite(etaG) &
                    np.isfinite(A_xi) &
                    np.isfinite(A_eta))

            # Interpolate vector components
            A_xi_interp = griddata((xiG[good], etaG[good]), A_xi[good], (xi, eta), method="linear")

            A_eta_interp = griddata((xiG[good], etaG[good]), A_eta[good], (xi, eta), method="linear")

            # Convert back to geographic coordinates
            _, _, A_east_interp, A_north_interp = projection.vector_cube_to_geo(A_xi_interp, A_eta_interp, xi, eta)

            return (A_east_interp.reshape(lon.shape),
                    A_north_interp.reshape(lon.shape))

        raise ValueError("Provide either var=<scalar> or vec=(Ae, An).")



import pandas as pd
import lompe
import datetime as dt
import matplotlib.pyplot as plt
from polplot import Polarplot

# validate interpolation
if __name__ == '__main__':
    import polplot

    event = '2012-04-05'
    hour = 1
    minute = 19

    # Derived parameters 
    event_date = event.replace('-', '') # format YYYYMMDD
    time = pd.to_datetime(event)
    stime = dt.datetime(time.year, time.month, time.day, hour, minute) # specific time to model


    mlt_offset = 9 # [hours] TODO find a way to deal with that (in lompeosse instead of here)
    ntime = stime + dt.timedelta(hours=mlt_offset)


    go = GameraData(ntime, timestep = 0, hemisphere = 'NORTH')
    
    time = dt.datetime(2020, 1, 1, 10)
    dp = dipole.Dipole(time.year)
    apx = apexpy.Apex(time, RI-RE)

    mlat, mlt = np.meshgrid(np.linspace(50, 90, 23), np.linspace(0, 24, 23))
    mlon = dp.mlt2mlon(mlt, time)

    glat, glon, err = apx.apex2geo(mlat, mlon, 110)

    psi = go.get_potential(glon, glat, time) # using interpolation function
    opsi = go.gamera_data['Potential'] # original gamera potential

    fig, axes = plt.subplots(ncols = 2)
    paxes = list(map(Polarplot, axes))
    paxes[0].contour(mlat, mlt, psi, levels = np.r_[-300:300:30]) # interpolated gamera potential
    paxes[1].contour(90 - np.rad2deg(go.theta_trim), go.phi_trim*(12/np.pi), opsi, levels = np.r_[-300:300:30]) # original gamera variables
    plt.show()
