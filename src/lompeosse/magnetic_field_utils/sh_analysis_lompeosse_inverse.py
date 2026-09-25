""" 
    This script:
        - performs spherical harmonic analysis of REMIX horizontal ionospheric currents
        - uses a lot of memory because of the large number of SH coefficients
        - is included for completion, it is not intended to be run by users
        - uses N=M=110 and additive Pedersen/Hall currents, consistent with the stored coefficients
        - writes candidates to B_coeffs_regenerated for comparison before replacing stored files


"""

import numpy as np
import h5py
import dipole
from scipy.linalg import cho_factor, cho_solve
from pathlib import Path
import argparse
from sh_basis import SHBasis
from grid import Grid
from basis_evaluator import BasisEvaluator
#import pynamit # https://github.com/DynaMIT-uib/PynaMIT
import matplotlib.pyplot as plt
dp = dipole.Dipole(2020)
mu0 = 4 * np.pi * 1e-7


def magnetic_energy_weights(shbasis, radius, ri):
        """Return relative magnetic-energy weights for the SH coefficients.

        The weights are the analytic surface integral of |B|^2 for a
        unit-amplitude coefficient on a spherical shell at ``radius``. The
        returned weights are divided by their median, so that the inversion
        option using them has a dimensionless, convenient strength. This
        normalization does not change the preferred coefficient ratios.

        The field expressions match GameraData.get_B. A shell immediately
        above the current sheet includes both the curl-free and
        divergence-free current contributions.
        """

        # Integrate each Schmidt-normalized SH basis function over the unit
        # sphere with Gauss-Legendre quadrature.  2*N+1 nodes exactly resolve
        # the squared degree-N functions.
        nodes, node_weights = np.polynomial.legendre.leggauss(2 * shbasis.n.max() + 1)
        P = shbasis.legendre(np.arccos(nodes))
        if shbasis.schmidt_normalization:
                P = P * shbasis.schmidt_factors

        Pc = P[:, shbasis.cnm_filter]
        Ps = P[:, shbasis.snm_filter]
        cosine_phi_integral = np.pi * (1 + (shbasis.cnm.m.flatten() == 0))
        sine_phi_integral   = np.pi * np.ones(shbasis.snm.m.size)
        harmonic_norm = np.hstack((
                np.sum(node_weights[:, None] * Pc**2, axis = 0) * cosine_phi_integral,
                np.sum(node_weights[:, None] * Ps**2, axis = 0) * sine_phi_integral,
        ))

        n = shbasis.n.flatten()
        if radius < ri:
                # Below the sheet only the divergence-free/poloidal field is
                # present in this model.
                energy_cf = np.zeros_like(n, dtype = float)
                scale_df = mu0 * (n + 1) / (2 * n + 1) * (radius / ri)**(n - 1)
                energy_df = scale_df**2 * n * (2 * n + 1) * harmonic_norm
        else:
                # Immediately above the sheet both the curl-free/toroidal and
                # divergence-free/poloidal fields contribute.
                scale_cf = mu0 * ri / radius
                energy_cf = scale_cf**2 * n * (n + 1) * harmonic_norm
                scale_df = mu0 * n / (2 * n + 1) * (ri / radius)**(n + 2)
                energy_df = scale_df**2 * (n + 1) * (2 * n + 1) * harmonic_norm

        energy = np.hstack((energy_cf, energy_df))
        return energy / np.median(energy[energy > 0])

# copied from the remix code
def efield(x, y, Psi, returnDeltas=False, ri = 6.5*1e3):
        """
        Calculate the electric field at each point in the grid.
        Args:
            returnDeltas (bool, optional): Whether to return the differences in theta and phi along with the electric field.
            ri (float, optional): The value of Ri multiplied by 1e3.
        Returns:
            tuple: A tuple containing the electric field components (-etheta, -ephi) in V/m.
                   If `returnDeltas` is True, it also includes the differences in theta and phi (dtheta, dphi).
        Raises:
            SystemExit: If the variables have not been initialized for the specific hemisphere.
        Note:
            This method assumes that the variables have been initialized for the specific hemisphere
            by calling the `init_var` method prior to calculating the electric field.
        """
        Nt,Np = Psi.shape
        theta = np.arcsin(np.sqrt(x**2 + y**2))
        phi   = np.arctan2(y, x)
        # interpolate Psi to corners
        Psi_c = np.zeros(x.shape)
        Psi_c[1:-1,1:-1] = 0.25*(Psi[1:,1:]+Psi[:-1,1:]+Psi[1:,:-1]+Psi[:-1,:-1])
        # fix up periodic
        Psi_c[1:-1,0]  = 0.25*(Psi[1:,0]+Psi[:-1,0]+Psi[1:,-1]+Psi[:-1,-1])
        Psi_c[1:-1,-1] = Psi_c[1:-1,0]
        # fix up pole
        Psi_pole = Psi[0,:].mean()
        Psi_c[0,1:-1] = 0.25*(2.*Psi_pole + Psi[0,:-1]+Psi[0,1:])
        Psi_c[0,0]    = 0.25*(2.*Psi_pole + Psi[0,-1]+Psi[0,0])
        Psi_c[0,-1]   = 0.25*(2.*Psi_pole + Psi[0,-1]+Psi[0,0])
        # fix up low lat boundary
        # extrapolate linearly just like we did for the coordinates
        # (see genOutGrid in src/remix/mixio.F90)
        # note, neglecting the possibly non-uniform spacing (don't care)
        Psi_c[-1,:] = 2*Psi_c[-2,:]-Psi_c[-3,:]
        # now, do the differencing
        # for each cell corner on the original grid, I have the coordinates and Psi_c
        # need to find the gradient at cell center
        # the result is the same size as Psi
        # first etheta
        tmp    = 0.5*(Psi_c[:,1:]+Psi_c[:,:-1])  # move to edge center
        dPsi   = tmp[1:,:]-tmp[:-1,:]
        tmp    = 0.5*(theta[:,1:]+theta[:,:-1])
        dtheta = tmp[1:,:]-tmp[:-1,:]
        etheta = dPsi/dtheta/ri  # this is in V/m
        # now ephi
        tmp    = 0.5*(Psi_c[1:,:]+Psi_c[:-1,:])  # move to edge center
        dPsi   = tmp[:,1:]-tmp[:,:-1]
        tmp    = 0.5*(phi[1:,:]+phi[:-1,:])
        dphi   = tmp[:,1:]-tmp[:,:-1]
        tc = 0.25*(theta[:-1,:-1]+theta[1:,:-1]+theta[:-1,1:]+theta[1:,1:]) # need this additionally
        ephi = dPsi/dphi/np.sin(tc)/ri  # this is in V/m
        if returnDeltas:
            return (-etheta,-ephi,dtheta,dphi)  # E = -grad Psi
        else:
            return (-etheta,-ephi)  # E = -grad Psi


script_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafile', type=Path, default=script_dir.parents[2] / 'data' / 'Gamera_data.h5')
parser.add_argument('--output-dir', type=Path, default=script_dir / 'B_coeffs_regenerated')
parser.add_argument('--steps', nargs='+', default=['Step#0', 'Step#13', 'Step#19', 'Step#20', 'Step#21'])
parser.add_argument('--magnetic-energy-weight', type=float, default=0.,
                    help = 'Dimensionless strength of the surface magnetic-energy penalty. 0 disables it.')
parser.add_argument('--energy-radius-km', type=float, default=6501.,
                    help = 'Radius of the spherical shell used for the energy penalty [km].')
args = parser.parse_args()
datafile = args.datafile
steps = args.steps
args.output_dir.mkdir(parents=True, exist_ok=True)

if args.magnetic_energy_weight < 0:
	parser.error('--magnetic-energy-weight must be non-negative')

data = h5py.File(datafile, 'r')
# read coords and calculate angles
x = data['X'][:]
y = data['Y'][:]
theta = np.arcsin(np.sqrt(x**2 + y**2))
phi   = np.arctan2(y, x)
theta = theta[:-1, :-1] + np.diff(theta, axis = 0)[:, :-1]/2
phi   = phi[:-1, :-1] + np.diff(phi, axis = 1)[:-1, :]/2


for count, step in enumerate(steps):
	print('doing step ' + step)
	# read data
	data_step = data[step]
	Hall_n = data_step['Hall conductance NORTH'][:]
	Hall_s = data_step['Hall conductance SOUTH'][:]
	Pedersen_n = data_step['Pedersen conductance NORTH'][:]
	Pedersen_s = data_step['Pedersen conductance SOUTH'][:]
	Phi_n = data_step['Potential NORTH'][:]
	Phi_s = data_step['Potential SOUTH'][:]

	# calculate the electric field
	Eth_n, Eph_n = efield(x, y, Phi_n)
	Eth_s, Eph_s = efield(x, y, Phi_s)
	Eth = np.hstack((Eth_n.flatten(), Eth_s.flatten()))
	Eph = np.hstack((Eph_n.flatten(), Eph_s.flatten()))
	Eh  = np.vstack((Eth, Eph)) # 2 x N array with theta and phi components in the rows


	# calculate the current - but first I need the main magnetic field unit vector (dipole field)
	#   and the third component of E:
	lat = np.hstack((90 - np.rad2deg(theta).flatten(), -90 + np.rad2deg(theta).flatten()))
	lon = np.hstack((np.rad2deg(phi).flatten(), np.rad2deg(phi).flatten()))
	B  = np.vstack(dp.B(lat, 1))
	b  = B / np.linalg.norm(B, axis = 0)
	b  = np.vstack((b[1], -b[0], np.zeros(b.shape[1]))) # r, theta phi
	E = np.vstack((-(b[1] * Eh[0])/b[0], Eh[0], Eh[1])) # r, theta, phi

	SP = np.hstack((Pedersen_n.flatten(), Pedersen_s.flatten())).reshape((1, -1)) # shape (1, N) 
	SH = np.hstack((Hall_n.flatten()    , Hall_s.flatten())    ).reshape((1, -1)) # shape (1, N)
	bxE = np.cross(b, E, axisa = 0, axisb = 0, axisc = 0)
	j = SP * E[1:] + SH * bxE[1:] # horizontal components, A/m


	# make some low latitude zero current points
	minlat = 90 - np.rad2deg(theta).max()
	ll_lat = np.linspace(-minlat, minlat, 60)
	ll_lon = np.linspace(0, 360, 120)
	ll_lat, ll_lon = np.meshgrid(ll_lat, ll_lon)
	lat = np.hstack((lat, ll_lat.flatten()))
	lon = np.hstack((lon, ll_lon.flatten()))	
	j   = np.hstack((j, np.zeros((2, ll_lon.size))))


	# spherical harmonic analysis
	if count == 0: # only do this once
		N, M = 110, 110 # 12320 coefficients per current component
		shbasis  = SHBasis(N, M)
		datagrid = Grid(lat = lat, lon = lon)
		datagrid_evaluator = BasisEvaluator(shbasis, datagrid, reg_lambda = 0)# 1e-5)#1e0)# 10**1)
		#gtg = datagrid_evaluator.least_squares_helmholtz.ATWA
		#j_coeffs = datagrid_evaluator.grid_to_basis(j, helmholtz = True)
		
		grad   = datagrid_evaluator.G_grad
		rxgrad = datagrid_evaluator.G_rxgrad
		Gs = np.dstack((-grad, rxgrad))
		G  = np.vstack((Gs))

		if args.magnetic_energy_weight > 0:
			energy = magnetic_energy_weights(shbasis, args.energy_radius_km * 1e3, 6500e3)
			normal_matrix = G.T @ G
			data_diagonal = np.median(np.diag(normal_matrix))
			normal_matrix.flat[::normal_matrix.shape[0] + 1] += (
				args.magnetic_energy_weight * data_diagonal * energy
			)
			normal_factor = cho_factor(normal_matrix, lower = True, check_finite = False)
			print('magnetic-energy regularization:', args.magnetic_energy_weight,
			      'at', args.energy_radius_km, 'km')

	d = np.hstack(j)

	if args.magnetic_energy_weight > 0:
		j_coeffs = cho_solve(normal_factor, G.T @ d, check_finite = False)
	else:
		# Do not retain singular modes below the numerical precision of the design
		# matrix.  With rcond=0, those nearly-null modes reproduce the sheet current
		# but give enormous magnetic fields when continued away from the sheet.
		# NumPy's dimension-aware default is the appropriate cutoff here.
		j_coeffs = np.linalg.lstsq(G, d, rcond = None)[0]

	j_m = G.dot(j_coeffs)

	j_coeff_cf, j_coeff_df = np.split(j_coeffs, 2)

	# save coefficient for each time step
	np.save(args.output_dir / f'cfcoeff_{step}.npy', j_coeff_cf)
	np.save(args.output_dir / f'dfcoeff_{step}.npy', j_coeff_df)

	# make some plots to test if it worked
	fig, ax = plt.subplots(figsize = (8, 8))
	ax.hist2d(j_m.flatten(), j.flatten(), bins = (100, 100), norm = 'log', range = ((-.3, .3), (-.3, .3)))
	ax.set_aspect('equal')
	ax.set_xlabel(f'j components from inversion')
	ax.set_ylabel(f'j components from REMIX')
	ax.plot([-.3, .3], [-.3, .3], 'k-')
	plt.savefig(args.output_dir / ('magneticfield'+step.replace('#', '') + '.png'), dpi = 250)
	plt.close(fig)
	print('done step '+step)
