"""Sample Gamera ground magnetic field, invert with Lompe, and compare E.

"""

import datetime as dt

import lompe
import matplotlib.pyplot as plt
import numpy as np

from lompeosse import GameraData


l1, l2 = 1, 1
timestep = 0
resolution = 300

# The Gamera pattern is organized in MLT. Changing the UT hour rotates this
# fixed geographic grid through that pattern. Changing only the date by a few
# days produces a much smaller MLT shift. Change ``timestep`` above to select a
# different Gamera simulation snapshot.
time = dt.datetime(2012, 4, 8, 6, 19)
earth_radius = 6371.2e3
current_radius = 6500e3

gamera = GameraData(time, timestep=timestep, hemisphere='NORTH')

resolution = resolution * 1e3
projection = lompe.cs.CSprojection((90, 83), 0)
grid = lompe.cs.CSgrid(projection, 3000e3, 3000e3, resolution, resolution, R=current_radius, )

print(f'Cubed-sphere model grid: {grid.shape[0]} x {grid.shape[1]} cells')
print(f'Magnetic observation mesh: {grid.lon_mesh.shape[0]} x {grid.lon_mesh.shape[1]} points')

grid_qdlat, grid_mlon = gamera.apex.geo2qd(grid.lat.flatten(), grid.lon.flatten(), 0)
grid_mlt = gamera.dp.mlon2mlt(grid_mlon, time)
center_qdlat, center_mlon = gamera.apex.geo2qd(83, 90, 0)
center_mlt = gamera.dp.mlon2mlt(center_mlon, time)

print(f'Time used for the geographic-to-MLT rotation: {time:%Y-%m-%d %H:%M} UT')
print(f'Grid-center QD latitude / MLT: {center_qdlat:.2f}° / {center_mlt:.2f} h')
print(f'Grid-cell MLT range: {grid_mlt.min():.2f} to {grid_mlt.max():.2f} h')


# Sample Gamera B at every cubed-sphere mesh vertex.
observation_lon = grid.lon_mesh.flatten()
observation_lat = grid.lat_mesh.flatten()
observation_radius = np.full(observation_lon.size, earth_radius)

B_gamera_observations = np.array(gamera.get_B(observation_lon, observation_lat, observation_radius, time=time ))

magnetic_data = lompe.Data(B_gamera_observations, np.vstack((observation_lon, observation_lat, observation_radius)), datatype='ground_mag', error= 10 * 1e-9, iweight=1 )


# Lompe repeatedly evaluates its conductance functions on grid_J. Cache the
# Gamera interpolation once so those repeated calls are inexpensive.
hall_on_grid = gamera.get_Hall(grid.lon, grid.lat, time)
pedersen_on_grid = gamera.get_Pedersen(grid.lon, grid.lat, time)


def hall_conductance(lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.shape == grid.lon.shape and np.array_equal(lon, grid.lon) and np.array_equal(lat, grid.lat):
        return hall_on_grid

    raise ValueError('This focused example only evaluates conductance on grid_J')


def pedersen_conductance(lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.shape == grid.lon.shape and np.array_equal(lon, grid.lon) and np.array_equal(lat, grid.lat):
        return pedersen_on_grid

    raise ValueError('This focused example only evaluates conductance on grid_J')


model = lompe.Emodel(grid, (hall_conductance, pedersen_conductance), epoch=time.year )
model.add_data(magnetic_data)
model.run_inversion(l1=l1, l2=l2, data_density_weight=False)


# Compare E at the cell centres. GameraData.get_E and Lompe both return
# geographic eastward and northward components in V/m.
comparison_lon = grid.lon.flatten()
comparison_lat = grid.lat.flatten()

E_gamera = np.array(gamera.get_E(comparison_lon, comparison_lat, time))
E_lompe = np.array(model.E(comparison_lon, comparison_lat))


def component_metrics(truth, estimate):
    valid = np.isfinite(truth) & np.isfinite(estimate)
    truth = truth[valid]
    estimate = estimate[valid]
    difference = estimate - truth

    return {
        'truth_rms': np.sqrt(np.mean(truth**2)),
        'estimate_rms': np.sqrt(np.mean(estimate**2)),
        'rmse': np.sqrt(np.mean(difference**2)),
        'correlation': np.corrcoef(truth, estimate)[0, 1],
    }


print('\nElectric-field comparison at grid-cell centres')
for index, component in enumerate(('east', 'north')):
    metrics = component_metrics(E_gamera[index], E_lompe[index])
    print(
        f'  {component:5s}: '
        f'Gamera RMS={metrics["truth_rms"] * 1e3:6.2f} mV/m, '
        f'Lompe RMS={metrics["estimate_rms"] * 1e3:6.2f} mV/m, '
        f'RMSE={metrics["rmse"] * 1e3:6.2f} mV/m, '
        f'correlation={metrics["correlation"]:6.3f}'
    )


# Confirm how closely Lompe fits the magnetic observations used by the inversion.
B_lompe_observations = np.array(model.B_ground(
    observation_lon,
    observation_lat,
    observation_radius,
))
B_residual = (B_lompe_observations - B_gamera_observations) * 1e9

print('\nGround magnetic observation fit')
print(f'  combined RMS residual: {np.sqrt(np.mean(B_residual**2)):6.2f} nT')
print(f'  maximum absolute residual: {np.max(np.abs(B_residual)):6.2f} nT')


shape = grid.shape
plot_fields = (
    E_gamera[0].reshape(shape) * 1e3,
    E_lompe[0].reshape(shape) * 1e3,
    (E_lompe[0] - E_gamera[0]).reshape(shape) * 1e3,
    E_gamera[1].reshape(shape) * 1e3,
    E_lompe[1].reshape(shape) * 1e3,
    (E_lompe[1] - E_gamera[1]).reshape(shape) * 1e3,
)
titles = ('Gamera $E_e$', 'Lompe $E_e$', 'Lompe - Gamera $E_e$', 'Gamera $E_n$', 'Lompe $E_n$', 'Lompe - Gamera $E_n$', )

xi = grid.xi.reshape(shape)
eta = grid.eta.reshape(shape)
fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

for row in range(2):
    component_limit = max(
        np.nanmax(np.abs(plot_fields[row * 3])),
        np.nanmax(np.abs(plot_fields[row * 3 + 1])),
    )
    difference_limit = np.nanmax(np.abs(plot_fields[row * 3 + 2]))

    for column in range(3):
        index = row * 3 + column
        limit = component_limit if column < 2 else difference_limit
        limit = max(limit, 1e-12)
        levels = np.linspace(-limit, limit, 31)
        image = axes[row, column].contourf(
            xi,
            eta,
            plot_fields[index],
            levels=levels,
            cmap='bwr',
            extend='both',
        )
        axes[row, column].set_aspect('equal')
        axes[row, column].set_title(titles[index])
        axes[row, column].set_xlabel(r'$\xi$')
        axes[row, column].set_ylabel(r'$\eta$')
        fig.colorbar(image, ax=axes[row, column], label='mV/m')

fig.suptitle(
    f'Gamera B to Lompe E: Step {timestep}, {time:%Y-%m-%d %H:%M} UT, '
    f'{resolution / 1e3:g} km grid, l1={l1:g}, l2={l2:g}'
)
fig.savefig('get_B_lompe_E_comparison.png', dpi=160)
print(f'\nSaved figure to get_B_lompe_E_comparison.png')

plt.show()
