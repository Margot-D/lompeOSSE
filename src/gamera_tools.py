"""
The functions in this module are largely inspired by the kaipy.remix module.
They provide similar functionality for loading, processing, and visualizing
Gamera snapshots, without requiring Kaipy as a dependency.

This allows users to explore Gamera data in a similar way to the original
Kaipy/Remix workflow, while avoiding the additional dependency on Kaipy and
its Fortran compiler requirements.
"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.cm as cm

RI = 6.5 # in [1000km], radius of the ionosphere used in Gamera 
facMax = 1.5
facCM = cm.RdBu_r
flxCM = cm.inferno

def efield(data, ri=RI * 1e3):
    """Calculate the electric field from the Gamera potential."""

    Psi = data["potential"]
    x = data["X"]
    y = data["Y"]

    # Spherical coordinates
    theta = np.arcsin(np.sqrt(x**2 + y**2))
    phi = np.arctan2(y, x)

    # Interpolate potential to corners
    Psi_c = np.zeros(x.shape)
    Psi_c[1:-1, 1:-1] = 0.25 * (Psi[1:, 1:] + Psi[:-1, 1:] + Psi[1:, :-1] + Psi[:-1, :-1])

    # Periodic boundary
    Psi_c[1:-1, 0] = 0.25 * (Psi[1:, 0] + Psi[:-1, 0] + Psi[1:, -1] + Psi[:-1, -1])
    Psi_c[1:-1, -1] = Psi_c[1:-1, 0]

    # Pole
    Psi_pole = Psi[0, :].mean()
    Psi_c[0, 1:-1] = 0.25 * (2 * Psi_pole + Psi[0, :-1] + Psi[0, 1:])
    Psi_c[0, 0] = 0.25 * (2 * Psi_pole + Psi[0, -1] + Psi[0, 0])
    Psi_c[0, -1] = 0.25 * (2 * Psi_pole + Psi[0, -1] + Psi[0, 0])

    # Low-latitude boundary
    Psi_c[-1, :] = 2 * Psi_c[-2, :] - Psi_c[-3, :]

    # E_theta
    tmp = 0.5 * (Psi_c[:, 1:] + Psi_c[:, :-1])
    dPsi = tmp[1:, :] - tmp[:-1, :]

    tmp = 0.5 * (theta[:, 1:] + theta[:, :-1])
    dtheta = tmp[1:, :] - tmp[:-1, :]

    etheta = dPsi / dtheta / ri

    # E_phi
    tmp = 0.5 * (Psi_c[1:, :] + Psi_c[:-1, :])
    dPsi = tmp[:, 1:] - tmp[:, :-1]

    tmp = 0.5 * (phi[1:, :] + phi[:-1, :])
    dphi = tmp[:, 1:] - tmp[:, :-1]

    theta_center = 0.25 * (theta[:-1, :-1] + theta[1:, :-1] + theta[:-1, 1:] + theta[1:, 1:])

    ephi = dPsi / dphi / np.sin(theta_center) / ri

    # E = -grad(Psi)
    return -etheta, -ephi


def joule(data, ri=RI * 1e3):
    """Calculate Joule heating from the Gamera potential and Pedersen conductance."""

    etheta, ephi = efield(data, ri)
    joule = data["sigmap"] * (etheta**2 + ephi**2) 

    return joule


def load_gamera_snapshot(filename, timestep, hemisphere="NORTH"):
    """Load one Gamera snapshot and return the relevant variables."""

    hemisphere = hemisphere.upper()

    with h5py.File(filename, "r") as f:
        snapshot = f[f"Step#{timestep}"]

        data = {"X": f["X"][:],
                "Y": f["Y"][:],
                "potential": snapshot[f"Potential {hemisphere}"][:],
                "current": snapshot[f"Field-aligned current {hemisphere}"][:],
                "sigmap": snapshot[f"Pedersen conductance {hemisphere}"][:],
                "sigmah": snapshot[f"Hall conductance {hemisphere}"][:],
                "energy": snapshot[f"Average energy {hemisphere}"][:],
                "flux": snapshot[f"Number flux {hemisphere}"][:]}

        data["joule"] = joule(data)
        data["eflux"] = data["energy"] * data["flux"] * 1.6e-9
        
    if hemisphere == "NORTH":
        data["current"] *= -1 # upward current is positive
    else:
        # Reproduce Remix's SOUTH orientation
        for key in data:
            if key not in ("X", "Y"):
                data[key] = data[key][:, ::-1]

    return data


def plot_gamera(data, parameter):
    """Plot a Gamera variable on a polar grid."""

    # Grid coordinates
    r = np.sqrt(data["X"]**2 + data["Y"]**2)
    theta = np.arctan2(data["Y"], data["X"])

    # Variable and plotting settings
    if parameter == "potential":
        variable = data["potential"]
        label = "Potential [kV]"
        vmin, vmax = -100, 100
        cmap = facCM

    elif parameter == "current":
        variable = data["current"]
        label = r"Current density [$\mu$A/m$^2$]"
        vmin, vmax = -facMax, facMax 
        cmap = facCM

    elif parameter == "joule":
        variable = data["joule"] * 1e3  # W/m² -> mW/m²
        label = r"Joule heating [mW/m$^2$]"
        vmin, vmax = 0, 10
        cmap = flxCM

    elif parameter == "sigmap":
        variable = data["sigmap"]
        label = r"Pedersen conductance [S]"
        vmin, vmax = 1, 20
        cmap = flxCM

    elif parameter == "sigmah":
        variable = data["sigmah"]
        label = r"Hall conductance [S]"
        vmin, vmax = 2, 40
        cmap = flxCM

    elif parameter == "energy":
        variable = data["energy"]
        label = "Average energy [keV]"
        vmin, vmax = 0, 20
        cmap = flxCM

    elif parameter == "flux":
        variable = data["flux"]
        label = r"Number flux [cm$^{-2}$ s$^{-1}$]"
        vmin, vmax = 0, 1e9
        cmap = flxCM

    elif parameter == "eflux":
        variable = data["eflux"]
        label = r"Energy flux [erg cm$^{-2}$ s$^{-1}$]"
        vmin, vmax = 0, 10
        cmap = flxCM

    else:
        raise ValueError(f"Unknown parameter: {parameter}")

    fig, ax = plt.subplots(subplot_kw={"polar": True})

    # Plot
    p = ax.pcolormesh(theta + np.pi / 2, r, variable,
                      cmap=cmap, vmin=vmin, vmax=vmax)

    # MLT labels
    ax.set_thetagrids([0, 90, 180, 270], ["06", "12", "18", "00"])

    # Latitude circles
    circle_list = [10, 20, 30, 40]
    circles = np.sin(np.deg2rad(circle_list))

    ax.set_rgrids(circles, [f"{lat}°" for lat in circle_list], fontsize=8)

    ax.grid(True)
    ax.set_ylim(0, r.max())

    # Colorbar
    cb = fig.colorbar(p, ax=ax, pad=0.1, shrink=0.85)
    cb.set_label(label)

    # Potential contours on current plot
    if parameter == "current":
        potential = data["potential"]

        ax.contour(theta[:-1, :-1] + np.pi / 2, r[:-1, :-1], potential, levels=np.linspace(-100, 100, 16), colors="black", linewidths=0.5)

    ax.set_title(parameter.capitalize())

    return ax
