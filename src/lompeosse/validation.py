import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import secsy as cs

from lompe.model.visualization import *

def validate(osse_Emodel, gamera_data, ntime, primary="potential", overlay='fac', suptitle=None, savekw=None):
    """
    Assess Lompe's performance by comparing reconstructed quantities
    against the corresponding Gamera simulation quantities.

    TODO currently only implemented for potential and fac. Good enough? 

    Parameters
    ----------
    osse_Emodel : LompeOSSE
        LompeOSSE model used for the reconstruction.

    ntime : int
        Time of event.

    gamera_data : GameraData
        Gamera simulation data object.

    primary : str, optional
        Quantity to plot as contour lines. Default is "potential".

    overlay : str, optional
        Quantity to plot as filled contours on top of the primary quantity (e.g., "fac" on top of "potential"). 
        Set to None to plot only the primary quantity.

    savekw: dict, optional
        Keyword arguments passed to matplotlib.pyplot.savefig. 
        If None the figure is displayed with matplotlib.pyplot.show(). 
        For example, {'fname': 'validation.png', 'dpi': 300}.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Validation figure.
    metrics : dict
        Validation metrics

    """

    # ------------------------#
    # Retrieve Gamera and LompeOSSE quantities
    # ------------------------#

    gam_primary, lo_primary = get_quantity(primary, osse_Emodel, gamera_data, ntime)

    if overlay is not None:
        gam_overlay, lo_overlay = get_quantity(overlay, osse_Emodel, gamera_data, ntime)

    # ------------------------#
    # Compute validation metrics
    # ------------------------#

    metrics = {} 

    metrics[primary] = calculate_scalar_metrics(gam_primary, lo_primary)

    if overlay is not None:
        metrics[overlay] = calculate_scalar_metrics(gam_overlay, lo_overlay)

    # ------------------------#
    # Plot
    # ------------------------#

    # TODO currently plottint in xi eta space. do we want same plots as lomoestyle plot?
    # TODO add colorbars/scales on the side? or maybe not necessary since its in the lompestyle plot

    grid = osse_Emodel.grid_J

    # figheight = 9 

    # ar = osse_Emodel.grid_E.shape[1] / osse_Emodel.grid_E.shape[0] # aspect ratio

    # figwidth=(3 * ar + 1)/2 * figheight * .8
    # figsize = (figwidth, figheight)

    # area_scale = np.sqrt((figwidth * figheight) / (12 * 9))
    # font_scale = np.clip(area_scale, 0.8, 1.35)

    # fig = plt.figure(figsize = figsize)
    # fig.suptitle(suptitle, fontsize=22*font_scale, color="black", y=0.99) 

    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(suptitle, fontsize=16)

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    primary_settings = get_plot_settings(primary)
    scale_primary = primary_settings["scale"]
    plotting_unit_primary = primary_settings["unit"]

    if overlay is not None:
        overlay_settings = get_plot_settings(overlay)
        scale_overlay = overlay_settings["scale"]
        levels_overlay = overlay_settings["levels"]
        plotting_unit_overlay = overlay_settings["unit"]

    # Top-left: Gamera quantities
    ax1 = fig.add_subplot(gs[0, 0])  
    csax1 = cs.CSplot(ax1, grid, gridtype='cs')
    csax1.contour(grid.lon, grid.lat, gam_primary*scale_primary, colors='k')
    if overlay is not None:
        csax1.contourf(grid.lon, grid.lat, gam_overlay*scale_overlay, cmap='bwr', levels=levels_overlay)
        ax1.set_title(f"Gamera {primary} (black)\n and {overlay} (color)")
    else:
        ax1.set_title(f"Gamera {primary}")

    # Top-right: LompeOSSE-reconstructed quantities
    ax2 = fig.add_subplot(gs[0, 1])  
    csax2 = cs.CSplot(ax2, grid, gridtype='cs')
    csax2.contour(grid.lon, grid.lat, lo_primary*scale_primary, colors='k')
    if overlay is not None:
        csax2.contourf(grid.lon, grid.lat, lo_overlay*scale_overlay, cmap='bwr', levels=levels_overlay)
        ax2.set_title(f"LompeOSSE-reconstructed {primary} (black)\n and {overlay} (color)")
    else:
        ax2.set_title(f"LompeOSSE-reconstructed {primary}")

    ax1.set_axis_off()
    ax2.set_axis_off()

    # Bottom: validation scatter plot (primary quantity only)
    ax3 = fig.add_subplot(gs[1, :])
    plot_validation_scatter(ax3, gam_primary, lo_primary, primary, metrics[primary], scale_primary, plotting_unit_primary)

    # if overlay is None:
    #     ax3 = fig.add_subplot(gs[1, :])
    #     plot_validation_scatter(ax3, gam_primary, lo_primary, primary, metrics[primary], scale_primary, plotting_unit_primary)

    # else:
    #     ax3 = fig.add_subplot(gs[1, 0])
    #     plot_validation_scatter(ax3, gam_primary, lo_primary, primary, metrics[primary], scale_primary, plotting_unit_primary)
    #     ax4 = fig.add_subplot(gs[1, 1])
    #     plot_validation_scatter(ax4, gam_overlay, lo_overlay, overlay, metrics[overlay], scale_overlay, plotting_unit_overlay)

    plt.tight_layout()

    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()

    return fig, metrics

def get_quantity(quantity, osse_Emodel, gamera_data, time):
    """
    Retrieve electrodynamics quantities from Gamera simulation 
    and the corresponding LompeOSSE-reconstruted quantities
    """

    grid = osse_Emodel.grid_J

    if quantity == "potential":

        gamera_qty = gamera_data.get_potential(grid.lon, grid.lat, time) # in [V]

        lompeosse_qty = osse_Emodel.E_pot(lon=grid.lon, lat=grid.lat) # in [V]
        lompeosse_qty = lompeosse_qty.reshape(grid.lon.shape)

    elif quantity == "fac":

        gamera_qty = gamera_data.get_FAC(grid.lon, grid.lat, time) # in [A/m²]
 
        lompeosse_qty = osse_Emodel.FAC(lon=grid.lon, lat=grid.lat) # in [A/m²]
        lompeosse_qty = lompeosse_qty.reshape(grid.lon.shape)
    
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    return gamera_qty, lompeosse_qty

def calculate_scalar_metrics(truth, reconstruction):
    """
    Calculate validation metrics for two scalar fields.

    Parameters
    ----------
    truth : array-like
        Reference values from the Gamera simulation.
    reconstruction : array-like
        Corresponding values reconstructed by LompeOSSE.

    Returns
    -------
    dict
        Validation metrics. 
        RMSE, MAE, and bias are given in the
        same (SI) units as the input quantities. Pearson correlation
        and NRMSE are dimensionless.
    """

    truth = np.asarray(truth).flatten()
    reconstruction = np.asarray(reconstruction).flatten()

    # Only keep grid points where both values are finite
    valid = np.isfinite(truth) & np.isfinite(reconstruction)
    truth = truth[valid]
    reconstruction = reconstruction[valid]

    # Difference between LompeOSSE and Gamera at each grid point
    error = reconstruction - truth

    # Root Mean Square Error (RMSE) = How wrong is the reconstruction?
    # Measures the overall error scale (expressed in the same units as the targeted quantity)
    # Large errors have a stronger influence because the errors are squared
    rmse = np.sqrt(np.mean(error**2))

    # Bias/average signed error = Does the reconstruction tend to be wrong in one direction?
    # Measures the tendency for Lompe to be either too high or too low (systematic offset)
    # Positive/negative -> LompeOSSE is generally higher/lower than Gamera
    bias = np.mean(error)

    # Mean Absolute Error (MAE):
    # Measures the average absolute difference between reconstruction and truth
    mae = np.mean(np.abs(error))

    # Pearson correlation coefficient:
    # Measures how well the spatial variations in the reconstruction follow the spatial variations in the Gamera truth.
    # +-1 = perfect linear correlation; 0 = no linear correlation
    correlation = np.corrcoef(truth, reconstruction)[0, 1]

    # Normalized RMSE:
    # RMSE divided by the standard deviation of the Gamera truth.
    # This expresses the reconstruction error relative to the natural spatial variability of the reference field.
    nrmse = rmse / np.std(truth)

    return {"correlation": correlation,
            "rmse": rmse,
            "nrmse": nrmse,
            "mae": mae,
            "bias": bias}

def get_plot_settings(quantity):
    """Return plotting settings for a given quantity."""

    if quantity == "potential":
        return {"levels": None,
                "scale": 1e-3,
                "unit": "kV"}
    
    if quantity == "fac":
        return {"levels": np.linspace(-1.95, 1.95, 40) * 2,
                "scale": 1e6,
                "unit": "µA/m²"} # display range: ±3.9 µA/m²

    return {"levels": None, "scale": 1, "unit": ""}

def plot_validation_scatter(ax, gamera_qty, lompeosse_qty, quantity, metrics, scale, unit):

    ax.scatter(gamera_qty.flatten()*scale, lompeosse_qty.flatten()*scale, alpha=.3, color='grey')
    ax.set_xlabel(f"Gamera {quantity} [{unit}]")
    ax.set_ylabel(f"LompeOSSE {quantity} [{unit}]")
    ax.set_title(f"Validation metrics")

    # # 1:1 reference line
    # min_val = min(gamera_qty.min(), lompeosse_qty.min())
    # max_val = max(gamera_qty.max(), lompeosse_qty.max())
    # ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1')
    # ax.legend()

    # Validation metrics
    text = (f"r = {metrics['correlation']:.3f}\n"
            f"NRMSE = {metrics['nrmse']:.3f}\n"
            f"RMSE = {metrics['rmse'] * scale:.3g} {unit}\n"
            f"Bias = {metrics['bias'] * scale:.3g} {unit}")

    ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.65))
