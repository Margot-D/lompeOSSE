import numpy as np
import matplotlib.pyplot as plt
import lompe
from lompe.model.visualization import *
import apexpy
from polplot import Polarplot
from dipole import Dipole


###############################
# DEFINE SOME GLOBAL PARAMETERS

# Default arrow scales (all SI units):
QUIVERSCALES = {'ground_mag':       600 * 1e-9 , # ground magnetic field scale [T]
                'space_mag_fac':    600 * 1e-9 , # FAC magnetic field scale [T]
                'convection':       2000       , # convection velocity scale [m/s]
                'efield':           100  * 1e-3, # electric field scale [V/m]
                'electric_current': 1000 * 1e-3, # electric surface current density [A/m] Ohm's law 
                'secs_current':     1000 * 1e-3, # electric surface current density [A/m] SECS 
                'space_mag_full':   600 * 1e-9 } # FAC magnetic field scale [T]

# Default color scales (SI units):
COLORSCALES =  {'fac':        np.linspace(-1.95, 1.95, 40) * 1e-6 * 2,
                'ground_mag': np.linspace(-980, 980, 50) * 1e-9 / 3, # upward component
                # 'ground_mag': np.linspace(-500, 500, 50)* 1e-9 / 3, # upward component
                'hall':       np.linspace(0, 20, 32), # mho
                'pedersen':   np.linspace(0, 20, 32)} # mho

# Default color map:
CMAP = plt.cm.magma

RE = 6371.2e3 # Earth radius in meters

def plot_gamera_lompe_style(osse_Emodel, gamera_data, ntime, figheight=9, suptitle=None, quiverscales=None, colorscales=None, savekw=None, clkw = {}, return_axes=False):
    """
    Plot Gamera simulation quantities using a Lompe-style visualization.

    The plot uses visualization functions from Lompe to display
    electrodynamics quantities extracted from a Gamera simulation on
    the same grid as the LompeOSSE model.

    Parameters
    ----------
    osse_Emodel : lompe.Emodel
        Lompe model defining the grid used for the visualization.
    gamera_data : GameraData
        Gamera simulation data from which the quantities are extracted.
    ntime : int
        Index of the Gamera simulation time step to plot.
    
    figheight: float, optional
        figure height - the width is determined automatically based on aspect ratios,
        and (width, height) is the figsize given to matplotlib.pyplot.figure
    suptitle: TODO
    quiverscales: dict, optional
        dictionary of scales (in inches) to use for quiver plots. keys must be valid datatype. 
        default values are used for datatypes that are not in list of keys
    colorscales: dict, optional
        dictionary of colorscales to use in contour plots. keys must be valid datatype. 
        default values are used for datatypes that are not in list of keys
    savekw: dictionary, optional
        keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
    clkw: dictionary, optional
        keywords for Polarplot.coastlines(), used to show coastlines in polarplot. Ignored 
        if apex or time are not specified 
    return_axes: bool, optional
        Set to True to return the matplotlib figure and axes objects.
        Default is False and will only return the matplotlib figure object

    Returns
    -------
    matplotlib.figure.Figure
    """

    apx = apexpy.Apex(ntime.year) # apex object for magnetic coordinate calculations

    # ------------------------#
    # Build plotting grid 
    # ------------------------#

    grid = osse_Emodel.grid_J
    lon, lat = grid.lon, grid.lat
    xi, eta = grid.xi, grid.eta

    # grid for plotting vectors: (taken from lompe.visualization.plot_quiver)
    sh = np.array(grid.shape) #TODO should it be grid_E?
    NN = 12 # Number of arrows to plot along smallest dimension
    sh = sh // sh.min() * NN 
    ximin  = grid.xi .min() + grid.dxi  / 3 #TODO check that.. from quiver function
    ximax  = grid.xi .max() - grid.dxi  / 3
    etamin = grid.eta.min() + grid.deta / 3
    etamax = grid.eta.max() - grid.deta / 3
    qxi, qeta = np.meshgrid(np.linspace(ximin, ximax, sh[1]), np.linspace(etamin, etamax, sh[1]))
    qlo, qla = grid.projection.cube2geo(qxi, qeta)

    # ------------------------#
    # Derive the Gamera quantities 
    # scalar are derived on osse_Emodel.grid_J, vectors in the quiver_grid
    # TODO this is following the lompe code but check if this is how it should be done?
    # ------------------------#

    # Electric potential in [V]
    Epot = gamera_data.get_potential(lon, lat, ntime) 
    V = Epot - Epot.min() - (Epot.max() - Epot.min())/2 
    V = V.reshape(grid.shape) * 1e-3 # Convert from [V] to [kV] to match lompeplot

    # Convection velocity
    Ve, Vn = gamera_data.get_V(qlo, qla, ntime)
    x, y, Vx, Vy = grid.projection.vector_cube_projection(Ve, Vn, qlo, qla)

    # Field-aligned current (positive upward, matching Lompe)
    facG = gamera_data.get_FAC(lon, lat, ntime)

    # Space magnetic field
    r_space = RE + osse_Emodel.refh*1e3 # in [m]
    Be_space, Bn_space, Bu_space = gamera_data.get_B(qlo, qla, r=np.array(r_space), no_df_current=True, time=ntime)
    x, y, Bx_space, By_space = grid.projection.vector_cube_projection(Be_space, Bn_space, qlo, qla)

    # Ground magnetic field 
    r_ground = RE # in [m]

    ## on quiver grid
    Be_ground, Bn_ground, _ = gamera_data.get_B(qlo, qla, r=np.full_like(qlo, r_ground), time=ntime)
    x, y, Bx_ground, By_ground = grid.projection.vector_cube_projection(Be_ground, Bn_ground, qlo, qla)

    ## on OSSE model grid
    _, _, Bu_ground = gamera_data.get_B(lon, lat, r=np.full_like(lon, r_ground), time=ntime)
    Bu_ground = Bu_ground.reshape(grid.shape) #TODO put reshape(grid.shape) in get_B directly maybe?

    # Conductances
    HallG = gamera_data.get_Hall(lon, lat, ntime)
    PedersenG = gamera_data.get_Pedersen(lon, lat, ntime)

    # Electric currents #TODO should it be on grid_E maybe?
    Ee, En = gamera_data.get_E(qlo, qla, ntime)
    x, y, Ex, Ey = grid.projection.vector_cube_projection(Ee, En, qlo, qla)

    # ------------------------#
    # Set up figure (lompe.visualization.lompeplot)
    # ------------------------#

    if quiverscales == None:
        quiverscales = QUIVERSCALES
    else:
        QUIVERSCALES.update(quiverscales)
        quiverscales = QUIVERSCALES

    if colorscales == None:
        colorscales = COLORSCALES
    else:
        COLORSCALES.update(colorscales)
        colorscales = COLORSCALES

    # potential_levels
    dV = 5 # contour level step size in kV
    potential_levels = np.r_[(V.min()//dV)*dV :(V.max()//dV)*dV + dV:dV]

    # Set up figures
    # --------------
    ar = osse_Emodel.grid_E.shape[1] / osse_Emodel.grid_E.shape[0] # aspect ratio

    figwidth=(3 * ar + 1)/2 * figheight * .8
    figsize = (figwidth, figheight)

    area_scale = np.sqrt((figwidth * figheight) / (12 * 9))
    font_scale = np.clip(area_scale, 0.8, 1.35)

    fig_gamera = plt.figure(figsize = figsize)
    suptitle=f'Gamera ("truth") electrodynamics'
    fig_gamera.suptitle(suptitle, fontsize=22*font_scale, color="black", y=0.99) 

    row_gap = int(np.clip(0.5 + 2*(1/ar - 1), 0, 3))
    second_row = 10 + row_gap

    axes = np.vstack(([plt.subplot2grid((20, 4), (0, j), rowspan=10) for j in range(3)],
                      [plt.subplot2grid((20, 4), (second_row, j), rowspan=10) for j in range(3)]))
    for ax in axes.flatten():
        lompe.visualization.format_ax(ax, osse_Emodel, apex = apx)

    # Velocity
    # --------
    # Convection velocity and electric potential 
    ax1 = axes[0,0] 
    ax1.quiver(x, y, Vx, Vy) # convection
    ax1.contour(xi, eta, V, colors='C0', linewidths=2, levels=potential_levels) # potential
    ax1.set_title("Convection velocity \n and \n electric potential", fontsize=15*font_scale)

    # Space magnetic field
    # --------------------
    # FAC and space magnetic field
    ax2 = axes[0,1]
    ax2.quiver(x, y, Bx_space, By_space, zorder=3, scale=quiverscales['space_mag_fac'], scale_units="inches")
    ax2.contourf(xi, eta, facG, cmap='bwr', levels=colorscales['fac'], zorder=0, extend='both')
    ax2.set_title("Field-aligned currents \n and magnetic field", fontsize=15*font_scale)

    # Ground magnetic field
    # ---------------------
    ax3 = axes[0,2]  
    ax3.quiver(x, y, Bx_ground, By_ground, zorder=3, scale=quiverscales['ground_mag'], scale_units="inches")
    ax3.contourf(xi, eta, Bu_ground, cmap='bwr', levels=colorscales['ground_mag'], zorder=0, extend='both')
    ax3.set_title("Ground magnetic field", fontsize=15*font_scale)

    # Hall conductance
    # ----------------
    ax4 = axes[1,0]
    ax4.contourf(xi, eta, HallG, cmap=CMAP, levels=colorscales['hall'], zorder=0, extend='both')
    lompe.visualization.plot_coastlines(ax4, osse_Emodel, color = 'grey')
    lompe.visualization.plot_mlt(ax4, osse_Emodel, ntime, apx, color = 'grey')
    ax4.set_title("Hall conductance", fontsize=15*font_scale)

    # Pedersen conductance
    # --------------------
    ax5 = axes[1,1]
    ax5.contourf(xi, eta, PedersenG, cmap=CMAP, levels=colorscales['pedersen'], zorder=0, extend='both')
    lompe.visualization.plot_coastlines(ax5, osse_Emodel, color = 'grey')
    lompe.visualization.plot_mlt(ax5, osse_Emodel, ntime, apx, color = 'grey')
    ax5.set_title("Pedersen conductance", fontsize=15*font_scale)

    # Current densities
    # -----------------
    ax6 = axes[1,2]
    ax6.quiver(x, y, Ex, Ey, zorder=3, scale_units="inches") #TODO scale does not work here #scale=quiverscales['electric_current'],
    ax6.set_title("Electric currents", fontsize=15*font_scale)

    # Polarplot (lompe.visualization.polarplot)
    # ---------
    if ntime != None and apx != None:
        ax = plt.subplot2grid((20, 4), (0, 3), rowspan = 10) 

        pax = Polarplot(ax, minlat = 50)
        cd = Dipole(apx.year)

        # coastlines
        if 'resolution' not in clkw.keys():
            resolution = '110m'
            kwargs = clkw.copy()
        else:
            resolution = clkw.pop('resolution')
            kwargs = clkw.copy()
        
        if 'color' not in kwargs.keys():
            kwargs['color'] = 'lightgrey'
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 2
            
        pax.coastlines(time = ntime, mag = apx, north = True if osse_Emodel.hemisphere > 0 else False, resolution = resolution, **kwargs)

        xs = (grid.lon_mesh[0, :], grid.lon_mesh[-1, :], grid.lon_mesh[:, 0], grid.lon_mesh[:, -1])
        ys = (grid.lat_mesh[0, :], grid.lat_mesh[-1, :], grid.lat_mesh[:, 0], grid.lat_mesh[:, -1])
        for i, c in enumerate(zip(xs, ys)):
            lon, lat = c
            if not osse_Emodel.dipole:    
                lat, lon = apx.geo2apex(lat, lon, (osse_Emodel.R-RE)*1e-3)   # to magnetic apex
            mlt = cd.mlon2mlt(lon, ntime)
            pax.plot(lat, mlt, color = 'black', linewidth = 1.5 if i == 0 else .5, zorder = 2)

        if dV != None: # plot electric potential
            lat, lon = grid.lat, grid.lon
            if not osse_Emodel.dipole:
                lat, lon = apx.geo2apex(lat, lon, (osse_Emodel.R-RE)*1e-3)   # to magnetic apex
            mlt = cd.mlon2mlt(lon, ntime)

            pax.contour(lat, mlt, V, levels = potential_levels, colors = 'C0', linewidths = 1, zorder = 3)

    # Make scales
    #------------
    cbarax1 = plt.subplot2grid((20, 40), (16, 32), rowspan = 1, colspan = 7)
    cbarax2 = plt.subplot2grid((20, 40), (12, 32), rowspan = 1, colspan = 7)

    arrowax = plt.subplot2grid((20, 40), (19, 29), rowspan = 1, colspan = 7)

    xshift = np.clip(1.2*(ar - 1), 0, 3)
    xarrow = 7 + xshift
    xtext = np.clip(12 + 3*(1/ar - 1), 5, 15) + xshift

    arrowax.set_axis_off()
    arrowax.quiver(xarrow, .5, 1, 0, scale = 2, scale_units = 'inches')
    arrowax.set_ylim(0, 1)
    arrowax.set_xlim(0, 20)
    arrowax.text(xtext, 1, '{:.0f} nT (ground), \n {:.0f} nT (space), \n{:.0f} mA/m, {:.0f} m/s'.format(quiverscales['ground_mag'] * 1e9 // 2, quiverscales['space_mag_full'] * 1e9 // 2, quiverscales['electric_current'] * 1e3 // 2, quiverscales['convection'] // 2 ), ha = 'left', va = 'top', fontsize=12*font_scale)

    if ntime != None:
        cbarax2.set_title(str(ntime) + ' UT', fontweight = 'bold')

    fac_levels = colorscales['fac']
    xx = np.vstack((fac_levels, fac_levels)) * 1e6
    yy = np.vstack((np.zeros_like(fac_levels), np.ones_like(fac_levels)))
    cbarax1.contourf(xx, yy, xx, cmap = plt.cm.bwr, levels = fac_levels * 1e6)
    cbarax1.set_xlabel(r'$\mu$A/m$^2$')
    cbarax1.set_yticks([])
    buax = plt.twiny(cbarax1)
    buax.set_xlim(colorscales['ground_mag'].min() *1e9, colorscales['ground_mag'].max() * 1e9)
    buax.set_xlabel('nT')

    conductance_levels = colorscales['hall']
    xx = np.vstack((conductance_levels, conductance_levels)) 
    yy = np.vstack((np.zeros_like(conductance_levels), np.ones_like(conductance_levels)))
    cbarax2.contourf(xx, yy, xx, levels = conductance_levels, cmap = CMAP)
    cbarax2.set_xlabel('mho')
    cbarax2.set_yticks([])

    # Finish
    # ------
    top = np.clip(0.86 - 0.06*(1/ar - 1) - 0.018*np.exp(-((ar - 1.33)/0.18)**2), 0.75, 0.91)
    plt.subplots_adjust(top=top, bottom=0.065, left=0.01, right=0.98) 

    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
    if return_axes==True:
        return fig_gamera, axes, arrowax, [cbarax1, cbarax2]
    else:
        return fig_gamera
