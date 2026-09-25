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
                'space_mag_full':   600 * 1e-9 } # FAC magnetic field scale TODO correct?? [T]

# Default color scales (SI units):
COLORSCALES =  {'fac':        np.linspace(-1.95, 1.95, 40) * 1e-6 * 2,
                'ground_mag': np.linspace(-980, 980, 50) * 1e-9 / 3, # upward component
                'hall':       np.linspace(0, 20, 32), # mho
                'pedersen':   np.linspace(0, 20, 32)} # mho

# Default color map:
CMAP = plt.cm.magma

RE = 6371.2e3 # Earth radius in meters

def plot_gamera_lompe_style(osse_Emodel, gamera_data, ntime, figheight=9, suptitle=None, quiverscales=None, colorscales=None, savekw=None, clkw = {}):
    """
    Plot Gamera simulation quantities using a Lompe-style visualization.

    Gamera electrodynamic quantities are extracted at the specified event time 
    and plotted on the spatial grid defined by the LompeOSSE model. 
    The figure follows the visualization approach used by Lompe, including maps 
    of electric potential, convection velocity, field-aligned currents, magnetic fields,
    conductances, and electric currents, as well as a polar view of the model domain.

    Parameters
    ----------
    osse_Emodel : lompe.Emodel
        LompeOSSE model (synthetic Lompe model) defining the spatial grid.

    gamera_data : GameraData
        Gamera simulation data object providing methods for extracting electric potential, 
        convection velocity, field-aligned currents, magnetic fields, Hall conductance, 
        Pedersen conductance, and electric fields.

    ntime : datetime-like
        Event time used to identify the Gamera data to plot.

    figheight: float, optional
        Height of the resulting figure in inches. Default is 9.
        The figure width is determined from the aspect ratio of the model grid. 

    suptitle: str, optional 
        Title displayed at the top of the figure. Default is None.
        
    quiverscales: dict, optional
        Custom scales for the vector plots. 
        Values not provided use the defaults defined in ``QUIVERSCALES``.

    colorscales: dict, optional
        Custom contour levels for scalar quantities. 
        Values not provided use the defaults defined in ``COLORSCALES``.

    savekw: dict, optional
        Keyword arguments passed to matplotlib.pyplot.savefig. 
        If None the figure is displayed with matplotlib.pyplot.show(). 
        For example, {'fname': 'gamera.png', 'dpi': 300}.

    clkw: dict, optional
        Keyword arguments passed to Polarplot.coastlines().

    Returns
    -------
    matplotlib.figure.Figure
    """

    apx = apexpy.Apex(ntime.year) # apex object for magnetic coordinate calculations

    # ------------------------#
    # Build evaluating/plotting grid 
    # ------------------------#

    # Grid for scalar quantities
    grid = osse_Emodel.grid_J
    slo, sla = grid.lon, grid.lat
    sxi, seta = grid.xi, grid.eta

    # Reduced grid used for vector quantities (based on the grid used by lompe.visualization.plot_quiver)
    sh = np.array(grid.shape)
    NN = 12 # Number of arrows to plot along smallest dimension
    sh = sh // sh.min() * NN 
    ximin  = grid.xi .min() + grid.dxi  / 3 
    ximax  = grid.xi .max() - grid.dxi  / 3
    etamin = grid.eta.min() + grid.deta / 3
    etamax = grid.eta.max() - grid.deta / 3
    qxi, qeta = np.meshgrid(np.linspace(ximin, ximax, sh[1]), np.linspace(etamin, etamax, sh[1]))
    qlo, qla = grid.projection.cube2geo(qxi, qeta)

    # ------------------------#
    # Derive the Gamera quantities 
    # ------------------------#

    # Electric potential
    Epot = gamera_data.get_potential(slo, sla, ntime) # in [V]
    V = Epot - Epot.min() - (Epot.max() - Epot.min())/2 
    V = V * 1e-3 # Convert from [V] to [kV] to match plot(potential) in lompeplot.visualization 

    # Convection velocity
    Ve, Vn = gamera_data.get_V(qlo, qla, ntime) # in [m/s]
    x, y, Vx, Vy = grid.projection.vector_cube_projection(Ve, Vn, qlo, qla)

    # Field-aligned current (positive upward, matching Lompe)
    facG = gamera_data.get_FAC(slo, sla, ntime) # in [A/m²]

    # Space magnetic field
    r_space = RE + osse_Emodel.refh*1e3 # in [m]
    Be_space, Bn_space, Bu_space = gamera_data.get_B(qlo, qla, r=np.array(r_space), no_df_current=True, time=ntime) # in [T]
    x, y, Bx_space, By_space = grid.projection.vector_cube_projection(Be_space, Bn_space, qlo, qla)

    # Ground magnetic field 
    r_ground = RE # Earth's surface in [m]

    # Vector components evaluated on the quiver grid
    Be_ground, Bn_ground, _ = gamera_data.get_B(qlo, qla, r=np.full_like(qlo, r_ground), time=ntime) # in [T]
    x, y, Bx_ground, By_ground = grid.projection.vector_cube_projection(Be_ground, Bn_ground, qlo, qla)

    # Upward component evaluated on the model grid for the contour plot
    _, _, Bu_ground = gamera_data.get_B(slo, sla, r=np.full_like(slo, r_ground), time=ntime) # in [T]

    # Conductances
    HallG = gamera_data.get_Hall(slo, sla, ntime) # in [S] = [mho]
    PedersenG = gamera_data.get_Pedersen(slo, sla, ntime) # in [S] = [mho]

    # Electric currents (horizontal ionospheric surface current density)
    je, jn = gamera_data.get_hCurrents(qlo, qla, ntime) # in [A/m]
    x, y, jx, jy = grid.projection.vector_cube_projection(je, jn, qlo, qla)

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

    # potential levels
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
    fig_gamera.suptitle(suptitle, fontsize=22*font_scale, color="black", y=0.99) 

    row_gap = int(np.clip(0.5 + 2*(1/ar - 1), 0, 3))
    second_row = 10 + row_gap

    axes = np.vstack(([plt.subplot2grid((20, 4), (0, j), rowspan=10) for j in range(3)],
                      [plt.subplot2grid((20, 4), (second_row, j), rowspan=10) for j in range(3)]))
    for ax in axes.flatten():
        lompe.visualization.format_ax(ax, osse_Emodel, apex = apx)

    # Convection velocity and electric potential
    # -------- 
    ax1 = axes[0,0] 
    ax1.quiver(x, y, Vx, Vy) # convection
    ax1.contour(sxi, seta, V, colors='C0', linewidths=2, levels=potential_levels) # potential
    ax1.set_title("Convection velocity \n and \n electric potential", fontsize=15*font_scale)

    # FACs and space magnetic field
    # --------------------
    ax2 = axes[0,1]
    ax2.quiver(x, y, Bx_space, By_space, zorder=3, scale=quiverscales['space_mag_fac'], scale_units="inches")
    ax2.contourf(sxi, seta, facG, cmap='bwr', levels=colorscales['fac'], zorder=0, extend='both')
    ax2.set_title("Field-aligned currents \n and magnetic field", fontsize=15*font_scale)

    # Ground magnetic field
    # ---------------------
    ax3 = axes[0,2]  
    ax3.quiver(x, y, Bx_ground, By_ground, zorder=3, scale=quiverscales['ground_mag'], scale_units="inches")
    ax3.contourf(sxi, seta, Bu_ground, cmap='bwr', levels=colorscales['ground_mag'], zorder=0, extend='both')
    ax3.set_title("Ground magnetic field", fontsize=15*font_scale)

    # Hall conductance
    # ----------------
    ax4 = axes[1,0]
    ax4.contourf(sxi, seta, HallG, cmap=CMAP, levels=colorscales['hall'], zorder=0, extend='both')
    lompe.visualization.plot_coastlines(ax4, osse_Emodel, color = 'grey')
    lompe.visualization.plot_mlt(ax4, osse_Emodel, ntime, apx, color = 'grey')
    ax4.set_title("Hall conductance", fontsize=15*font_scale)

    # Pedersen conductance
    # --------------------
    ax5 = axes[1,1]
    ax5.contourf(sxi, seta, PedersenG, cmap=CMAP, levels=colorscales['pedersen'], zorder=0, extend='both')
    lompe.visualization.plot_coastlines(ax5, osse_Emodel, color = 'grey')
    lompe.visualization.plot_mlt(ax5, osse_Emodel, ntime, apx, color = 'grey')
    ax5.set_title("Pedersen conductance", fontsize=15*font_scale)

    # Current densities
    # -----------------
    ax6 = axes[1,2]
    ax6.quiver(x, y, jx, jy, zorder=3, scale=quiverscales['electric_current'], scale_units="inches")
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
    cbarax1.contourf(xx, yy, xx, cmap = plt.cm.bwr, levels = fac_levels * 1e6) # in [muA/m^2]  
    cbarax1.set_xlabel(r'$\mu$A/m$^2$')
    cbarax1.set_yticks([])
    buax = plt.twiny(cbarax1)
    buax.set_xlim(colorscales['ground_mag'].min() *1e9, colorscales['ground_mag'].max() * 1e9) # in [nT]
    buax.set_xlabel('nT')

    conductance_levels = colorscales['hall']
    xx = np.vstack((conductance_levels, conductance_levels)) 
    yy = np.vstack((np.zeros_like(conductance_levels), np.ones_like(conductance_levels)))
    cbarax2.contourf(xx, yy, xx, levels = conductance_levels, cmap = CMAP)
    cbarax2.set_xlabel('mho')
    cbarax2.set_yticks([])

    # Finish
    # ------
    top = np.clip(0.82 - 0.06*(1/ar - 1) - 0.018*np.exp(-((ar - 1.33)/0.18)**2), 0.72, 0.91)
    plt.subplots_adjust(top=top, bottom=0.065, left=0.01, right=0.98) 

    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()

    return fig_gamera
