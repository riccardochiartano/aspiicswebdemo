from sunpy.coordinates import frames
import sunpy.map
import streamlit as st
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import FancyArrow, Circle, Wedge
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

def sun_center (solar_map):
    sun_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=solar_map.coordinate_frame)
    sun_x, sun_y = solar_map.wcs.world_to_pixel(sun_center)
    return (int(sun_x), int(sun_y))

def plot_NE_labels(ax, solar_map, length_arcsec=600, offset=90, color='gray', fontsize=17, draw_limb=False, limb_c='gray', limb_s='-'):

    center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=solar_map.coordinate_frame)

    north = SkyCoord(0*u.arcsec,  length_arcsec*u.arcsec, frame=solar_map.coordinate_frame)
    south = SkyCoord(0*u.arcsec,  -length_arcsec*u.arcsec, frame=solar_map.coordinate_frame)
    east  = SkyCoord(-length_arcsec*u.arcsec, 0*u.arcsec, frame=solar_map.coordinate_frame)
    west  = SkyCoord(length_arcsec*u.arcsec, 0*u.arcsec, frame=solar_map.coordinate_frame)

    cx, cy = solar_map.world_to_pixel(center)
    nx, ny = solar_map.world_to_pixel(north)
    sx, sy = solar_map.world_to_pixel(south)
    ex, ey = solar_map.world_to_pixel(east)
    wx, wy = solar_map.world_to_pixel(west)

    dxN, dyN = nx.value - cx.value, ny.value - cy.value
    dxE, dyE = ex.value - cx.value, ey.value - cy.value
    dxS, dyS = sx.value - cx.value, sy.value - cy.value
    dxW, dyW = wx.value - cx.value, wy.value - cy.value

    normN = np.hypot(dxN, dyN)      #same as np.sqrt(dxN**2 + dyN**2)
    normE = np.hypot(dxE, dyE)
    normS = np.hypot(dxS, dyS)
    normW = np.hypot(dxW, dyW)

    fontsize = fontsize/(solar_map.scale[0].value)**0.4
    offset = offset/solar_map.scale[0].value

    # "N"
    ax.text(nx.value + offset * dxN / normN,
            ny.value + offset * dyN / normN,
            'N', color=color, fontsize=fontsize,
            ha='center', va='center', fontweight='bold')

    # "E"
    ax.text(ex.value + offset * dxE / normE,
            ey.value + offset * dyE / normE,
            'E', color=color, fontsize=fontsize,
            ha='center', va='center', fontweight='bold')
    
    # "S"
    ax.text(sx.value + offset * dxS / normS,
            sy.value + offset * dyS / normS,
            'S', color=color, fontsize=fontsize,
            ha='center', va='center', fontweight='bold')

    # "W"
    ax.text(wx.value + offset * dxW / normW,
            wy.value + offset * dyW / normW,
            'W', color=color, fontsize=fontsize,
            ha='center', va='center', fontweight='bold')
    
    if draw_limb:
        rsun = 960 * u.arcsec
        rsun_pix = rsun.value / solar_map.scale[0].value
        circle = Circle((cx.value, cy.value), radius=rsun_pix,
                        edgecolor=color, facecolor='none', lw=1.5, linestyle='--')
        ax.add_patch(circle)


def plot_profile(solar_map, line, r_start):
    '''
    Plotta il profilo di intensità della linea. 
    '''
    intensity_coords = sunpy.map.pixelate_coord_path(solar_map, line)
    intensity = sunpy.map.sample_at_coords(solar_map, intensity_coords)
    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    rsun = solar_map.rsun_obs.to(u.arcsec)
    angular_separation_rsun = (angular_separation/rsun).decompose().value + r_start

    fig, ax = plt.subplots(figsize=(6, 4))  
    ax.plot(angular_separation_rsun, intensity)
    ax.set_xlabel("Angular distance from solar center [Rsun]")
    ax.set_ylabel(f"Intensity [{solar_map.meta['bunit']}]")
    return fig

def plot_rad_profile(fig, ax, solar_map, profile, radii, angle):
    ax.plot(radii, profile, label=f'{angle:.0f}°')
    ax.set_xlabel("Distance from solar center [Rsun]")
    ax.set_ylabel(f"Intensity [{solar_map.meta.get('bunit', 'bunit')}]")
    return fig

def plot_pol_profile(fig, ax, solar_map, profile, angles, dist):
    ax.plot(angles, profile, label=f'{dist:.1f} Rsun')
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel(f"Intensity [{solar_map.meta.get('bunit', 'bunit')}]")
    return fig

def plot_hist_AoLP(image, nbins):
    # regola di Sturges
    if nbins == 0:
        angles = np.array(image).flatten() 
        n = len(angles) 
        nbins = int(np.ceil(np.log2(n) + 1))
    
    counts, bins = np.histogram(image, bins=nbins)
    psi_mean = np.nanmean(image)
    psi_std = np.nanstd(image)
    peak_bin_index = np.argmax(counts)
    peak_value = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2  
    print(peak_value)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(image, bins=nbins, histtype='step')
    ax.axvline(peak_value, color='C0', linestyle='--', alpha=0.5, label=f'Peak = {peak_value:.2f}°')
    ax.axvline(psi_mean, color='C1', linestyle='--', alpha=0.5, label=f'AoLP avg = {psi_mean:.3f}° +- {psi_std:.3f}')
    ax.set_title(f'AoLP Local RF')
    ax.set_xlabel('[°]')
    ax.set_ylabel('[counts]')
    ax.legend()
    ax.set_xlim(-25, 25)

    return fig

def plot_local_RF_AoLP(image, title):
    #plot
    fig, ax = plt.subplots()
    im = ax.imshow(image, origin='lower', vmin=-5, vmax=5)
    ax.set_xlabel('Solar X [arcsec]')
    ax.set_ylabel('Solar Y [arcsec]')
    cbar = plt.colorbar(im, ax=ax, label='Deviation from tangentiality angle [deg]')
    plt.title(title)
    
    return fig

def plot_rprof(ax, solar_map, profiles):
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs.value
    sun_x, sun_y = sun_center(solar_map)

    for i, (angle, prof) in enumerate(profiles.items()):
        radii = prof['radii']
        ampl = prof['ampl']
        r_in = radii[0] * rsun_arcsec / scale
        r_out = radii[-1] * rsun_arcsec / scale
        
        #angle_start = (angle - ampl/2)
        #angle_end = (angle + ampl/2)
        angle_start = 90 + (angle - ampl/2)
        angle_end = 90 + (angle + ampl/2)

        wedge = Wedge((sun_x, sun_y), 
                      r_out, 
                      angle_start, angle_end, 
                      width=r_out-r_in, 
                      facecolor=f'C{i}', alpha=0.5)
        ax.add_patch(wedge)
        ax.figure.canvas.draw()

def plot_pprof(ax, solar_map, profiles):
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs.value
    sun_x, sun_y = sun_center(solar_map)

    for i, (dist, prof) in enumerate(profiles.items()):
        angles = prof['angles']
        width = prof['width']
        r_in = dist * rsun_arcsec / scale
        r_out = r_in + width * rsun_arcsec / scale
        #angle_start = angles[0]
        #angle_end = angles[-1] 
        angle_start = 90 + angles[0]
        angle_end = 90 + angles[-1] 

        wedge = Wedge((sun_x, sun_y), 
                      r_out, 
                      angle_start, angle_end, 
                      width=r_out-r_in, 
                      facecolor=f'C{i}', alpha=0.5)
        ax.add_patch(wedge)
        ax.figure.canvas.draw()

##############################
# just one profile at a time #
##############################

def plot_onerprof(ax, solar_map, radii, angle, ampl, color='gray'):
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs.value
    sun_x, sun_y = sun_center(solar_map)

    r_in = radii[0] * rsun_arcsec / scale
    r_out = radii[-1] * rsun_arcsec / scale

    # avendo angle in rif solare
    angle_start = 90 - (angle + ampl/2)
    angle_end = 90 - (angle - ampl/2)

    wedge = Wedge((sun_x, sun_y), 
                    r_out, 
                    angle_start, angle_end, 
                    width=r_out-r_in, 
                    facecolor=color, alpha=0.5)
    wedge.set_transform(ax.get_transform('pixel'))
    ax.add_patch(wedge)
    ax.figure.canvas.draw()
    
def plot_onepprof(ax, solar_map, angles, width, dist, color='gray'):# profiles):
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs.value
    sun_x, sun_y = sun_center(solar_map)

    r_in = dist * rsun_arcsec / scale
    r_out = r_in + width * rsun_arcsec / scale
    angle_start = 90 - angles[-1]
    angle_end = 90 - angles[0] 

    wedge = Wedge((sun_x, sun_y), 
                    r_out, 
                    angle_start, angle_end, 
                    width=r_out-r_in, 
                    facecolor=color, alpha=0.6)
    ax.add_patch(wedge)
    ax.figure.canvas.draw()

##############################

def plot_stars(ax, catalog_stars):
    offset_x = 10  
    offset_y = 10
    if not catalog_stars.empty:
        ax.scatter(catalog_stars.xsensor, catalog_stars.ysensor,
                s=80, edgecolors='orange', facecolors='none', 
                linewidths=1, linestyle='--', alpha=0.8)
        for i, row in catalog_stars.iterrows():
            ax.text(row.xsensor + offset_x, row.ysensor + offset_y, 
                    str(row.main_id), color='orange', fontsize=6, 
                    alpha=0.7, clip_on=True)
    else:
        st.write('No stars in frame.')

##############################

def plot_map(map):
    #fig = plt.figure()
    #ax = fig.add_subplot(projection=map.wcs)
    #map.plot(axes=ax)
    #map.draw_grid(axes=ax)
    #
    #return fig
    image_plot = map.data
    flat_data = image_plot[np.isfinite(image_plot)].flatten()
    vmin, vmax = np.percentile(flat_data, [1,99])

    fig_map, ax_map = plt.subplots(figsize=(6,6), subplot_kw={'projection': map.wcs})
    im = map.plot(axes=ax_map, clim=(vmin, vmax), cmap=aspiics_cmap_new(map))
    ax_map.grid(True)  # opzionale: rimuove griglia
    cbar = plt.colorbar(im, ax=ax_map, label=f"Intensity [{map.meta.get('bunit', '')}]")
    
    return fig_map

def aspiics_cmap(map_plot):
    color = '#92ff00'
    if 'fe' in map_plot.meta['filename']:
        color = '#2ffda4'
    if 'he' in map_plot.meta['filename']:
        color = '#ffe000'
    if map_plot.meta['bunit'] == 'rad':
        color = "#FFFFFF"
        
    cmap = LinearSegmentedColormap.from_list('greenish', ['#000000', color], N=256)
    cmap.set_bad('black')
    return cmap

def aspiics_cmap_new(map_plot):
    if "metis" in map_plot.meta["filename"]:
        obsv = map_plot.observatory
        instr = map_plot.instrument
        prod = get_prodtype(map_plot)
        cmap_string = f"{obsv}{instr}{prod}".lower()
        return metis_color_table(cmap_string)

    # if there's already a defined cmap
    if map_plot.plot_settings['cmap'] != 'gray':
        cmap = map_plot.plot_settings['cmap']
        #cmap.set_bad('black')
        return cmap

    dir_cmaps = os.path.join(base_dir, 'resources', 'rob_calib_data')
    c_path = os.path.join(dir_cmaps, 'wb_colormap.txt')
    filter = map_plot.meta.get('filename', '')
    if 'pB' in filter or 'pb' in filter:
        c_path = os.path.join(dir_cmaps, 'p_colormap.txt')
    if 'fe' in filter:
        c_path = os.path.join(dir_cmaps, 'fe_colormap.txt')
    if 'he' in filter:
        c_path = os.path.join(dir_cmaps, 'he_colormap.txt')
    if 'ne' in filter:
        c_path = os.path.join(dir_cmaps, 'ne_colormap.txt')
        
    if map_plot.meta.get('bunit', '') == 'rad':
        color = "#FFFFFF"
        cmap = LinearSegmentedColormap.from_list('greenish', ['#000000', color], N=256)
        cmap.set_bad('black')
        return cmap
        
    colortable = np.loadtxt(c_path)
    cmap = ListedColormap(colortable, name="aspiics_cmap")
    cmap.set_bad('black')
    #cmap.set_bad(color=(0, 0, 0, 0))
    return cmap


def get_prodtype(smap):
    """
    Define the type of the Metis data product.

    Returns
    -------
    prodtype : `str`
        Name of the Metis data product.

    """
    
    btype_suff_dict = {
        'VL total brightness':             ('-TB', '-TB'), 
        'VL polarized brightness':         ('-PB', '-PB'), 
        'VL fixed-polarization intensity': ('-FP', '-Fix. Pol.'), 
        'VL polarization angle':           ('-PA', '-Pol. Angle'), 
        'Stokes I':                        ('-SI', '-Stokes I'), 
        'Stokes Q':                        ('-SQ', '-Stokes Q'), 
        'Stokes U':                        ('-SU', '-Stokes U'),
        'Pixel quality':                   ('-PQ', '-Pixel quality'), 
        'Absolute error':                  ('-AE', '-Abs. err.'),
        'Relative error':                  ('-RE', '-Rel. err.'), #modAB2
        'UV Lyman-alpha intensity':        ('', ''),
    }
    
    btype = smap.meta['btype']
    prodtype = smap.meta['filter']
    
    if btype in btype_suff_dict:
        suff, nickname_add = btype_suff_dict[btype]
        prodtype += suff
        #smap._nickname += nickname_add 
    else:
        raise ValueError(
            f"Error. smap.meta['btype']='{btype}' is not known."
        ) 

    return prodtype

def metis_color_table(cmap_name):
    """
    Credits: V. Andretta, A. Liberatore, A. Burtovoi, G. Jerse
    NB:
     - Names from _get_cmap_name()_ should be defined in sunpy.visualization.colormaps.cm
     - They can be in turn defined by calling metis_color_table() inserted in sunpy.visualization.colormaps.color_tables
     - Current function is a prototype of function which should be inserted in sunpy.visualization.colormaps.color_tables
     - [?] Should we define different colormaps for L0 and L1?
    """
    ### Temporary imports ###
    import matplotlib
    import sunpy.visualization.colormaps as cm
    import cmcrameri

    #st.write(cmap_name)

    if cmap_name == 'solar orbitermetisvl-tb':
# #         aia_wave_dict = create_aia_wave_dict()
#         aia_wave_dict = cm.color_tables.create_aia_wave_dict()  # temp
#         r, g, b = aia_wave_dict[193*u.angstrom]
#         cmap = cm.color_tables._cmap_from_rgb(
#             r, g, b, 'SolO Metis VL Total Brightness'
#         )
        #cmap = matplotlib.colormaps['pink'].copy()  # ASk Vincenzo ???  # np.savetxt('py_cmap_pink.csv', np.array(cmap.colors)*255, delimiter=',')
        #cmap = cmcrameri.cm.batlow.copy()
        cmap = cmcrameri.cm.batlow.copy()
        # chk also Stokes I
        cmap.name =  'SolO Metis VL Total Brightness'
        
    elif cmap_name == 'solar orbitermetisvl-pb':
        '''
        Metis VL/pB images uses AIA color table
        '''
#         aia_wave_dict = create_aia_wave_dict()
        aia_wave_dict = cm.color_tables.create_aia_wave_dict()  # temp
        r, g, b = aia_wave_dict[304*u.angstrom]
        cmap = cm.color_tables._cmap_from_rgb(
            r, g, b, 'SolO Metis VL Polarized Brightness'
        )

    elif cmap_name == 'solar orbitermetisvl-fp':
        r = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
        g = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
        b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 36, 38, 39, 40, 40, 42, 43, 44, 44, 46, 47, 48, 48, 50, 51, 52, 52, 54, 55, 56, 56, 58, 59, 60, 60, 62, 63, 64, 65, 65, 67, 68, 69, 70, 71, 72, 73, 73, 75, 76, 77, 78, 79, 80, 81, 81, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 97, 99, 100, 101, 102, 103, 104, 105, 105, 107, 108, 109, 110, 111, 112, 113, 113, 115, 116, 117, 118, 119, 120, 121, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 243, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255])
        cmap = cm.color_tables._cmap_from_rgb(
            r, g, b, 'SolO Metis VL Fixed Polarization'
        )

    elif cmap_name == 'solar orbitermetisvl-pa':
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.name =  'SolO Metis VL Polarization Angle'

    elif cmap_name == 'solar orbitermetisvl-si':
        #cmap = matplotlib.colormaps['pink'].copy()  # ASk Vincenzo ???  # np.savetxt('py_cmap_pink.csv', np.array(cmap.colors)*255, delimiter=',')
        cmap = cmcrameri.cm.batlow.copy()
        cmap.name =  'SolO Metis VL Stokes I'

    elif cmap_name == 'solar orbitermetisvl-sq':
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.name =  'SolO Metis VL Stokes Q'

    elif cmap_name == 'solar orbitermetisvl-su':
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.name =  'SolO Metis VL Stokes U'

    elif cmap_name == 'solar orbitermetisvl-pq':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also UV PQ
        cmap.name =  'SolO Metis VL Pixel Quality'

    elif cmap_name == 'solar orbitermetisvl-ae':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also UV AE
        cmap.name =  'SolO Metis VL Absolute Error'

    elif cmap_name == 'solar orbitermetisvl-re':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also UV RE
        cmap.name =  'SolO Metis VL Relative Error'
        
    elif cmap_name == 'solar orbitermetisuv':
        cmap = matplotlib.colormaps['Blues_r'].copy()  # Blues_r, PuBu, BuGn
        cmap.name =  'SolO Metis UV'

    elif cmap_name == 'solar orbitermetisuv-pq':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also VL PQ
        cmap.name =  'SolO Metis UV Pixel Quality'

    elif cmap_name == 'solar orbitermetisuv-ae':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also VL AE
        cmap.name =  'SolO Metis UV Absolute Error'

    elif cmap_name == 'solar orbitermetisuv-re':
        cmap = matplotlib.colormaps['plasma'].copy()  # cividis, plasma
        # chk also VL RE
        cmap.name =  'SolO Metis UV Relative Error'


    cmap.set_bad(color='k')
    
    return cmap
