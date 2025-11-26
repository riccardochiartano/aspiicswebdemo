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

        wedge = Wedge((sun_x, sun_y), 
                      r_out, 
                      angle-ampl/2, angle+ampl/2, 
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
        angle_start = angles[0]
        angle_end = angles[-1] 

        wedge = Wedge((sun_x, sun_y), 
                      r_out, 
                      angle_start, angle_end, 
                      width=r_out-r_in, 
                      facecolor=f'C{i}', alpha=0.5)
        ax.add_patch(wedge)
        ax.figure.canvas.draw()
    

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
    # if there's already a defined cmap
    if map_plot.plot_settings['cmap'] != 'gray':
        cmap = map_plot.plot_settings['cmap']
        #cmap.set_bad('black')
        return cmap

    dir_cmaps = os.path.join(base_dir, 'resources', 'rob_calib_data')
    c_path = os.path.join(dir_cmaps, 'wb_colormap.txt')
    filter = map_plot.meta.get('filename', '')
    if 'pb' in filter:
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
