import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import tempfile
import os
import pandas as pd
from scipy.interpolate import interp1d, griddata
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.special import gamma

from logic.utils import file_to_smap, download_map_btn, sun_center
from logic.plot import plot_map

base_dir = Path(__file__).resolve().parent.parent


def e_density_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    with col_main:
        st.subheader("Electron Density Map")

        pb_file = st.file_uploader("Upload pB map:", type=["fits", "fts"])

        if st.button('Calculate Electron Density map'):

            if pb_file:                
                pb_map = file_to_smap(pb_file)
                #pb_map = pb_map * 2.082e+20
    
                ne_map = ne_calc(pb_map)
                
                st.pyplot(plot_map(ne_map))
                download_map_btn(ne_map)
            else:
                st.error('Load maps first.')


def ne_calc(pb_map):
    r, angles_grid, rsun_arcsec = precompute_geometry(pb_map)
    ampl = 1
    n_angles = int((360)/ampl)
    ne_profiles = []
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    r_in, r_out = 1.1, 3.0
    for angle in angles:
        rad_prof, radii, rad_error = radial_profile_fast(
            pb_map, r, angles_grid, rsun_arcsec,
            angle, start_rsun=r_in, end_rsun=r_out, ampl=np.deg2rad(ampl)
        )

        params, _, _ = fit_pb_profile(radii, rad_prof, rad_error)
        
        ne_prof = calc_ne_profile(params, radii)

        ne_profiles.append(ne_prof)
    
    ne_profiles = np.array(ne_profiles)

    ne_image = radial_profiles_to_map(pb_map, ne_profiles, angles, radii)
    
    ne_header = make_ne_header(pb_map.meta) 
    
    ne_map = sunpy.map.Map(ne_image, ne_header)

    #st.write(profiles[:3], angles[:3])
    return ne_map

def fit_pb_profile(solar_radii, pb_intensity, sigma_pb):
    params0 = [1e-7, 1e-7, 7, 7]
    inf_bounds = [0, 0, 0, 0]
    sup_bounds = [1e-5, 1e-5, np.inf, np.inf]
    sigma_pb = np.ones(pb_intensity.size) * 1e-7
    params, cov_matrix = curve_fit(pb_func, solar_radii, pb_intensity, sigma=sigma_pb, p0=params0,
                        bounds=(inf_bounds, sup_bounds))
    #chi quadro
    residui = pb_intensity - pb_func(solar_radii, params[0], params[1], params[2], params[3])
    ssr = np.sum((residui/sigma_pb)**2)
    df = (len(solar_radii) - len(params))
    chi_quadro = ssr / df

    errors = np.sqrt(np.diag(cov_matrix))

    return params, errors, chi_quadro

    table_data = {
        "c0": [f"{params[0]:.1e} ± {errors[0]:.1e}"],
        "c1": [f"{params[1]:.1e} ± {errors[1]:.1e}"],
        "d0": [f"{params[2]:.1f} ± {errors[2]:.1f}"],
        "d1": [f"{params[3]:.1f} ± {errors[3]:.1f}"],
        "Chi-squared": [f"{chi_quadro:.2e}"]
    }

    table = Table(table_data)
    st.write(table)

    fit_func = pb_func(solar_radii, params[0], params[1], params[2], params[3])

    fig = plt.figure()
    ax2 = fig.add_subplot()
    ax2.errorbar(solar_radii, pb_intensity, yerr=sigma_pb, fmt='o-', label='Polarized Brightness', capsize=3)
    ax2.plot(solar_radii, fit_func, label=f'fit (χ²_rid = {chi_quadro:.2e})')
    ax2.set_xlabel("Solar radii [Rsun]")
    ax2.set_ylabel(f"pB [B/Bsun]")
    ax2.set_yscale('log')
    ax2.legend()
    
    st.pyplot(fig)

def calc_ne_profile(params, solar_radii):
    u_ld = 0.63                 #coefficiente limb-darkening 0.63 balboni
    sigma_t = 6.6524e-25        # cm^2    thomson cross sect 
    Rsun = 6.9634e10            # u.R_sun cm
    Bsun = 1                    # u.W / u.m^2

    K = (3*Bsun*sigma_t) / ((1-u_ld/3)*16*np.pi)

    b0 = params[2] + 1
    b1 = params[3] + 1
    a0 = params[0] * (np.sqrt(np.pi)*K*Rsun)**(-1) * (gamma((params[2]+3)*0.5) / gamma((params[2]+2)*0.5))
    a1 = params[1] * (np.sqrt(np.pi)*K*Rsun)**(-1) * (gamma((params[3]+3)*0.5) / gamma((params[3]+2)*0.5))

    ne = ne_func(solar_radii, a0, a1, b0, b1, u_ld)

    return ne


def pb_func(x, c0, c1, d0, d1):
    return c0*x**(-d0) + c1*x**(-d1)

def ne_func(r, a0, a1, b0, b1, u_ld):
    return (a0*r**(-b0) + a1*r**(-b1)) / ((1-u_ld)*A(r) + u_ld*B(r))

def sin_omega(r):
    return (1/r)        #r in R_sun

def cos_omega(r):
    return np.sqrt(1 - (sin_omega(r))**2)

def A(r):
    return (cos_omega(r) * (sin_omega(r))**2)

def B(r):
    term1 = 1 - 3 * sin_omega(r)**2
    term2 = cos_omega(r)**2 * ((1 + 3 * sin_omega(r)**2) / sin_omega(r))
    term3 = np.log((1 + sin_omega(r)) / cos_omega(r))
    return (-1/8 * (term1 - term2 * term3)) 


def precompute_geometry(solar_map):
    """Precalcola r e angles_grid per il riuso in più profili."""
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs
    sun_x, sun_y = sun_center(solar_map)

    y, x = np.indices(solar_map.data.shape)
    dx = (x - sun_x)
    dy = (y - sun_y)
    r = np.sqrt(dx**2 + dy**2) * scale  # in arcsec
    angles_grid = np.mod(np.arctan2(dy, dx), 2*np.pi)

    return r, angles_grid, rsun_arcsec


def radial_profile_fast(solar_map, r, angles_grid, rsun_arcsec,
                               angle, start_rsun, end_rsun, ampl=1, nradii=50):
    """Versione veloce: riusa geometria già calcolata."""
    angle_start, angle_end = angle - ampl/2, angle + ampl/2
    angle_mask = (angles_grid >= angle_start) & (angles_grid <= angle_end)

    radii = np.linspace(start_rsun, end_rsun, nradii+1)
    radii_avg = 0.5 * (radii[:-1] + radii[1:])
    radii_arcsec = radii * rsun_arcsec.value

    mask = np.isfinite(solar_map.data) & angle_mask
    r_vals = r[mask].ravel()
    data_vals = solar_map.data[mask].ravel()

    bin_idx = np.digitize(r_vals, radii_arcsec) - 1
    valid = (bin_idx >= 0) & (bin_idx < nradii)
    bin_idx = bin_idx[valid]
    data_vals = data_vals[valid]

    sum_vals = np.bincount(bin_idx, weights=data_vals, minlength=nradii)
    count_vals = np.bincount(bin_idx, minlength=nradii)
    mean_vals = np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals>0)

    sum_sq = np.bincount(bin_idx, weights=data_vals**2, minlength=nradii)
    variance = np.divide(sum_sq, count_vals, out=np.zeros_like(sum_sq), where=count_vals>0) - mean_vals**2
    variance[count_vals <= 1] = np.nan
    rad_error = np.sqrt(variance / count_vals)

    return mean_vals, radii_avg, rad_error

def radial_profiles_to_map(smap, ne_profiles, angles, radii, method='linear'):
    sun_x, sun_y = sun_center(smap)
    scale = smap.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = smap.rsun_obs.to(u.arcsec).value
    rsun_pix = rsun_arcsec / scale

    ne_profiles = np.array(ne_profiles)  # shape (n_angles, nradii)

    r_grid, theta_grid = np.meshgrid(radii, angles)
    x_grid = sun_x + r_grid * rsun_pix * np.cos(theta_grid)
    y_grid = sun_y + r_grid * rsun_pix * np.sin(theta_grid)

    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    values = ne_profiles.ravel()

    r_in = radii.min() * rsun_pix
    r_out = radii.max() * rsun_pix
    
    x, y = np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) 
    r = np.sqrt((x-sun_x)**2 + (y-sun_y)**2)

    mask = (r >= r_in) & (r <= r_out)

    ne_image = np.empty_like(smap.data)
    ne_image[mask] = griddata(points, values, (x[mask], y[mask]), method=method)

    ne_image[ne_image < 1] = np.nan
    ne_image[ne_image > 1e12] = np.nan

    return ne_image

def make_ne_header(header):
    if header.get('filename', None):
        filename = header.get('filename')
        parts = filename.split('_')
        parts[1] = 'ne'
        parts[2] = 'l3'
        header['filename'] = '_'.join(parts)
    header['level'] = 'L3'
    header['bunit'] = 'cm^-3'
    return header