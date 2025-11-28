import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import tempfile

from logic.process import wow_filter, log_scale
from logic.plot import plot_profile, plot_pol_profile, plot_rad_profile, aspiics_cmap
from logic.utils import sun_center, download_map_btn, download_profile, polar_profile, radial_profile


def profile_window():

    if 'show_radial' not in st.session_state:
        st.session_state.show_radial = None
    if 'show_polar' not in st.session_state:
        st.session_state.show_polar = None
    if 'rad_profiles' not in st.session_state:
        st.session_state.rad_profiles = {}
    if 'pol_profiles' not in st.session_state:
        st.session_state.pol_profiles = {}

    #col_left, col_main, col_right = st.columns([1, 3, 1])
    
    #with col_main:
    st.subheader("Plot Radial Profile")

    with st.form(key='rad_profile_form'):
        angle_deg = st.number_input("Angle [deg]", value=0.0, step=1.0)
        line_ampl = st.number_input("Line amplitude [deg]", value=1.0, step=1.0)
        start_rsun, end_rsun = st.slider(
            "Select profile start / end [Rsun]",
            min_value=1.0,
            max_value=10.0,
            value=(1.3, 3.0),
            step=0.1
        )

        r_submitted = st.form_submit_button("Plot radial profile")
    
    if r_submitted:
        if st.session_state.current_map is None:
            st.warning("Carica prima una mappa FITS")
        else:
            st.session_state.show_radial = True
            profile_coords, plot_radii, _ = radial_profile(st.session_state.current_map, np.deg2rad(angle_deg), start_rsun, end_rsun, np.deg2rad(line_ampl))
            #st.session_state.rad_profile, st.session_state.rad_plot_radii = profile_coords, plot_radii
            st.session_state.rad_profiles[angle_deg] = {
                "radii": plot_radii,
                "values": profile_coords,
                "ampl": line_ampl,
            }

    if st.session_state.get("show_radial", False):
        with st.container(horizontal=True):
            #st.space("stretch")
            rad_logscale = st.checkbox('Log scale', key='rlogscale_ckb')

    if st.session_state.get("show_radial", False):                
        fig_rad, ax_rad = plt.subplots(figsize=(6, 4), clear=True) 
        for angle, prof in st.session_state.rad_profiles.items():
            fig_rad = plot_rad_profile(fig_rad, ax_rad, st.session_state.current_map, prof["values"], prof["radii"], angle)
        ax_rad.legend()
        ax_rad.grid()
        if rad_logscale:
            ax_rad.set_yscale('log')
        st.pyplot(fig_rad)
        with st.container(horizontal=True):
            download_profile('rad', st.session_state.current_map.meta.get('bunit', 'bunit'), st.session_state.rad_profiles)
            st.space("stretch")
            if st.button("Reset plot", key='btn_reset_radplot'):
                st.session_state.rad_profiles = {}
                st.session_state.show_radial = False
                st.rerun()
    
    st.subheader("Plot Polar Profile")

    with st.form(key='pol_profile_form'):
        dist = st.number_input("Distance from solar centre [Rsun]", value=1.3, step=0.1)
        line_width = st.number_input("Line width [Rsun]", value=0.1, step=0.1)
        angle_start, angle_end = st.slider(
            "Select angle start / end [deg]",
            min_value=0,
            max_value=360,
            value=(0, 360),
            step=1
        )

        p_submitted = st.form_submit_button("Plot polar profile")

    if p_submitted:
        if st.session_state.current_map is None:
            st.warning("Carica prima una mappa FITS")
        else:
            st.session_state.show_polar = True
            profile, plot_angles = polar_profile(st.session_state.current_map, dist, angle_start, angle_end, step_rsun=line_width)
            #st.session_state.pol_profile, st.session_state.pol_plot_angles = profile, plot_angles
            st.session_state.pol_profiles[dist] = {
                    "angles": plot_angles,
                    "values": profile,
                    "width": line_width,
                }
            
    if st.session_state.get("show_polar", False): 
        with st.container(horizontal=True):
                pol_logscale = st.checkbox('Log scale', key='plogscale_ckb')

    if st.session_state.get("show_polar", False): 
        fig_pol, ax_pol = plt.subplots(figsize=(6, 4), clear=True) 
        for dist, prof in st.session_state.pol_profiles.items():
            fig_pol = plot_pol_profile(fig_pol, ax_pol, st.session_state.current_map, prof["values"], prof["angles"], dist)
        ax_pol.legend()
        ax_pol.grid()
        if pol_logscale:
            ax_pol.set_yscale('log')
        st.pyplot(fig_pol)
        with st.container(horizontal=True):
            download_profile('pol', st.session_state.current_map.meta.get('bunit', 'bunit'), st.session_state.pol_profiles)
            st.space("stretch")            
            if st.button("Reset plot", key='btn_reset_polplot'):
                st.session_state.pol_profiles = {}
                st.session_state.show_polar = False
                st.rerun()



'''def radial_profile(solar_map, angle, ampl, start_rsun, end_rsun):
    rsun_arcsec = solar_map.rsun_obs
    sun_x, sun_y = sun_center(solar_map)
    date_obs = solar_map.meta.get("date-obs")

    sun_center_coord = SkyCoord(
        Tx=0 * u.arcsec,
        Ty=0 * u.arcsec,
        frame=solar_map.coordinate_frame,  
    )

    angle_start, angle_end = angle - ampl/2, angle + ampl*2

    x, y = np.meshgrid(*[np.arange(v.value) for v in solar_map.dimensions])
    hpc_coords = solar_map.pixel_to_world(x * u.pixel, y * u.pixel)
    r = hpc_coords.separation(sun_center_coord).to(u.arcsec)

    angles_grid = np.arctan2(y - sun_y, x - sun_x)
    angles_grid = np.mod(angles_grid, 2 * np.pi)

    radii = np.linspace(start_rsun, end_rsun, 50+1)
    radii_avg = 0.5 * (radii[:-1] + radii[1:])
    radii_arcsec = radii * rsun_arcsec

    rad_profile = np.zeros(50)

    for i in range(50):
        r_start, r_end = radii_arcsec[i], radii_arcsec[i+1]
            
        mask = (
            (r >= r_start) & 
            (r <= r_end) & 
            (angles_grid >= angle_start) & 
            (angles_grid <= angle_end)
        )
        
        rad_profile[i] = np.nanmean(solar_map.data[mask])

    return rad_profile, radii_avg


def polar_profile(solar_map, dist, first_angle, last_angle, n_angles=360):
    rsun_arcsec = solar_map.rsun_obs /u.R_sun
    sun_x, sun_y = sun_center(solar_map)
    date_obs = solar_map.meta.get("date-obs")

    sun_center_coord = SkyCoord(
        Tx=0 * u.arcsec,
        Ty=0 * u.arcsec,
        frame=solar_map.coordinate_frame,  
    )

    angles = np.linspace(np.deg2rad(first_angle), np.deg2rad(last_angle), n_angles+1)
    angles_medi = 0.5 * (angles[:-1] + angles[1:])

    x, y = np.meshgrid(*[np.arange(v.value) for v in solar_map.dimensions])
    hpc_coords = solar_map.pixel_to_world(x * u.pixel, y * u.pixel)
    r = hpc_coords.separation(sun_center_coord).to(u.arcsec)

    angles_grid = np.arctan2(y - sun_y, x - sun_x)
    angles_grid = np.mod(angles_grid, 2 * np.pi)

    # aggiungi insert step
    step = 0.1 *u.R_sun
    r_in = dist *u.R_sun
    r_out = r_in + step
    r_in_arcsec = (r_in * rsun_arcsec) 
    r_out_arcsec = (r_out * rsun_arcsec) 

    pol_profile = np.zeros(n_angles)

    for i in range(n_angles):
        angle_start, angle_end = angles[i], angles[i+1]
            
        mask = (
            (r >= r_in_arcsec) & 
            (r <= r_out_arcsec) & 
            (angles_grid >= angle_start) & 
            (angles_grid <= angle_end)
        )
        
        pol_profile[i] = np.nanmean(solar_map.data[mask])

    return pol_profile, np.rad2deg(angles_medi)'''