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
from matplotlib import colors
import plotly.express as px
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio


from logic.utils import sun_center, remove_occulter, define_mask, file_to_smap, radial_profile, polar_profile
from logic.plot import plot_rad_profile, plot_pol_profile

base_dir = Path(__file__).resolve().parent.parent

tabname = 'comparison'

def comparison_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])

    with col_main:
        st.subheader("Comparison")

        tab_profiles, tab_maps = st.tabs(["Profile comparison", "Map comparison"])

        with tab_profiles:
            profile_comparison_tab()

        with tab_maps:
            map_comparison_tab()

def map_comparison_tab():
    with st.form(key='comparison_form'):
        col1, col2 = st.columns([1,1])
        with col1:
            file1_name = st.text_input("File name:", key='file1name_txtin', placeholder='File 1')
            file1 = st.file_uploader("Upload first file:", type=["fits", "fts"])
        with col2: 
            file2_name = st.text_input("File name:", key='file2name_txtin', placeholder='File 2')
            file2 = st.file_uploader("Upload second file:", type=["fits", "fts"])
        
        header_comp = st.checkbox('Compare headers')
        
        with st.container(horizontal=True):
            st.space('stretch')
            do_compare = st.form_submit_button("Compare maps")
            st.space('stretch')

    if do_compare:
        try:
            map1 = file_to_smap(file1)
            map2 = file_to_smap(file2)
            show_comparison(map1, map2, file1_name, file2_name, header_comp)
        except:
            st.error('Upload all the files first')

def profile_comparison_tab():
    with st.form(key='prof_comparison_form'):
        col1, col2 = st.columns([1,1])
        with col1:
            file1_name = st.text_input("File name:", key='prof_file1name_txtin', placeholder='File 1')
            file1 = st.file_uploader("Upload first file:", type=["fits", "fts"])
            rmin1 = st.number_input('R min for first file', value=1.0)
            rmax1 = st.number_input('R_max for first file', value=3.0)
            angle1 = st.number_input('Angle for first file', value=0.0)
            ampl1 = st.number_input('Amplitude for first file', value=1.0)
        with col2:
            file2_name = st.text_input("File name:", key='prof_file2name_txtin', placeholder='File 2')
            file2 = st.file_uploader("Upload second file:", type=["fits", "fts"])
            rmin2 = st.number_input('R min for second file', value=1.0)
            rmax2 = st.number_input('R_max for second file', value=3.0)
            angle2 = st.number_input('Angle for second file', value=0.0)
            ampl2 = st.number_input('Amplitude for second file', value=1.0)

        with st.container(horizontal=True):
            st.space('stretch')
            do_profiles = st.form_submit_button("Compare profiles")
            st.space('stretch')

    if do_profiles:
        try:
            map1 = file_to_smap(file1)
            map2 = file_to_smap(file2)
            show_profile_comparison(
                map1, map2,
                file1_name, file2_name,
                angle1, rmin1, rmax1, ampl1,
                angle2, rmin2, rmax2, ampl2,
            )
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.info("BETA: Only radial profiles for now..")

def show_comparison(m1, m2, m1_name='File 1', m2_name='File 2', header_comp=False):
    if header_comp:
        st.info('"Compare headers" still being implemented..')

    if not m1_name:  
        m1_name = 'File 1'
    if not m2_name:
        m2_name = 'File 2'
    #m1_name = m1.meta.get('filename', 'File 1')
    #m2_name = m2.meta.get('filename', 'File 2')

    # remove occulter
    m1 = remove_occulter(m1)
    m2 = remove_occulter(m2)

    diff = m1.data - m2.data
    divide = m1.data / m2.data
    #m_diff = sunpy.map.Map(diff, m1.meta)

    st.pyplot(diff_ratio_images(diff, divide, m1_name, m2_name))
    st.pyplot(diff_ratio_hists(diff, divide, m1_name, m2_name))
    
    return

def show_profile_comparison(m1, m2,
                            m1_name='File 1', m2_name='File 2',
                            angle1=0.0, rmin1=1.0, rmax1=2.5, ampl1=1.0,
                            angle2=0.0, rmin2=1.0, rmax2=2.5, ampl2=1.0,
                            nradii=100):
    if not m1_name:
        m1_name = m1.meta.get('filename', 'File 1')
    if not m2_name:
        m2_name = m2.meta.get('filename', 'File 2')

    rad_prof1, radii1, err1 = radial_profile(
        m1, np.deg2rad(angle1), rmin1, rmax1, np.deg2rad(ampl1), nradii=nradii
    )
    rad_prof2, radii2, err2 = radial_profile(
        m2, np.deg2rad(angle2), rmin2, rmax2, np.deg2rad(ampl2), nradii=nradii
    )

    fig_rad, ax_rad = plt.subplots(figsize=(6, 4), clear=True)

    fig_rad = plot_rad_profile(fig_rad, ax_rad, m1, rad_prof1, radii1, angle1, label=f'{m1_name} @ {angle1:.0f}°')
    fig_rad = plot_rad_profile(fig_rad, ax_rad, m2, rad_prof2, radii2, angle2, label=f'{m2_name} @ {angle2:.0f}°')

    ax_rad.legend()
    ax_rad.grid()
    ax_rad.set_yscale('log')
    st.pyplot(fig_rad)    

def diff_ratio_images(diff, divide, m1_name, m2_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 

    #im1 = axes[0].imshow(diff, origin='lower', cmap='gray', vmin=1e-10, vmax=1e-8)
    vmin, vmax = np.nanpercentile(diff, [1, 99])
    im1 = axes[0].imshow(diff, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Difference: {m1_name} - {m2_name}')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Difference')

    #vmin, vmax = np.percentile(divide, [0, 100])
    #im2 = axes[1].imshow(divide, origin='lower', cmap='gray', vmin=1, vmax=1.2)
    vmin, vmax = np.nanpercentile(divide, [1, 99])
    im2 = axes[1].imshow(divide, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Ratio: {m1_name} / {m2_name}')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Ratio')

    for ax in axes:
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')

    plt.tight_layout()
    
    return fig

def diff_ratio_hists(diff, divide, m1_name, m2_name):
    diff_vals = diff[np.isfinite(diff)]
    #vmin, vmax = np.nanpercentile(diff_vals, [1, 99])
    #diff_vals = diff_vals[(diff_vals >= vmin) & (diff_vals <= vmax)]
    divide_vals = divide[np.isfinite(divide)]
    #vmin, vmax = np.nanpercentile(divide, [1, 99])
    #divide_vals = divide_vals[(divide_vals >= vmin) & (divide_vals <= vmax)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(diff_vals, bins=200, density=True, color='gray')
    axes[0].set_title(f'Difference: {m1_name} - {m2_name}')
    axes[0].set_xlabel('Difference value')

    axes[1].hist(divide_vals, bins=200, density=True, color='gray')
    axes[1].set_title(f'Ratio: {m1_name} / {m2_name}')
    axes[1].set_xlabel('Ratio value')

    plt.tight_layout()
    
    return fig

