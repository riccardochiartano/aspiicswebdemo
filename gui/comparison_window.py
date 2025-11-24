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


from logic.utils import sun_center, remove_occulter, define_mask, file_to_smap


base_dir = Path(__file__).resolve().parent.parent

tabname = 'comparison'

def comparison_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    with col_main:
        st.subheader("Comparison")

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
