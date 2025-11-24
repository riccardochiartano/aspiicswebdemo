import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import tempfile
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate

from logic.utils import file_to_smap, download_map_btn
from logic.plot import plot_map

base_dir = Path(__file__).resolve().parent.parent

########################################
# manca il reproject con stesso centro #
########################################

def continuum_sub_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    with col_main:
        st.subheader("Continuum subtraction (FeXIV/HeI)")

        spectrum = st.radio("Spectrum:",
                ["Allen", "AM0", "Koutchmy"],
                horizontal=True  
            )

        filter_file = st.file_uploader("Upload FeXIV or HeI map:", type=["fits", "fts"])
        WBF_file = st.file_uploader("Upload WBF map:", type=["fits", "fts"])

        if st.button('plot FeXIV/HeI no continuum'):

            interp_spectrum = load_spectrum(spectrum)
            if filter_file and WBF_file:
                if '_fe_' in filter_file.name and '_wb_' in WBF_file.name:
                    rate = filters_rate(interp_spectrum, type='FeXIV')
                elif '_he_' in filter_file.name and '_wb_' in WBF_file.name:
                    rate = filters_rate(interp_spectrum, type='HeI')
                else:
                    st.error('FeXIV/HeI file must contain "fe"/"he" and WBF file must contain "wb".')
                
                filter_map = file_to_smap(filter_file)
                WBF_map = file_to_smap(WBF_file)

                #filter_map = filter_map / filter_map.meta['exptime']
                #WBF_map = WBF_map / WBF_map.meta['exptime']
                
                filter_map = no_neg(filter_map)
                WBF_map = no_neg(WBF_map) 

                FeXIV_sub_image = filter_map.data - WBF_map.data * rate
                FeXIV_sub_map = sunpy.map.Map(FeXIV_sub_image, filter_map.meta)
                
                FeXIV_sub_map = change_header(FeXIV_sub_map)

                st.pyplot(plot_map(FeXIV_sub_map))
                download_map_btn(FeXIV_sub_map)
            else:
                st.error('Load maps first.')


def integrate_spectrum(interp_func, type, n_points=200):
    if type == 'WBF':
        x_min = 540.54
        x_max = 569.86
    if type == 'FeXIV':
        x_min = 530.09
        x_max = 530.67
    if type == 'HeI':
        x_min = 586.52
        x_max = 588.72

    x_new = np.linspace(x_min, x_max, n_points)
    y_new = interp_func(x_new)

    R = np.trapezoid(y_new, x_new)

    return R

def filters_rate(interp_func, type):
    B_filter = integrate_spectrum(interp_func, type)
    B_WBF = integrate_spectrum(interp_func, type='WBF')
    rate = B_filter/B_WBF
    st.write(f'B_{type} / B_WBF = {rate}')
    return rate

def no_neg(map):
    image = map.data
    mask = image < 0
    image[mask] = np.nan
    new_map = sunpy.map.Map(image, map.meta)
    return new_map

def change_header(map):
    filename = map.meta['filename']
    comps = filename.split('.')
    filename_new = comps[0] + '_sub.' + comps[1]
    parts = filename_new.split('_')
    parts[2] = 'l3'
    filename_new = '_'.join(parts)
    map.meta['filename'] = filename_new
    map.meta['bunit'] = 'MSB'
    return map

def load_spectrum(spectrum):
    if spectrum == 'Allen':
        csv_file = os.path.join(base_dir, 'resources', 'spectra', 'allen_spectrum.csv')
    elif spectrum == 'Koutchmy':
        csv_file = os.path.join(base_dir, 'resources', 'spectra', 'spectrum_all.csv')
    elif spectrum == 'AM0':
        csv_file = os.path.join(base_dir, 'resources', 'spectra', 'e490_00a_amo.csv')

    df = pd.read_csv(csv_file)
    lambda_ = df.iloc[:,0].values
    rad_ = df.iloc[:,1].values
    #interp_spectrum = interp1d(lambda_, rad_, kind='cubic', fill_value="extrapolate")
    interp_spectrum = interpolate.CubicSpline(lambda_,rad_,bc_type='natural')
    
    return interp_spectrum