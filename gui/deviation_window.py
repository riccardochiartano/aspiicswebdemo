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

from logic.demodulation import demodulate_merged, save_all
from logic.merging import merge
from logic.utils import download_all_maps_btn, deviation_from_RF
from logic.plot import plot_hist_AoLP

base_dir = Path(__file__).resolve().parent.parent


def deviation_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    with col_main:
        st.subheader("Deviation from Reference Frame")
        
        if st.session_state.current_map is None and not st.session_state.demod_maps:
            st.info("No map loaded...")
            return
    
        st.write('Be sure to load an AoLP image.')
        nbins = st.number_input('Number of bins: (if 0, it uses Sturges law)', min_value=0 ,step=1)
        if st.button("Plot deviation from RF"):
            if np.nanmax(st.session_state.current_map.data) <= np.pi/2 and np.nanmin(st.session_state.current_map.data) >= -np.pi/2:
                rad =  True
            else: 
                rad = False
            if st.session_state.current_map.meta['bunit'] == 'rad' or rad==True:
                deviation_from_RF(st.session_state.current_map, nbins)
            else:
                st.error('Load an image in "rad" units.')
