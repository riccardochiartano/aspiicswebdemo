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

from logic.utils import file_to_smap, download_map_btn, sun_center, fcorona_removed_map, fcorona_added_map
from logic.plot import plot_map

base_dir = Path(__file__).resolve().parent.parent

def fcorona_window():
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    with col_main:
        st.subheader("Add/Remove F-Corona model")

        wbf_file = st.file_uploader("Upload WBF map:", type=["fits", "fts"])

        #### to add load from filename
        ###upl_filename = st.text_input("Filename:", "", key=f'ulfn_{tabname}', help="Must be the exact 'filename.fits'")
        ###if st.button('Upload map', key=f'upload_map2_{tabname}'):
        ###    baseDataURL = "https://p3sc.oma.be/datarepfiles"
        ###    parts = upl_filename.split('_')
        ###    fileLevel = parts[2].upper()
        ###    if 'v03' in upl_filename:
        ###        file_version = 'v03'
        ###    else:
        ###        file_version = 'v2'
        ###    file_url = f"{baseDataURL}/{fileLevel}/{file_version}/{upl_filename}"
        ###    st.write(f'Link: {file_url}')
        ###    upl_map = sunpy.map.Map(file_url)
        ###    st.session_state.original_map = sunpy.map.Map(upl_map.data.astype('float'), upl_map.meta)
        ###    st.success(f"Loaded web file")
        ###    st.rerun()


        if wbf_file:
            wbf_map = file_to_smap(wbf_file)

            with st.container(border=True):

                st.markdown("**Add or Remove F-Corona**")
                
                action = st.radio(
                    "Action",
                    options=["Add", "Remove"],
                    horizontal=True
                )
                
                fcorona_choice = st.selectbox(
                    "F-Corona Model",
                    options=["Koutchmy, 2000", "Allen, 1977"]
                )
                fcorona_model = "standard" if fcorona_choice == "Koutchmy, 2000" else "Allen"

            if st.button('Calculate new WBF map', type='primary'):
                if action == "Remove":
                    newwbf_map = fcorona_removed_map(wbf_map, model=fcorona_model)
                elif action == "Add":
                    newwbf_map = fcorona_added_map(wbf_map, model=fcorona_model)
                
                st.success("Calculation completed!")
                st.pyplot(plot_map(newwbf_map))
                download_map_btn(newwbf_map)
        else:
            st.info('Please upload a WBF FITS file to start.')

