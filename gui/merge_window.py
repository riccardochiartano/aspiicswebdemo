import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import tempfile
import json
import os
import datetime

from logic.merging import merge, merge_rob
from logic.utils import download_map_btn, get_filter, calibrate, aspiics_files_url
from logic.plot import plot_map

base_dir = Path(__file__).resolve().parent.parent

tabname = 'merge'

def merge_window():

    if 'run_merge' not in st.session_state:
        st.session_state.run_merge = None
    if 'load_web_merge' not in st. session_state:
        st.session_state.load_web_merge = False
    
    col_left, col_main, col_right = st.columns([1, 3, 1])

    with col_main:
        st.subheader("Merging")
        st.write('Use l1 files for the INAF procedure, l2 files for the ROB one.')
        
        st.divider()
    
        merge_mode = st.radio("Merging procedure:",
            ["INAF", "ROB (L2 files)"],
            horizontal=True,
            key="which_merge"
        )

        st.divider()
        
        col1, col2 = st.columns([2, 3])

        with col1:
            st.write(f"**Map files:** ")

        with col2:
            uploaded_merge_files = st.file_uploader(
                f"Load maps",
                type=["fits"],
                key=f"merge_uploader",
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

        if uploaded_merge_files:
            paths = []
            names = []
            for uf in uploaded_merge_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                    tmp.write(uf.read())
                    paths.append(tmp.name)
                names.append(uf.name)

            st.session_state[f"merge_files_path"] = paths   # real path
            st.session_state[f"merge_files_name"] = names   # real name

        with col2:
            if st.button("Load from web", key=f'load_from_web_{tabname}'):
                st.session_state.load_web_merge = not st.session_state.load_web_merge
                #st.write("work in progress..")
        if st.session_state.load_web_merge:
            show_webloader()
        

        st.divider()

        if merge_mode == 'INAF':
            default_paths = {
                "bias_A": os.path.join(base_dir, 'resources', 'rob_calib_data', 'bias_A.fits'),
                "bias_B": os.path.join(base_dir, 'resources', 'rob_calib_data', 'bias_B.fits'),
                "dark_A2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_A2.fits'),
                "dark_B2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_B2.fits'),
                "dark_C2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_C2.fits'),
            }

            for key, path in default_paths.items():
                if f"merge_{key}_path" not in st.session_state:
                    st.session_state[f"merge_{key}_path"] = path
                    st.session_state[f"merge_{key}_name"] = os.path.basename(path)

            for key in default_paths.keys():
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.write(f"**{key}:** {st.session_state[f'merge_{key}_name']}")

                with col2:
                    uploaded_file = st.file_uploader(f"Load new {key}", type=["fits"], key=f"merge_{key}_uploader", label_visibility="collapsed")
                    
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    st.session_state[f"merge_{key}_path"] = tmp_path              # real path 
                    st.session_state[f"merge_{key}_name"] = uploaded_file.name    # real name 

            st.divider()
            
            default_flat = {
                "flatfield_P1": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_P1.fits'),
                "flatfield_P2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_P2.fits'),
                "flatfield_P3": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_P3.fits'),
                "flatfield_WB": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_WB.fits'),
                "flatfield_Fe": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_Fe.fits'),
                "flatfield_He": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_He.fits'),
            }

            if "flat_choice" not in st.session_state:
                st.session_state.flat_choice = "flatfield_P1"
                st.session_state.merge_flat_path = default_flat["flatfield_P1"]

            col1, col2 = st.columns([1, 3])

            with col1:
                flat_choice = st.selectbox(
                    "flatfield",
                    ["flatfield_P1", "flatfield_P2", "flatfield_P3", "flatfield_WB", 'flatfield_Fe', 'flatfield_He', 'Other'],
                    index=["flatfield_P1", "flatfield_P2", "flatfield_P3", "flatfield_WB", 'flatfield_Fe', 'flatfield_He', 'Other'].index(st.session_state.flat_choice)
                )

            with col2:
                if flat_choice == "Other":
                    uploaded_flat = st.file_uploader("Load personal flatfield", type=["fits"], key="flat_uploader")
                    if uploaded_flat is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                            tmp.write(uploaded_flat.read())
                            tmp_path = tmp.name
                        st.session_state.flat_choice = "Altro"
                        st.session_state.merge_flat_path = tmp_path
                        st.success(f"Loaded Matrix: {uploaded_flat.name}")
                else:
                    st.session_state.flat_choice = flat_choice
                    st.session_state.merge_flat_path = default_flat[flat_choice]
                    st.write(f"Path: `{st.session_state.merge_flat_path}`")

            #st.info(f"**Using flatfield:** {os.path.basename(st.session_state.merge_flat_path)}")

            st.divider()

            merge_calib_unit = st.radio("Measure unit:",
                    ["DN/s", "ph/(s cm^2 sr)", "MSB"],
                    horizontal=True,
                    key="merge_measure_unit"
                )
            
            st.divider()
        
            if st.button("Merge (and Calibrate)"):
                st.session_state.run_merge = True

            if st.session_state.run_merge:
                with st.expander("Files: ", expanded=True):
                    order = [
                        "files", "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2", "flat"
                    ]
                    for name in order:
                        key_path = f"merge_{name}_path"
                        if key_path in st.session_state:
                            # nome originale per file angles
                            if name.startswith("files"):
                                key_name = f"merge_{name}_name"
                                st.write(f"{name}:")
                                for fname in st.session_state[key_name]:
                                    st.markdown(f"&emsp;{fname}", unsafe_allow_html=True)
                            else:
                                val = st.session_state[key_path]
                                st.write(f"{name}: {val}")

                map = do_merge(merge_calib_unit)
                
                st.pyplot(plot_map(map))
                download_map_btn(map)
            return


        else:                                
            if st.button("Merge"):
                st.session_state.run_merge = True

            if st.session_state.run_merge:
                with st.expander("Files: ", expanded=True):
                    order = [
                        "files"
                    ]
                    for name in order:
                        key_path = f"merge_{name}_path"
                        if key_path in st.session_state:
                            # nome originale per file angles
                            if name.startswith("files"):
                                key_name = f"merge_{name}_name"
                                st.write(f"{name}:")
                                for fname in st.session_state[key_name]:
                                    st.markdown(f"&emsp;{fname}", unsafe_allow_html=True)
                            else:
                                val = st.session_state[key_path]
                                st.write(f"{name}: {val}")

                map = do_merge_rob()
            
                st.pyplot(plot_map(map))
                download_map_btn(map)
            return


def do_merge(unit):

    keys_order = [
        "files", "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2", "flat"
    ]

    files = {}

    for key in keys_order:
        session_key = f"merge_{key}_path"
        if session_key in st.session_state:
            val = st.session_state[session_key]
            # per lista files
            if isinstance(val, list):
                files[key] = val
            else:
                files[key] = val  

    # merging
    merged_map = merge(files)
    filter = files['flat'].split('/')[-1].split('.')[0].split('_')[-1]
    calibrated_map = calibrate(merged_map, filter, unit)

    return calibrated_map
    
def do_merge_rob(unit='MSB'):

    keys_order = [
        "files", "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2", "flat"
    ]

    files = {}
    files_name = {}

    for key in keys_order:
        session_key = f"merge_{key}_path"
        session_name = f"merge_{key}_name"
        if session_key in st.session_state:
            val = st.session_state[session_key]
            # per lista files
            if isinstance(val, list):
                files[key] = val
            else:
                files[key] = val  
        if session_name in st.session_state:
            val = st.session_state[session_name]
            # per lista files name
            if isinstance(val, list):
                files_name[key] = val
            else:
                files_name[key] = val  

    # merging
    merged_map = merge_rob(files['files'])
    #filter = files['flat'].split('/')[-1].split('.')[0].split('_')[-1]
    #calibrated_map = calibrate(merged_map, filter, unit)

    return merged_map

def show_webloader():
    st.subheader('Upload from web')
    st.write('')
    
    level = st.radio("Level", ["L1", "L2"], horizontal=True, key=f'level_{tabname}')
    
    col, col1, col2, col3 = st.columns([1,1,1,1])
    with col:
        st.write('Filter')
    with col1:
        flt_wb = st.checkbox('Wideband', key=f'wb_{tabname}')    
        flt_p2 = st.checkbox('Polarizer 0°', key=f'p2_{tabname}')    
    with col2:
        flt_fe = st.checkbox('Fe XIV', key=f'fe_{tabname}')
        flt_p1 = st.checkbox('Polarizer 60°', key=f'p1_{tabname}')    
    with col3:
        flt_he = st.checkbox('He I D3', key=f'he_{tabname}')    
        flt_p3 = st.checkbox('Polarizer 120°', key=f'p3_{tabname}')    

    checkboxes = {
        "wb": flt_wb,
        "fe": flt_fe,
        "he": flt_he,
        "p1": flt_p1,
        "p2": flt_p2,
        "p3": flt_p3
    }

    cycle_id = st.text_input("Cycle ID (8 digits)", "1560607E", key=f'cid_{tabname}')
    #seq_num = st.text_input("Sequence number", "")
    #acq_num = st.text_input("Acquisition number", "")
    #exp_num = st.text_input("Exposure number", "")

    col1, col2 = st.columns([1,1])
    with col1:
        date_start = st.date_input("Start date", value=datetime.date(2025, 5, 1), format="DD.MM.YYYY", key=f'date_start_{tabname}')
        date_end = st.date_input("End date", value=datetime.date(2026, 9, 1), format="DD.MM.YYYY", key=f'date_end_{tabname}')
        #date_start = st.date_input("Start date", value=None, format="DD.MM.YYYY")
        #date_end = st.date_input("End date", value=None, format="DD.MM.YYYY")
    with col2:    
        #time_start = st.time_input("Time", value=None, key='time_start', step=60)
        #time_end = st.time_input("Time", value=None, key='time_end', step=60)
        time_start = st.time_input("Time", datetime.time(0,0), step=600, key=f'time_start_{tabname}')
        time_end = st.time_input("Time", datetime.time(23,59), step=600, key=f'time_end_{tabname}')
    
    if st.button("Show files", key=f'shw_files_{tabname}'):
    #    st.session_state.show_files = not st.session_state.show_files
    #
    #if st.session_state.show_files:
        flt_list = [key for key, checked in checkboxes.items() if checked]

        dt_start = datetime.datetime.combine(date_start, time_start)
        dt_end = datetime.datetime.combine(date_end, time_end)
            
        files_url = aspiics_files_url(flt_list, level, cycle_id, dt_start, dt_end)
        if files_url == []:
            st.error('No files with these keys...')
        st.session_state.files_url = files_url
        
    if st.session_state.files_url:
        st.write('Files found:')
        selected_files = []
        for f in st.session_state.files_url:
            filename = f.split('/')[-1]
            if st.checkbox(filename, key=f'{filename}_{tabname}'):
                selected_files.append(f)

        st.write("Selected files:", selected_files)
        st.session_state.selected_files = selected_files

        st.session_state[f"merge_files_path"] = selected_files
        st.session_state[f"merge_files_name"] = [file.split('/')[-1] for file in selected_files]    