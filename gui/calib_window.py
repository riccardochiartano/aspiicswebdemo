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
from logic.utils import download_map_btn, download_all_maps_btn, get_filter, calibrate, aspiics_files_url
from logic.plot import plot_map
from logic.calibration import calibrate_rob

base_dir = Path(__file__).resolve().parent.parent

tabname = 'calibration'


def calib_window():

    if 'run_calib' not in st.session_state:
        st.session_state.run_calib = None
    if 'load_web_calib' not in st.session_state:
        st.session_state.load_web_calib = False
    if 'calib_files_name' not in st.session_state:
        st.session_state.calib_files_name = None
    
    col_left, col_main, col_right = st.columns([1, 3, 1])

    with col_main:
        st.subheader("Calibrate (l1 files)")

        st.divider()
        
        col1, col2 = st.columns([2, 3])

        with col1:
            st.write(f"**Map file:** ")

        with col2:
            uploaded_calib_file = st.file_uploader(
                f"Load maps",
                type=["fits"],
                key=f"calib_uploader",
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

        if uploaded_calib_file:
            paths = []
            names = []
            for uf in uploaded_calib_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                    tmp.write(uf.read())
                    paths.append(tmp.name)
                names.append(uf.name)

            st.session_state[f"calib_files_path"] = paths   # real path
            st.session_state[f"calib_files_name"] = names   # real name

        with col2:
            if st.button("Load from web", key='load_from_web_calib'):
                st.session_state.load_web_calib = not st.session_state.load_web_calib
                #st.write("work in progress..")
        if st.session_state.load_web_calib:
            show_webloader()
        
        st.divider()

        uploaded = [True]

        default_paths = {
            "bias_A": os.path.join(base_dir, 'resources', 'rob_calib_data', 'bias_A.fits'),
            "bias_B": os.path.join(base_dir, 'resources', 'rob_calib_data', 'bias_B.fits'),
            "dark_A2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_A2.fits'),
            "dark_B2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_B2.fits'),
            "dark_C2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_C2.fits'),
        }

        for key, path in default_paths.items():
            if f"calib_{key}_path" not in st.session_state:
                st.session_state[f"calib_{key}_path"] = path
                st.session_state[f"calib_{key}_name"] = os.path.basename(path)

        for key in default_paths.keys():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write(f"**{key}:** {st.session_state[f'calib_{key}_name']}")

            with col2:
                uploaded_file = st.file_uploader(f"Load new {key}", type=["fits"], key=f"calib_{key}_uploader", label_visibility="collapsed")
                
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                st.session_state[f"calib_{key}_path"] = tmp_path              # real path 
                st.session_state[f"calib_{key}_name"] = uploaded_file.name    # real name 
                uploaded.append(True)
            else:
                uploaded.append(False)

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
            st.session_state.calib_flat_path = default_flat["flatfield_P1"]

        col1, col2 = st.columns([1, 3])

        with col1:
            flat_choice = st.selectbox(
                "flatfield",
                ["flatfield_P1", "flatfield_P2", "flatfield_P3", "flatfield_WB", 'flatfield_Fe', 'flatfield_He', 'Other'],
                index=["flatfield_P1", "flatfield_P2", "flatfield_P3", "flatfield_WB", 'flatfield_Fe', 'flatfield_He', 'Other'].index(st.session_state.flat_choice),
                key='calib_flat'
            )

        with col2:
            if flat_choice == "Other":
                uploaded_flat = st.file_uploader("Load personal flatfield", type=["fits"], key="flat_uploader")
                if uploaded_flat is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                        tmp.write(uploaded_flat.read())
                        tmp_path = tmp.name
                    st.session_state.flat_choice = "Other"
                    st.session_state.calib_flat_path = tmp_path
                    st.session_state.calib_flat_name = uploaded_flat.name
                    st.success(f"Loaded file: {uploaded_flat.name}")
                    uploaded.append(True)                    
            else:
                st.session_state.flat_choice = flat_choice
                st.session_state.calib_flat_path = default_flat[flat_choice]
                st.session_state.calib_flat_name = flat_choice
                st.write(f"Path: `{st.session_state.calib_flat_path}`")
                uploaded.append(False)


        #st.info(f"**Using flatfield:** {os.path.basename(st.session_state.merge_flat_path)}")

        st.divider()

        calib_unit = st.radio("Measure unit:",
                ["MSB"],
                horizontal=True,
                key="calib_measure_unit"
            )

        st.divider()
        
        calib_mode = st.radio("Calibration procedure:",
                ["ROB"],
                horizontal=True,
                key="which_calib1"
            )

        st.divider()
        #st.write(uploaded)
        #st.divider()

        if st.button("Calibrate"):
            st.session_state.run_calib = True

        if st.session_state.run_calib:
            with st.expander("Files: ", expanded=True):
                order = [
                    "files", "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2", "flat"
                ]
                for name, flag in zip(order, uploaded):
                    key_path = f"calib_{name}_path"
                    key_name = f"calib_{name}_name"
                    if key_path in st.session_state:
                        # nome originale per file 
                        if name.startswith("files"):
                            st.write(f"{name}:")
                            for fname in st.session_state[key_name]:
                                st.markdown(f"&emsp;{fname}", unsafe_allow_html=True)
                        else:
                            if flag:
                                val = st.session_state[key_name]
                                st.write(f"{name}: {val}")
                            else:    
                                val = st.session_state[key_path]
                                st.write(f"{name}: {val}")

            if calib_mode == 'INAF':
                map = do_calib(calib_unit)
            if calib_mode == 'ROB':
                map_dict = do_calib_rob(calib_unit)
            
            map = next(iter(map_dict.values()))
            st.pyplot(plot_map(map))
            if len(map_dict.items()) > 1:
                download_all_maps_btn(map_dict, label="💾 Download calibrated maps (.zip)", zipname="calibrated.zip")
            else:
                download_map_btn(map)
            return
            


def do_calib(unit):
    return

    keys_order = [
        "files", "bias", "dark_A2", "dark_B2", "dark_C2", "flat"
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
    
def do_calib_rob(unit):

    keys_order = [
        "files", "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2", "flat"
    ]

    files = {}
    files_name = {}

    for key in keys_order:
        session_key = f"calib_{key}_path"
        session_name = f"calib_{key}_name"
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

    # calibrating
    #st.write(files)
    #st.write(files_name)
    map_dict = {}
    if len(files['files']) > 1:
        st.warning('More than one file uploaded, showing only the first image.')
        manyfiles = True
    else:
        manyfiles = False
    for i in range(len(files['files'])):
        calibrated_map = calibrate_rob(files['files'][i], files, files_name, unit, manyfiles=manyfiles)
        map_name = calibrated_map.meta['filename'].split('.')[0]
        map_dict[map_name] = calibrated_map
    
    #merged_map = merge_rob(files['files'])
    #filter = files['flat'].split('/')[-1].split('.')[0].split('_')[-1]
    #calibrated_map = calibrate(merged_map, filter, unit)

    return map_dict

def show_webloader():
    st.subheader('Upload from web')
    st.write('')
    
    level = st.radio("Level", ["L1"], horizontal=True, key=f'level_{tabname}')
    
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

        st.session_state[f"calib_files_path"] = selected_files
        st.session_state[f"calib_files_name"] = [file.split('/')[-1] for file in selected_files]    