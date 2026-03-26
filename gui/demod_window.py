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
import datetime

from logic.demodulation import demodulate_merged, demodulate_rob, save_all
from logic.merging import merge_demodulation, merge
from logic.utils import download_all_maps_btn, aspiics_files_url
from logic.calibration import calibrate_rob

base_dir = Path(__file__).resolve().parent.parent

tabname = 'demodulation'

def demodulation_window():

    if 'run_demodulation' not in st.session_state:
        st.session_state.run_demodulation = None
    if 'load_web_demod' not in st. session_state:
        st.session_state.load_web_demod = False

    col_left, col_main, col_right = st.columns([1, 3, 1])

    with col_main:
        st.subheader("Demodulation")
        st.write('Upload l1 files. (l2 and l3 files work only for ROB pipeline)')

        st.divider()

        cal_mode = st.radio("Demodulation procedure:",
            ["INAF", "ROB"],
            horizontal=True,
            key="which_calib"
        )

        st.divider()
        
        # rimetti il true quando hai finito prove (per avere file demod di default)
        if False:
            angles = ["0°", "60°", "120°"]
            p_files = ['p2', 'p1', 'p3']

            for angle, p in zip(angles, p_files):
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.write(f"**File {angle} ({p}):** ")

                with col2:
                    uploaded_files = st.file_uploader(
                        f"Carica file {angle}",
                        type=["fits"],
                        key=f"uploader_{angle}",
                        accept_multiple_files=True,
                        label_visibility="collapsed"
                    )

                if uploaded_files:
                    paths = []
                    names = []
                    for uf in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                            tmp.write(uf.read())
                            paths.append(tmp.name)
                        names.append(uf.name)

                    st.session_state[f"files_{angle}_path"] = paths   # percorso reale
                    st.session_state[f"files_{angle}_name"] = names   # nome originale
        else:
            angles = ["0°", "60°", "120°"]
            p_files = ['p2', 'p1', 'p3']

            default_files = {
                "0°": ["/home/bob/Scuola/Dotts/aspiics/for_Silvano/aspiics_p2_l1_0CF1C005000212_20250523181806.fits"],
                "60°": ["/home/bob/Scuola/Dotts/aspiics/for_Silvano/aspiics_p1_l1_0CF1C006000112_20250523182220.fits"],
                "120°": ["/home/bob/Scuola/Dotts/aspiics/for_Silvano/aspiics_p3_l1_0CF1C004000212_20250523180951.fits"]
            }

            default_names = {
                "0°": ["aspiics_p2_l1_0CF1C005000212_20250523181806.fits"],
                "60°": ["aspiics_p1_l1_0CF1C006000112_20250523182220.fits"],
                "120°": ["aspiics_p3_l1_0CF1C004000212_20250523180951.fits"]
            }

            for angle, p in zip(angles, p_files):
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.write(f"**File {angle} ({p}):** ")

                # Se non esistono ancora nello session_state, inizializzo i default
                if f"files_{angle}_path" not in st.session_state:
                    st.session_state[f"files_{angle}_path"] = default_files[angle]
                    st.session_state[f"files_{angle}_name"] = default_names[angle]

                with col2:
                    uploaded_files = st.file_uploader(
                        f"Carica file {angle}",
                        type=["fits"],
                        key=f"uploader_{angle}",
                        accept_multiple_files=True,
                        label_visibility="collapsed"
                    )

                # Se l’utente carica nuovi file, li sovrascrive
                if uploaded_files:
                    paths = []
                    names = []
                    for uf in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                            tmp.write(uf.read())
                            paths.append(tmp.name)
                        names.append(uf.name)

                    st.session_state[f"files_{angle}_path"] = paths
                    st.session_state[f"files_{angle}_name"] = names

        with col2:
            if st.button("Load from web", key=f'load_from_web_{tabname}'):
                st.session_state.load_web_demod = not st.session_state.load_web_demod
                #st.write("work in progress..")
        if st.session_state.load_web_demod:
            show_webloader()
        
        st.divider()

        default_paths = {
            "bias_A": os.path.join(base_dir, 'resources', 'rob_calib_data', 'bias_A.fits'),
            "bias_B": os.path.join(base_dir, 'resources', 'rob_calib_data', 'bias_B.fits'),
            "dark_A2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_A2.fits'),
            "dark_B2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_B2.fits'),
            "dark_C2": os.path.join(base_dir, 'resources', 'rob_calib_data', 'dark_C2.fits'),
            #"ak 0°": os.path.join(base_dir, 'resources', 'ak_images', 'aspiics_ak000_inaf_1.0.fits'),
            #"ak 60°": os.path.join(base_dir, 'resources', 'ak_images', 'aspiics_ak060_inaf_1.0.fits'),
            #"ak 120°": os.path.join(base_dir, 'resources', 'ak_images', 'aspiics_ak120_inaf_1.0.fits')
            "ak 0°": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_P1.fits'),
            "ak 60°": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_P2.fits'),
            "ak 120°": os.path.join(base_dir, 'resources', 'rob_calib_data', 'flatfield_P3.fits')
        }

        for key, path in default_paths.items():
            if f"{key}_path" not in st.session_state:
                st.session_state[f"{key}_path"] = path
                st.session_state[f"{key}_name"] = os.path.basename(path)

        for key in default_paths.keys():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write(f"**{key}:** {st.session_state[f'{key}_name']}")

            with col2:
                uploaded_file = st.file_uploader(f"Carica nuovo {key}", type=["fits"], key=f"{key}_uploader", label_visibility="collapsed")
                
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                st.session_state[f"{key}_path"] = tmp_path              # percorso reale 
                st.session_state[f"{key}_name"] = uploaded_file.name    # nome originale 

        st.divider()

        if cal_mode == 'INAF':

            default_matrices = {
                "rob": os.path.join(base_dir, 'resources', 'rob_calib_data', 'demod_matrix.fits'),
                "finale": os.path.join(base_dir, 'resources', 'demod_matrix_final', 'demod_matrix_finale2.0.fits'),
                "two_steps": os.path.join(base_dir, 'resources', 'demod_matrix_final', 'demod_matrix_2step.fits'),
                "embedded": os.path.join(base_dir, 'resources', 'demod_matrix_final', 'demod_matrix_embedded.sav')
            }

            if "matrix_choice" not in st.session_state:
                st.session_state.matrix_choice = "rob"
                st.session_state.matrix_path = default_matrices["rob"]

            col1, col2 = st.columns([1, 3])

            with col1:
                matrix_choice = st.selectbox(
                    "Matrix",
                    ["rob", "finale", "two_steps", "embedded", "Other"],
                    index=["rob", "finale", "two_steps", "embedded", "Other"].index(st.session_state.matrix_choice)
                )

            with col2:
                if matrix_choice == "Other":
                    uploaded_matrix = st.file_uploader("Load personal matrix", type=["fits"], key="matrix_uploader")
                    if uploaded_matrix is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                            tmp.write(uploaded_matrix.read())
                            tmp_path = tmp.name
                        st.session_state.matrix_choice = "Altro"
                        st.session_state.matrix_path = tmp_path
                        st.success(f"Loaded Matrix: {uploaded_matrix.name}")
                else:
                    st.session_state.matrix_choice = matrix_choice
                    st.session_state.matrix_path = default_matrices[matrix_choice]
                    st.write(f"Path: `{st.session_state.matrix_path}`")

            st.info(f"**Using matrix:** {os.path.basename(st.session_state.matrix_path)}")

        elif cal_mode == 'ROB':
            default_paths = {
                "pol 0°": os.path.join(base_dir, 'resources', 'rob_calib_data', 'aspiics_p2_angle.fits'),
                "pol 60°": os.path.join(base_dir, 'resources', 'rob_calib_data', 'aspiics_p1_angle.fits'),
                "pol 120°": os.path.join(base_dir, 'resources', 'rob_calib_data', 'aspiics_p3_angle.fits')
            }

            for key, path in default_paths.items():
                if f"{key}_path" not in st.session_state:
                    st.session_state[f"{key}_path"] = path
                    st.session_state[f"{key}_name"] = os.path.basename(path)

            for key in default_paths.keys():
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.write(f"**{key}:** {st.session_state[f'{key}_name']}")

                with col2:
                    uploaded_file = st.file_uploader(f"Carica nuovo {key}", type=["fits"], key=f"{key}_uploader", label_visibility="collapsed")
                    
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    st.session_state[f"{key}_path"] = tmp_path              # percorso reale 
                    st.session_state[f"{key}_name"] = uploaded_file.name    # nome originale 

        st.divider()

        calib_unit = st.radio("Measure unit:",
                ["DN/s", "ph/(s cm^2 sr)", "MSB"],
                horizontal=True,
                key="demod_measure_unit"
            )

        st.divider()

        col1, col2 = st.columns([3, 1])

        with col1:   
            if st.button("Run Demodulation"):
                st.session_state.run_demodulation = True

        with col2:
            if st.button("Reset Demodulation"):
                if "demod_maps" in st.session_state:
                    st.session_state["demod_maps"] = {}
                st.session_state.run_demodulation = False



        if st.session_state.get("run_demodulation", False) and st.session_state.get('demod_maps') == {}:
            with st.spinner("Running demodulation..."):
                if st.session_state.load_web_demod:
                    st.warning('The FITS files are being downloaded. Demodulation may take longer.')
                if cal_mode == 'INAF':
                    maps = demodulation(calib_unit)
                if cal_mode == 'ROB':
                    maps = demodulation_rob(calib_unit)
                st.session_state["demod_maps"] = maps

                if st.session_state.get("current_map") is None and maps:
                    st.session_state.current_map = maps['I']

            st.success('Demodulation finished, go to "Map" tab to see the results.')

        if "demod_maps" in st.session_state:
            with st.expander("Files:", expanded=True):
                order = [
                    "files_0°", "files_60°", "files_120°",
                    "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2",
                    "ak 0°", "ak 60°", "ak 120°", "matrix"
                ]
                for name in order:
                    key_path = f"{name}_path"
                    if key_path in st.session_state:
                        if name.startswith("files_"):
                            key_name = f"{name}_name"
                            st.write(f"{name}:")
                            for fname in st.session_state[key_name]:
                                st.markdown(f"&emsp;{fname}", unsafe_allow_html=True)
                        else:
                            val = st.session_state[key_path]
                            st.write(f"{name}: {val}")

            download_all_maps_btn(st.session_state["demod_maps"])




def demodulation(unit):

    keys_order = [
        "files_0°", "files_60°", "files_120°",
        "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2",
        "ak 0°", "ak 60°", "ak 120°"
    ]

    files_0 = {}
    files_60 = {}
    files_120 = {}

    rename_map = {
        "files_0°":   ("files", files_0),
        "ak 0°":      ("flat",  files_0),
        "files_60°":  ("files", files_60),
        "ak 60°":     ("flat",  files_60),
        "files_120°": ("files", files_120),
        "ak 120°":    ("flat",  files_120),
    }   

    for key in keys_order:
        session_key = f"{key}_path"

        if session_key in st.session_state:
            val = st.session_state[session_key]

            if key in rename_map:
                new_name, group = rename_map[key]
                group[new_name] = val

            else:
                files_0[key] = val
                files_60[key] = val
                files_120[key] = val

    files = [files_0, files_60, files_120]

    # merging
    #st.write(files_0['files'])
    #merged_maps = merge_demodulation(files, unit)
    merged_maps = []
    for base_files in files:
        merged_maps.append(merge(base_files, unit))

    matrix_path = st.session_state.matrix_path

    maps = demodulate_merged(merged_maps, matrix_path)

    return maps

def demodulation_rob(unit):

    keys_order = [
        "files_0°", "files_60°", "files_120°",
        "bias_A", "bias_B", "dark_A2", "dark_B2", "dark_C2",
        "ak 0°", "ak 60°", "ak 120°"
    ]

    files_0 = {}
    files_60 = {}
    files_120 = {}
    files_0_name = {}
    files_60_name = {}
    files_120_name = {}

    rename_map = {
        "files_0°":   ("files", files_0),
        "ak 0°":      ("flat",  files_0),
        "files_60°":  ("files", files_60),
        "ak 60°":     ("flat",  files_60),
        "files_120°": ("files", files_120),
        "ak 120°":    ("flat",  files_120),
    }   
    rename_map_name = {
        "files_0°":   ("files", files_0_name),
        "ak 0°":      ("flat",  files_0_name),
        "files_60°":  ("files", files_60_name),
        "ak 60°":     ("flat",  files_60_name),
        "files_120°": ("files", files_120_name),
        "ak 120°":    ("flat",  files_120_name),
    }

    for key in keys_order:
        session_key = f"{key}_path"
        session_name = f"{key}_name"
        if session_key in st.session_state:
            val = st.session_state[session_key]
            if key in rename_map:
                new_name, group = rename_map[key]
                group[new_name] = val
            else:
                files_0[key] = val
                files_60[key] = val
                files_120[key] = val
        if session_name in st.session_state:
            val = st.session_state[session_name]
            if key in rename_map:
                new_name, group = rename_map_name[key]
                group[new_name] = val
            else:
                files_0_name[key] = val
                files_60_name[key] = val
                files_120_name[key] = val

    # calibrate maps
    if '_l1_' in files_0_name["files"][0]:
        #st.write('l1 files')
        calibrated_maps = [calibrate_rob(files_0["files"][0], files_0, files_0_name, unit), 
                           calibrate_rob(files_60["files"][0], files_60, files_60_name, unit), 
                           calibrate_rob(files_120["files"][0], files_120, files_120_name, unit)]
    #elif '_l2_' in files_0_name["files"][0]:
    else:
        #st.write('l2 files')
        calibrated_maps = [sunpy.map.Map(files_0["files"][0]),
                           sunpy.map.Map(files_60["files"][0]),
                           sunpy.map.Map(files_120["files"][0])]
        
    # merging
    #merged_maps = merge_demodulation(files, unit)
    polar_files = [st.session_state["pol 0°_path"], 
                   st.session_state["pol 60°_path"], 
                   st.session_state["pol 120°_path"]]

    maps = demodulate_rob(calibrated_maps, polar_files)

    return maps

def show_webloader():
    st.subheader('Upload from web')
    st.write('')

    #level = st.radio("Level", ["L1", "L2"], horizontal=True, key=f'level_{tabname}')
    level = st.radio("Level", ["L1"], horizontal=True, key=f'level_{tabname}')

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
        flt_list = ['p1', 'p2', 'p3']

        dt_start = datetime.datetime.combine(date_start, time_start)
        dt_end = datetime.datetime.combine(date_end, time_end)
            
        files_url = aspiics_files_url(flt_list, level, cycle_id, dt_start, dt_end)
        if files_url == []:
            st.error('No files with these keys...')
        st.session_state.files_url = files_url
        
    if st.session_state.files_url:
        st.write('Files found:')
        selected_files_0 = []
        selected_files_60 = []
        selected_files_120 = []
        for f in st.session_state.files_url:
            filename = f.split('/')[-1]
            checked = st.checkbox(filename, key=f'{filename}_{tabname}')
            if checked:
                if "p2" in filename:
                    selected_files_0.append(f)
                elif "p1" in filename:
                    selected_files_60.append(f)
                elif "p3" in filename:
                    selected_files_120.append(f)

        st.write("Selected files:", selected_files_0)
        st.write("               ", selected_files_60)
        st.write("               ", selected_files_120)
        
        st.session_state[f"files_0°_path"] = selected_files_0
        st.session_state[f"files_0°_name"] = [file.split('/')[-1] for file in selected_files_0]
        st.session_state[f"files_60°_path"] = selected_files_60
        st.session_state[f"files_60°_name"] = [file.split('/')[-1] for file in selected_files_60]
        st.session_state[f"files_120°_path"] = selected_files_120
        st.session_state[f"files_120°_name"] = [file.split('/')[-1] for file in selected_files_120]

