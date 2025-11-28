import numpy as np
import sunpy.map
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import streamlit as st
from io import BytesIO
import tempfile
import os
import zipfile
import json
import io
import pandas as pd
import re
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime

from logic.plot import plot_hist_AoLP, plot_local_RF_AoLP

base_dir = Path(__file__).resolve().parent.parent


def sun_center (solar_map):
    sun_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=solar_map.coordinate_frame)
    sun_x, sun_y = solar_map.wcs.world_to_pixel(sun_center)
    return (int(sun_x), int(sun_y))

def define_mask(image, center_y, center_x, r_in, r_out):
    y, x = np.indices(image.shape)
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist >= r_in) & (dist <= r_out) & ~np.isnan(image)
    return mask

def remove_occulter(solar_map):
    x, y = sun_center(solar_map)
    image = solar_map.data
    mask = define_mask(image, y, x, r_in=0, r_out=450)
    image[mask] = np.nan
    solar_map_no_occ = sunpy.map.Map(image, solar_map.meta)
    return solar_map_no_occ

def download_map_btn(map, label = "💾 Download map FITS", file_name = None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
        temp_path = tmp.name
        map.save(temp_path, overwrite=True)

    if not(file_name):
        #file_name = 'current_map.fits'
        file_name = map.meta.get('filename', 'map.fits') 

    with open(temp_path, "rb") as f:
        st.download_button(
            label=label,
            data=f,
            file_name=file_name,
            mime="application/fits"
        )

def download_all_maps_btn(maps_dict, label='💾 Download demodulated maps (.zip)', zipname='demodulated_maps.zip'):
    if not maps_dict:
        st.warning("No map to download.")
        return

    if "zip_ready" not in st.session_state:
        st.session_state.zip_ready = False

    if st.button("📦 Create ZIP archive"):
        with st.spinner("Creating the ZIP file..."):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for name, m in maps_dict.items():
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
                        temp_path = tmp.name
                        m.save(temp_path, overwrite=True)
                    zipf.write(temp_path, f"{name}.fits")
                    os.remove(temp_path)
            zip_buffer.seek(0)
            st.session_state.zip_data = zip_buffer.getvalue()
            st.session_state.zip_ready = True

    if st.session_state.zip_ready:
        st.download_button(
            label=label,
            data=st.session_state.zip_data,
            file_name=zipname,
            mime="application/zip"
        )

def download_web_files_btn(selected_files, label='Download files (.zip)'):
    if not selected_files:
        st.warning("No files selected.")
        return

    if "zip_ready" not in st.session_state:
        st.session_state.zip_ready = False

    if st.button("Create ZIP archive", key='create_zip_btn'):
        zip_buffer = BytesIO()
        n = len(selected_files)
        progress = st.progress(0, text='Creating the ZIP file...')
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, url in enumerate(selected_files):
                r = requests.get(url)
                r.raise_for_status()
                filename = url.split("/")[-1]
                zipf.writestr(filename, r.content)
                progress.progress(int((i + 1) / n * 100), text='Creating the ZIP file...')
        zip_buffer.seek(0)
        st.session_state.zip_data = zip_buffer.getvalue()
        st.session_state.zip_ready = True

    if st.session_state.zip_ready:
        st.download_button(
            label=label,
            data=st.session_state.zip_data,
            file_name="aspiics_files.zip",
            mime="application/zip"
        )

def download_profile(type, prf_unit, profiles):
    df = pd.DataFrame()

    if type == 'rad':
        for angle, data in profiles.items():
            radii = data["radii"]
            values = data["values"]
            df["radii[Rsun]"] = radii
            df[f"profile_{angle:.0f}[{prf_unit}]"] = values
    if type == 'pol':
        for dist, data in profiles.items():
            angles = data["angles"]
            values = data["values"]
            df["angle[deg]"] = angles
            df[f"profile_{dist:.1f}[{prf_unit}]"] = values

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    csv_data = buffer.getvalue()

    st.download_button(
        label="Download profiles",
        data=csv_data,
        file_name=f"profiles_{type}.csv",
        mime="text/csv",
        key=type
    )

def deviation_from_RF(map_psi, nbins):
    x, y = np.meshgrid(
        np.arange(map_psi.data.shape[1]),
        np.arange(map_psi.data.shape[0])
    )
    sun_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=map_psi.coordinate_frame)
    sun_x, sun_y = map_psi.wcs.world_to_pixel(sun_center)

    x_arcsec = (x - sun_x) * map_psi.scale.axis1.value
    y_arcsec = (y - sun_y) * map_psi.scale.axis2.value

    # dist da centro solare in Rsun
    sun_r = map_psi.rsun_obs.to('arcsec').value
    distance_from_center = np.sqrt(x_arcsec**2 + y_arcsec**2) / sun_r 
    mask = (distance_from_center > 1.3) & (distance_from_center < 2.0)

    height, width = map_psi.data.shape[0], map_psi.data.shape[1]
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    rad_angle = np.arctan2(y - sun_y, x - sun_x)
    rad_angle = rad_angle % np.pi
    tan_angle = rad_angle - 0.5*np.pi

    image_psi_rf = np.rad2deg(map_psi.data - tan_angle)
    image_psi_rf[image_psi_rf>90] = image_psi_rf[image_psi_rf>90] - 180
    image_psi_rf[image_psi_rf<-90] = image_psi_rf[image_psi_rf<-90] + 180

    image_psi_rf_masked = image_psi_rf[mask]
    sat_mask = ~np.isnan(image_psi_rf_masked)
    image_psi_rf_no_nan = image_psi_rf_masked[sat_mask]

    # get map title
    im0 = map_psi.plot()       
    ax0 = plt.gca()        
    title = ax0.get_title()
    
    fig_RF_image = plot_local_RF_AoLP(image_psi_rf, title)
    fig_hist = plot_hist_AoLP(image_psi_rf_no_nan, nbins)
    st.pyplot(fig_RF_image)
    st.pyplot(fig_hist)

def file_to_smap(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if 'metis' not in file.name:
        solar_map = sunpy.map.Map(tmp_path)
    else:
        solar_map = sunpy.map.Map(tmp_path)[0]
        
    solar_image = solar_map.data
    if not np.issubdtype(solar_image.dtype, np.floating):
        solar_image = solar_image.astype(float)
    solar_map = sunpy.map.Map(solar_image, solar_map.meta)

    return solar_map

def radial_profile(solar_map, angle, start_rsun, end_rsun, ampl=np.deg2rad(1), nradii=50):
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs.to(u.arcsec)
    sun_x, sun_y = sun_center(solar_map)

    y, x = np.indices(solar_map.data.shape)
    dx = (x - sun_x)
    dy = (y - sun_y)
    r = np.sqrt(dx**2 + dy**2) * scale * u.arcsec
    angles_grid = np.mod(np.arctan2(dy, dx), 2*np.pi)
    
    angle_start, angle_end = angle - ampl/2, angle + ampl*2
    angle_mask = (angles_grid >= angle_start) & (angles_grid <= angle_end)

    radii = np.linspace(start_rsun, end_rsun, nradii+1)
    radii_avg = 0.5 * (radii[:-1] + radii[1:])
    radii_arcsec = radii * rsun_arcsec

    mask = np.isfinite(solar_map.data) & angle_mask
    r_vals = r[mask].value
    data_vals = solar_map.data[mask]

    rad_profile, _ = np.histogram(
        r_vals,
        bins=radii_arcsec.value,
        weights=data_vals
    )
    counts, _ = np.histogram(r_vals, bins=radii_arcsec.value)
    rad_profile = np.divide(rad_profile, counts, out=np.zeros_like(rad_profile), where=counts>0)

    # mean std dev error
    sq_profile, _ = np.histogram(
        r_vals,
        bins=radii_arcsec.value,
        weights=data_vals**2
    )
    variance = np.divide(sq_profile, counts, out=np.zeros_like(rad_profile), where=counts>0) - rad_profile**2
    variance[counts <= 1] = np.nan  # evita divisione per 0 o 1
    rad_error = np.sqrt(variance / counts)  # SEM


    return rad_profile, radii_avg, rad_error

def polar_profile(solar_map, dist, first_angle, last_angle, n_angles=360, step_rsun=0.1):
    scale = solar_map.scale[0].to(u.arcsec/u.pixel).value
    rsun_arcsec = solar_map.rsun_obs.to(u.arcsec)
    sun_x, sun_y = sun_center(solar_map)
    
    y, x = np.indices(solar_map.data.shape)
    dx = (x - sun_x)
    dy = (y - sun_y)
    r = np.sqrt(dx**2 + dy**2) * scale * u.arcsec
    angles_grid = np.mod(np.arctan2(dy, dx), 2*np.pi)

    r_in_arcsec = dist * rsun_arcsec
    r_out_arcsec = (dist + step_rsun) * rsun_arcsec
    radial_mask = (r >= r_in_arcsec) & (r <= r_out_arcsec) & np.isfinite(solar_map.data)

    angle_edges = np.linspace(np.deg2rad(first_angle), np.deg2rad(last_angle), n_angles+1)
    angle_indices = np.digitize(angles_grid[radial_mask], angle_edges) - 1
    angle_indices = np.clip(angle_indices, 0, n_angles - 1)

    data_vals = solar_map.data[radial_mask]
    pol_profile_sum = np.bincount(angle_indices, weights=data_vals, minlength=n_angles)
    counts = np.bincount(angle_indices, minlength=n_angles)
    pol_profile = pol_profile_sum / counts
    pol_profile[counts == 0] = np.nan

    angles_medi = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    angles_deg = np.rad2deg(angles_medi)

    return pol_profile, angles_deg


def calibrate(map, filter, unit):
    filtername = get_filter(filter)
    calibrate_file = os.path.join(base_dir, 'resources', 'rob_calib_data', 'calibr_data.json.real')
    with open(calibrate_file, 'r') as file:
        calib_dict = json.load(file)
    Aphot = calib_dict['calib_data'][filtername]['Aphot']
    MSB_value = calib_dict['calib_data'][filtername]['MSB']
    map.meta['bunit'] = unit
    if unit == 'DN/s':
        return map
    elif unit == 'MSB':
        cal_map = map / Aphot
        cal_map.meta['bunit'] = 'MSB'
        return cal_map
    elif unit == 'ph/(s cm^2 sr)':
        return map * MSB_value / Aphot

def get_filter(filter):
    if filter == 'P1' or filter == '0°':
        return 'Polarizer 0'
    if filter == 'P2' or filter == '60°':
        return 'Polarizer 60'
    if filter == 'P3' or filter == '120°':
        return 'Polarizer 120'
    if filter == 'WB':
        return 'Wideband'
    if filter == 'Fe':
        return 'Fe XIV'
    if filter == 'He':
        return 'He I'

def aspiics_files_url(filter_list, level, cycle_id, start_dt, end_dt):
    """
    Search ASPIICS repository to find filenames and return the urls of the images. 
    Supports multiple filters separated by commas.
    If start_date and end_date are given, it filters the time.

    Args:
        filter: ASPIICS filter (also a list, ex: 'p1,p2')
        level: Calibration level of the files
        cycle_id: ID of the observation cycle
        start_date: date filter start. format: "YYYYMMDD'T'HHMMSS"
        end_date: date filter end. format: "YYYYMMDD'T'HHMMSS"

    Returns:
        list of urls of the images 
    """

    base_url = f"https://p3sc.oma.be/datarepfiles/{level}/v2/"          #L1/v2/"
    n_level = level[-1]

    r = requests.get(base_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    all_files = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".fits")]

    pattern = r"^aspiics_"

    if filter_list:
        pattern += '(?:' + '|'.join(map(re.escape, filter_list)) + ')'  
    else:
        pattern += r".+"

    pattern += rf"_l{n_level}_"

    if cycle_id != '':
        cycle_ids = [c.strip() for c in str(cycle_id).split(',')]
        pattern += '(?:' + '|'.join(map(re.escape, cycle_ids)) + ')'
    else:
        pattern += r"\d{8}"

    pattern += r"\d{3}"             # 000

    if level != 'L3':
        pattern += r"\d{3}"             # seq_acq_exp numbers
    else:
        pattern += r"\d{1}"
    pattern += r"_(\d{8}T\d{6})\.fits$"

    regex = re.compile(pattern)

    #st.warning(f'{len(all_files)}, {all_files[:4]}')
        
    fmt = "%Y%m%dT%H%M%S"

    matched_files = []
    for f in all_files:
        m = regex.match(f)
        if m:
            date_str = m.group(1)
            dt = datetime.strptime(date_str, fmt)
            if start_dt and dt < start_dt:
                continue
            if end_dt and dt > end_dt:
                continue

            matched_files.append(f'{base_url}{f}')

    return matched_files

def header_from_sunpymap(meta):
    header_dict = {}

    for k, v in meta.items():
        try:
            fits.Card(k, v)  
            header_dict[k] = v
        except Exception:
            pass 
    header = fits.Header(header_dict)
    return header
