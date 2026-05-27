import numpy as np
import sunpy.map
from sunpy.map import GenericMap
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.coordinates import SkyCoord, Longitude
import astropy.units as u
from astropy.io import fits
from astropy.wcs.utils import skycoord_to_pixel
import streamlit as st
from io import BytesIO
from scipy import interpolate
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
from photutils import centroids
from photutils.detection import find_peaks
from astropy.stats import sigma_clipped_stats
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad


from logic.plot import plot_hist_AoLP, plot_local_RF_AoLP


base_dir = Path(__file__).resolve().parent.parent

#@st.cache_resource
#def read_map(filepath=None, data=None, header=None):
#    if filepath is not None:
#        try:
#            with fits.open(filepath) as hdul:
#                hdul.verify('fix')
#                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
#                header = hdul[0].header if hdul[0].data is not None else hdul[1].header
#                header = dict(header)
#        except Exception as e:
#            raise ValueError(f"Error loading file {filepath}: {e}")
#
#    if data is not None and header is not None:
#        filename = str(header.get('FILENAME', '')).upper()
#        
#        if 'metis' in filename:
#            return METISMap(data, header)
#        else:
#            try:
#                return sunpy.map.Map(data, header)
#            except:
#                return GenericMap(data, header)
#            
#    raise ValueError("Need a 'filepath' or 'data'+'header'.")    

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

def download_all_maps_btn(maps_dict, label='💾 Download demodulated maps (.zip)', zipname='demodulated_maps.zip', unique_id='demod_downl'):
    if not maps_dict:
        st.warning("No map to download.")
        return

    if "zip_ready" not in st.session_state:
        st.session_state.zip_ready = False

    if st.button("📦 Create ZIP archive", key=f"zip_btn_{unique_id}"):
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
    #angles_grid = np.mod(np.arctan2(dy, dx), 2*np.pi)
    angles_grid = np.mod(np.arctan2(-dx, dy), 2*np.pi)          # nord = 0 deg and counterclockwise
    
    angle_start, angle_end = angle - ampl/2, angle + ampl/2
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
    #angles_grid = np.mod(np.arctan2(dy, dx), 2*np.pi)
    angles_grid = np.mod(np.arctan2(-dx, dy), 2*np.pi)          # nord = 0 deg and counterclockwise

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

def aspiics_files_api(filter_list, level, orbit_id, cycle_id, start_dt, end_dt, limit):
    # baseAPI url
    baseAPI = "https://p3sc.oma.be/api/"
    # select the columns
    select = "select=name,FILTER,orbit_id,version"
    # add order
    order = "order=DATE-OBS.desc.nullslast"
    # add the fixed filter
    version = "version=eq.v03"

    # add variable filter options
    if level == 'L3':
        filters = 'or('
        for filter in filter_list:
            filters += f'PROD_ID.like.{filter}*,'
        filters = filters[:-1] + ')'
    else:
        filters = 'or('
        for filter in filter_list:
            filters += f'FILTER.like.{filter}*,'
        filters = filters[:-1] + ')'

    start_dt = start_dt.isoformat()
    end_dt = end_dt.isoformat()
    
    #if cycle_id:
    #    if orbit_id:
    #        final_keywords = f"and(and(and({filters},orbit_id.eq.{orbit_id}),CYCLE_ID.eq.{cycle_id}),and(DATE-OBS.gt.{start_dt},DATE-OBS.lt.{end_dt}))"
    #    else:
    #        final_keywords = f"and(and({filters},CYCLE_ID.eq.{cycle_id}),and(DATE-OBS.gt.{start_dt},DATE-OBS.lt.{end_dt}))"
    #elif orbit_id:
    #    final_keywords = f"and(and({filters},orbit_id.eq.{orbit_id}),and(DATE-OBS.gt.{start_dt},DATE-OBS.lt.{end_dt}))"
    #else:
    #    final_keywords = f"and({filters},and(DATE-OBS.gt.{start_dt},DATE-OBS.lt.{end_dt}))"

    conditions = [
        filters,
        f"DATE-OBS.gt.{start_dt}",
        f"DATE-OBS.lt.{end_dt}"
    ]
    if orbit_id:
        conditions.append(f"orbit_id.eq.{orbit_id}")
    if cycle_id:
        conditions.append(f"CYCLE_ID.eq.{cycle_id}")
    final_keywords = f"and({','.join(conditions)})"
    
    # limit the amount of results
    limit_str = f"limit={limit}"

    #generate the final url
    apiQuery = f"{baseAPI}{level}?{select}&{version}&{order}&and=(active.eq.true,{final_keywords})&{limit_str}"
    #headers = {
    #    'User-Agent': 'ASPIICS-Web-Demo/1.1 (Contact: riccardo.chiartano@inaf.it)'
    #}
    #response = requests.get(apiQuery, headers=headers)
    response = requests.get(apiQuery)
    data = response.json()
    st.write(apiQuery)
    fileURLList = []
    baseDataURL = "https://p3sc.oma.be/datarepfiles"
    fileLevel = level
    #st.write(type(data))
    if isinstance(data, list):
        for item in data:
            file_name = item['name']
            file_version = item['version']
            file_url = f"{baseDataURL}/{fileLevel}/{file_version}/{file_name}"
            fileURLList.append(file_url)
    else:
        if data.get('message'):
            st.error(f'Error: {data.get('message')}')
            fileURLList=[]
        else:
            st.error(f'Error: "No files with those keywords"')
            fileURLList=[]
    #st.write(fileURLList)
    return fileURLList

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

def update_meta_rotangle(solar_map, rot_angle):
    new_angle_deg = solar_map.meta['crota'] + rot_angle
    theta = np.deg2rad(new_angle_deg)

    solar_map.meta['pc1_1'] = np.cos(theta)
    solar_map.meta['pc1_2'] = -np.sin(theta)
    solar_map.meta['pc2_1'] = np.sin(theta)
    solar_map.meta['pc2_2'] = np.cos(theta)

    solar_map.meta['crota'] = new_angle_deg

    return solar_map

def f_corona(xx,yy,**kwargs):
    """Gives the current model of F-corona in MSB interpolated to xx and yy [R_Sun] 2D arrays
    """
    #pixscale=2.8125 ; x_IO=1023.5 ; y_IO=1023.5 ; RSun=16.0*60.0
    #xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-x_IO) * pixscale / RSun
    #yy = np.outer(np.linspace(0,2047,num=2048)-y_IO,np.ones(2048)) * pixscale / RSun

    model=kwargs.get('model','standard')
    if model=='simple_sh' or model=='Allen':
        ## Simple polar-symmetrical model of F-corona from Allen 1977 used by sshestov in his initial
        ##    simulated data IDL software (b_corona.pro),  units - [1e-10 MSB]
        ##               *      *    *                    - these three are fake, to simplify interpolation inside 1.1R_Sun
        r_C = np.array([0.01,  0.5, 0.90, 1.01, 1.03, 1.06,  1.10, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 3.00, 4.00, 5.00, 10.0])
        B_F1= np.array([3.27, 3.26, 3.27, 3.22, 3.16, 3.06,  3.00, 2.80, 2.46, 2.24, 2.06, 1.93, 1.81, 1.65, 1.43, 1.10, 0.83, 0.23])-10.0
        #  from my IDL code       R_Sun =[1.01, 1.03, 1.06,  1.10, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 3.00, 4.00, 5.00, 10.0] ; Allen
        #                         B_F_A =[3.22, 3.16, 3.06,  3.00, 2.80, 2.46, 2.24, 2.06, 1.93, 1.81, 1.65, 1.43, 1.10, 0.83, 0.23]
        B_F2= B_F1.copy() #np.array([3.25, 3.24, 3.23, 3.22, 3.16, 3.06,  3.00, 2.80, 2.46, 2.24, 2.06, 1.93, 1.81, 1.65, 1.43, 1.10, 0.83, 0.23])-10.0
        origin='Allen 1977'
        
    else:      # implying standard model=='standard':          
        ## Brightness of the F-corona, Koutchmy (2000); units - [1e-10 MSB]
        ##                *     *      *    *      *     - these five are fake, to simplify interpolation inside 1.1R_Sun
        r_C  = np.array([0.1,  0.5,  0.95, 1.03, 1.06,  1.10, 1.20, 1.40, 1.60, 2.00, 2.50, 3.00, 4.00, 5.00,10.0])
        B_F1 = np.array([3.25, 3.24, 3.23, 3.21,  3.2,  3.10, 2.90, 2.50, 2.25, 1.91, 1.66, 1.48, 1.23, 1.00, 0.31])-10.0
        B_F2 = np.array([3.25, 3.23, 3.23, 3.21,  3.2,  3.10, 2.90, 2.50, 2.25, 1.82, 1.56, 1.33, 1.03, 0.80, 0.06])-10.0
        origin='Koutchmy2000'
    
    
    rr = np.sqrt( np.add(np.square(xx),np.square(yy)) )
    phi= np.arctan2(yy,xx)
    c1= np.abs(np.abs(phi)-np.pi/2.)/(np.pi/2.)
    c2= 1.0 - c1

    kind='linear'      #  ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’
    inter1 = interpolate.interp1d(r_C, B_F1, kind=kind, fill_value='extrapolate')   # bounds_error="False", fill_value=3.2)  gives error?
    inter2 = interpolate.interp1d(r_C, B_F2, kind=kind, fill_value='extrapolate')   

    Fcor1 = inter1(rr)
    Fcor2 = inter2(rr)
    Fcor = c1*Fcor1 + c2*Fcor2
    Fcor = np.power(10.0,Fcor)
    Fcor = Fcor.astype(np.float32)

    verbose=kwargs.get('verbose', False)
    if verbose:
        plt.plot(rr[1024,:],Fcor[1024,:],'b',label="Interpolated horiz. F-cor")
        plt.plot(rr[:,1024],Fcor[:,1024],'r',label="Interpolated vert. F-cor")
        plt.plot(r_C,np.power(10.,B_F1),'-o',label="Tabulated horiz. "+origin)
        plt.plot(r_C,np.power(10.,B_F2),'-*',label="Tabulated vert.  "+origin)
        plt.yscale('log')
        plt.ylim(1e-9,1e-6)        
        plt.xlim(0.0,3.0)
        plt.legend()
        plt.show()

    return Fcor, origin, kind

def remove_fcorona(smap, model='standard'):
    """
    Rimuove il modello di F-corona da una SunPy Map.
    
    Parameters
    ----------
    smap  : sunpy.map.GenericMap
    model : 'standard' (Koutchmy 2000) o 'Allen' (Allen 1977)
    
    Returns
    -------
    smap_kcorona : SunPy Map con F-corona sottratta
    smap_fcorona : SunPy Map della F-corona sottratta
    """
    
    data = smap.data
    header = smap.meta

    pixscale = header['CDELT1']
    CRPIX1   = header['CRPIX1']
    CRPIX2   = header['CRPIX2']
    #CRPIX1   = header['X_IO']-1.0  # !!!! to put back !!!! header['CRPIX1']-1.0            # these are center of the Sun in the image, re-centered during l3_merge
    #CRPIX2   = header['Y_IO']-1.0  # header['CRPIX2']-1.0
    RSUN_ARC = header['RSUN_ARC'] 

    ny, nx = data.shape
    #xx = np.outer(np.ones(ny),  np.arange(nx) - (CRPIX1 - 1)) * pixscale / RSUN_ARC
    #yy = np.outer(np.arange(ny) - (CRPIX2 - 1), np.ones(nx)) * pixscale / RSUN_ARC
    xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-CRPIX1) * pixscale / RSUN_ARC
    yy = np.outer(np.linspace(0,2047,num=2048)-CRPIX2,np.ones(2048)) * pixscale / RSUN_ARC
    
    #Fcor, Fcor_msg, Fcor_kind = f_corona(xx,yy,model='simple_sh')  ### --- Sergei's data were created with Allen model ### ,verbose=True --- with plots
    Fcor, Fcor_msg, Fcor_kind = f_corona(xx,yy,model=model)    ### --- Koutchmy et al 2002  ### ,verbose=True --- with plots
    new_data=data-Fcor

    smap_kcorona = sunpy.map.Map(new_data, smap.meta)
    smap_fcorona = sunpy.map.Map(Fcor, smap.meta)

    return smap_kcorona, smap_fcorona

def fcorona_removed_map(smap, model='standard'):
    """
    Rimuove il modello di F-corona da una Map ASPIICS in WBF o Total Brightness.
    
    Parameters
    ----------
    smap  : sunpy.map.GenericMap
    model : 'standard' (Koutchmy 2000) o 'Allen' (Allen 1977)
    
    Returns
    -------
    smap_kcorona : SunPy Map con F-corona sottratta
    """

    smap_removed, _ = remove_fcorona(smap, model=model)
    headerM = header_from_sunpymap(smap.meta)

    headerM.set("HISTORY", "F-Corona removed using model"+model+" ('standard' (Koutchmy 2000) o 'Allen' (Allen 1977))")
    headerM.set("HISTORY", "old filename: " + headerM["FILENAME"])

    newname=headerM['filename'].split('.')[0]+'.fits'
    if "l2" in newname:
        newname=newname.replace("l2","l3")
    if "wb" in newname:
        newname=newname.replace("wb","bt")
    headerM.set('FILENAME', newname)
    st.write(newname)

    smap_removed = sunpy.map.Map(smap_removed.data, headerM)

    return smap_removed

def search_stars(smap, catalog_name='gaia', fmagn='8'):
    image_data = smap.data
    header = smap.meta
    wcs = smap.wcs
    filename = header.get('filename', 'no_fname')
    
    # 2. Selezione e filtraggio Catalogo
    is_uv = 'uv' in filename.lower()
    channel = 'uv' if is_uv else 'vl'
    mag_col = 'Umag' if is_uv else 'Vmag'
    catalog_name = 'simbad' if is_uv else catalog_name
    mag_limit = fmagn

    if True:
        catalog_stars = catalog_query(smap, mag_limit, catalog=catalog_name, is_uv=is_uv)
    else:
        cat = pd.read_csv('simbad_vmag7.csv')
        # Filtraggio dinamico magnitudine
        cat = cat[cat[mag_col] < mag_limit].copy()
        cat = cat[~np.isnan(cat[mag_col])].copy()
        catalog_stars = cat.copy()

    # NO CORREZIONE ABERRAZIONE
    #print(catalog_stars[:10])
    #sys.exit()

    coords_stars = SkyCoord(
        ra=catalog_stars['ra'].values * u.deg, 
        dec=catalog_stars['dec'].values * u.deg, 
        distance=1e6 * u.pc,
        frame='icrs'
    )
    
    x, y = skycoord_to_pixel(coords_stars, wcs)
    catalog_stars['xsensor'] = x
    catalog_stars['ysensor'] = y

    cx = smap.meta.get('CRPIX1', smap.meta['naxis1'] / 2)
    cy = smap.meta.get('CRPIX2', smap.meta['naxis2'] / 2)
    catalog_stars['r_dist'] = np.sqrt((catalog_stars.xsensor - cx)**2 + 
                                    (catalog_stars.ysensor - cy)**2)
    FOV_min = smap.meta['naxis2']/5        # soluz temporanea

    # Filtro combinato: dentro i bordi del sensore E fuori dall'occultatore
    catalog_stars = catalog_stars[
        (catalog_stars.xsensor > 0) & (catalog_stars.xsensor < wcs.pixel_shape[0]) &
        (catalog_stars.ysensor > 0) & (catalog_stars.ysensor < wcs.pixel_shape[1]) &
        (catalog_stars.r_dist > FOV_min) 
    ].copy()

    st.write(f"Stelle nel sensore: {len(catalog_stars)}")
    #print(catalog_stars)
    if len(catalog_stars) > 30:
        st.warning("Troppe stelle nel sensore, errore??")
        return
    
    return catalog_stars
    # 5. Ricerca Picchi nell'immagine
    #if not catalog_stars.empty:
    #    catalog_stars = find_nearby_stars(image_data, catalog_stars, size=13)
    #    # Rimuoviamo quelle senza un picco identificato
    #    not_identified = catalog_stars[catalog_stars['peak_value'].isna()]
    #    catalog_stars = catalog_stars.dropna(subset=['peak_value'])
    #    st.write(f"Stelle con picco identificato: {len(catalog_stars)}")
    #else:
    #    st.warning("Nessuna stella identificata con successo.")
    #
    #return catalog_stars, not_identified

    # 6. Salvataggio Risultati
    #if args.save and not catalog_stars.empty:
    #    catalog_stars['source_file'] = file_path.name
    #    # Rimuoviamo eventuali colonne 'cutout' (oggetti complessi) prima di salvare in CSV
    #    df_to_save = catalog_stars.drop(columns=['cutout'], errors='ignore')
    #    df_to_save.to_csv(args.output, index=False)
    #    print(f"Risultati salvati in: {args.output}")

    # 7. Plotting
    #if args.plot:
    #    # A. Plot Generale del Campo
    #    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': wcs})
    #    smap.plot(axes=ax)
    #    
    #    # Plot stelle IDENTIFICATE (Cerchio + Croce sul picco)
    #    if not catalog_stars.empty:
    #        # Plot stelle NON identificate (Cerchi tratteggiati)
    #        if not not_identified.empty:
    #            ax.scatter(not_identified.xsensor, not_identified.ysensor,
    #                    s=100, edgecolors='orange', facecolors='none', 
    #                    linestyle='--', alpha=0.7, label='Teoriche (non trovate)')
    #        ax.scatter(catalog_stars.xsensor, catalog_stars.ysensor,
    #                s=150, edgecolors='cyan', facecolors='none', 
    #                linewidths=1.5, label='Teoriche (identificate)')
    #        #ax.scatter(catalog_stars.xsensor, catalog_stars.ysensor, 
    #        #        marker='x', s=60, color='cyan', label='Picco Identificato')
    #        #ax.scatter(catalog_stars.x_centroid, catalog_stars.y_centroid, 
    #        #        marker='x', s=60, color='red', label='Centroide Identificato')
    #
    #    ax.legend(loc='upper right', frameon=True)
    #    ax.set_title(f"Star Tracking - {file_path.name}")
    #    #plt.show()
    #
    #    # B. Plot dei Cutouts (Solo per le identificate)
    #    if not catalog_stars.empty:
    #        print("Generazione dei ritagli per le stelle identificate...")
    #        for i, row in catalog_stars.iterrows():
    #            cutout = image_cutout(image_data, row['x_peak'], row['y_peak'], size=13)
    #            
    #            if cutout is not None:
    #                plt.figure(figsize=(3, 3))
    #                plt.imshow(cutout, origin='lower', cmap=smap.cmap)
    #                
    #                mag_val = row.get(mag_col, np.nan)
    #                plt.title(f"{row.get('main_id', 'N/A')}\n"
    #                        f"{mag_col}: {mag_val:.2f} | Peak: {row['peak_value']:.2e}", 
    #                        fontsize=9)
    #                plt.colorbar(fraction=0.046, pad=0.04)
    #                #plt.show()
    #        if args.plotall:
    #            for i, row in not_identified.iterrows():
    #                cutout = image_cutout(image_data, row['xsensor'], row['ysensor'], size=13)
    #                
    #                if cutout is not None:
    #                    plt.figure(figsize=(3, 3))
    #                    plt.imshow(cutout, origin='lower', cmap=smap.cmap)
    #                    
    #                    mag_val = row.get(mag_col, np.nan)
    #                    plt.title(f"NON IDENTIFICATA\n"
    #                            f"{row.get('main_id', 'N/A')}\n"
    #                            f"{mag_col}: {mag_val:.2f} | Peak: {row['peak_value']:.2e}", 
    #                            fontsize=9)
    #                    plt.colorbar(fraction=0.046, pad=0.04)
    #                    #plt.show()

def catalog_query(s_map, mag_limit=8, catalog='gaia', is_uv=False):
    sun_to_hrcs = s_map.observer_coordinate.transform_to('hcrs')
    center_ra = Longitude(sun_to_hrcs.ra + 180*u.deg)
    center_dec = -sun_to_hrcs.dec
    center_coord = SkyCoord(ra=center_ra, dec=center_dec, frame='icrs')
    
    width = (s_map.meta['naxis1'] * abs(s_map.meta['cdelt1']) * u.arcsec).to(u.deg)
    radius = width * 1.2 / 2  # 20% di margine
    
    if catalog == 'gaia':
        # Query a Gaia DR3 (molto veloce e preciso)
        # Gmag è circa la magnitudine visibile
        job = Gaia.launch_job(f"SELECT ra, dec, pmra, pmdec, parallax, phot_g_mean_mag as Vmag, ref_epoch, source_id "
                            f"FROM gaiadr3.gaia_source "
                            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
                            f"CIRCLE('ICRS', {center_coord.ra.deg}, {center_coord.dec.deg}, {radius.value}))=1 "
                            f"AND phot_g_mean_mag < {mag_limit}")
        table = job.get_results()
        df = table.to_pandas()
        df['main_id'] = "Gaia_" + df['source_id'].astype(str)
    else:
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('V', 'U', 'pmra', 'pmdec', 'parallax')
        
        table = custom_simbad.query_region(center_coord, radius=radius)
        
        if table is not None:
            # 1. Rinominazione Colonne
            if 'V' in table.colnames: table.rename_column('V', 'Vmag')
            elif 'FLUX_V' in table.colnames: table.rename_column('FLUX_V', 'Vmag')
                
            if 'U' in table.colnames: table.rename_column('U', 'Umag')
            elif 'FLUX_U' in table.colnames: table.rename_column('FLUX_U', 'Umag')
            
            # 2. Conversione in Pandas per un filtraggio più agile
            df = table.to_pandas()
            
            # 3. Taglio sulla Magnitudine (Filtro dinamico)
            # Se siamo in UV usiamo Umag, altrimenti Vmag
            current_mag_col = 'Umag' if is_uv else 'Vmag'
            
            if current_mag_col in df.columns:
                # Rimuoviamo i NaN e applichiamo il limite
                initial_count = len(df)
                df = df[df[current_mag_col].notna()].copy()
                df = df[df[current_mag_col] < mag_limit].copy()
                print(f"Simbad: filtrate {len(df)} stelle (da {initial_count}) con {current_mag_col} < {mag_limit}")
            
            # 4. Standardizzazione nomi coordinate e epoca
            if 'RA' in df.columns: df.rename(columns={'RA': 'ra'}, inplace=True)
            if 'DEC' in df.columns: df.rename(columns={'DEC': 'dec'}, inplace=True)
            df['ref_epoch'] = 2000.0
                    
    return df

def image_cutout(image_data, x, y, size):
    """
    Returns a cutout at pixel x, y of size x size pixels

    Parameters
    ----------
    image_data : ndarray
        array with the image.
    x, y : float
        center of the cutout.
    size : int
        size of the cutout.

    Returns
    -------
    ndarray
        cutout of the image.

    """
    if np.isnan(x) | np.isnan(y):
        return None

    x, y = int(x), int(y)
    xmax, ymax = image_data.shape
    x0 = x-size if x-size > 0 else 0
    y0 = y-size if y-size > 0 else 0
    x1 = x+size+1 if x+size+1 < xmax else xmax
    y1 = y+size+1 if y+size+1 < ymax else ymax

    return image_data[y0:y1,x0:x1]

def find_nearby_stars(image_data, stars, size=10, 
                      centroid_func=centroids.centroid_2dg):
    '''
    Finds stars in an image nearby a set of location in pixels, 
    using a centroiding function if provided 
    (eg. the ones in photutils.centroids)

    Parameters
    ----------
    image_data : array
    stars : DataFrame
        coordinates of potential stars.
    size : int, optional
        size of the box where to look for stars. The default is 10.
    centroid_func : function, optional
        centroid function to use. The default is centroids.centroid_2dg.

    Returns
    -------
    stars : Astropy Table
        table with columns 'x_peak', 'y_peak', 'peak_value', 'x_centroid', 
        'y_centroid'. 'peak_value' includes the floor (median of the image)

    '''
  
    new_cols = ['x_peak', 'y_peak', 'peak_value', 'peak_floor', 'x_centroid', 'y_centroid']
    
    # add new columns to the df
    stars = stars.reindex(columns=stars.columns.tolist() + new_cols)
    
    g_size = size
    for row in stars.itertuples():
        # get a cutout of the probable star
        idx = row.Index
        x, y = row.xsensor, row.ysensor
        sub = image_cutout(image_data, x, y, size)
        if sub is not None:
          
            # mask 0 pixels because they don't contain data
            mask = sub == 0
            mean, median, std = sigma_clipped_stats(sub[~mask], sigma=3.0)
            
            # hack to fix issue in photutils 1.3.0
            if centroid_func is centroids.centroid_2dg:
                def centroid_func(data, mask):
                    return centroids.centroid_2dg(data, mask=mask)
            
            f = find_peaks(sub-median, 
                           threshold=3*std, 
                           box_size=size*2+1, 
                           npeaks=1,
                           mask=mask,
                           centroid_func=centroid_func)
            if f:
                # translate to global sensor coordinates
                stars.at[idx, 'x_peak'] = f['x_peak'] + int(x) - size 
                stars.at[idx, 'y_peak'] = f['y_peak'] + int(y) - size
                
                # add back floor to the peak and save it
                stars.at[idx,'peak_value'] = f['peak_value'] + median
                stars.at[idx, 'peak_floor'] = median
                
                # centroids outside of the sub are clearly a bad fit
                if (f['x_centroid'] > 0) & (f['x_centroid'] < size*2) & \
                   (f['y_centroid'] > 0) & (f['y_centroid'] < size*2):

                    stars.at[idx, 'x_centroid'] = \
                        f['x_centroid'] + int(x) - size 
                    stars.at[idx, 'y_centroid'] = \
                        f['y_centroid'] + int(y) - size 
        
    return stars
