from astropy.coordinates import SkyCoord
from astropy.io import fits
from glob import glob
from scipy.io import readsav
import astropy.units as u
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sunpy
import sunpy.map

from logic.process import calculate_darkbias
from logic.rob import aspiics_misc as am

def save_all(filepath, maps, map_names):
    for map_name in map_names:
        maps[map_name].save(f'{filepath}/{map_name}.fits', overwrite=True)
        print(f'Saved {filepath}/{map_name}.fits')

def define_mask(image, center_y, center_x, r_in, r_out):
    y, x = np.indices(image.shape)
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist >= r_in) & (dist <= r_out) & ~np.isnan(image)
    return mask

def mean_mask (image, mask):
    mean_val = np.nanmean(image[mask])
    return(mean_val)

def sun_center (map_path):
    sun_map = sunpy.map.Map(map_path)
    sun_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=sun_map.coordinate_frame)
    sun_x, sun_y = sun_map.wcs.world_to_pixel(sun_center)
    return (int(sun_x), int(sun_y))

def get_filenames_from_folder(file_path):
    filenames = []
    filenames000 = glob(os.path.join(file_path, '*p2*'))
    filenames060 = glob(os.path.join(file_path, '*p1*'))
    filenames120 = glob(os.path.join(file_path, '*p3*'))

def read_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data

def read_fits_2(path):
    with fits.open(path) as hdul:
        return hdul[1].data

def read_fits_header(path):
    with fits.open(path) as hdul:
        return hdul[0].header

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

def add_keywords_header(header):
    header['RSUN_OBS'] = 960.0  # [arcsec] raggio apparente del Sole
    header['DSUN_OBS'] = 1.496e11  # [m] distanza Terra-Sole (~1 AU)
    header['HGLT_OBS'] = 0  # [deg] B0 angle
    header['CRLN_OBS'] = 0  # [deg] L0 angle
    header['DATE-OBS'] = header['DATE']  # Timestamp reale
    return header

def create_maps(filename, image, header, header_keys):
    # aggiorno keywords header
    files, btype, bunit, file_dark, file_demod_matr, date_beg, date_avg, date_end = header_keys
    header['NAXIS1'] = (image.shape[0])
    header['NAXIS2'] = (image.shape[1])
    header['FILE_NME'] = filename
    header['FILENAME'] = filename
    header['DATE'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    header['AUTHOR'] = 'Riccardo Chiartano'
    header['PARENT0'] = files[0]
    header['PARENT60'] = files[1]
    header['PARNT120'] = files[2]
    header['BTYPE'] = btype
    header['BUNIT'] = bunit
    header['DARK'] = file_dark
    header['DMTRX'] = file_demod_matr
    header['MSB'] = 2.082e20
    header['LEVEL'] = 'L2'
    header['ORIGIN'] = 'INAF-OATo'
    header['CREATOR'] = 'ASPIICS demodulation v0.0'
    header['FILTER'] = '0, 60, 120'
    header['POLAR'] = ''
    header['DATE-BEG'] = date_beg.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    header['DATE-AVG'] = date_avg.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    header['DATE-END'] = date_end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

    _map = sunpy.map.Map(image, header)

    return(_map)


def demodulate(base_files, matrix_path):
    """
    Funzione che esegue la demodulazione.

    Args:
        base_files: dict con le chiavi come angoli ('0°', '60°', etc) e valori i filepath.
        matrix_path: stringa, con path matrice scelta.

    Returns:
        mappe in formato dizionario.
    """

    files = [base_files['0°'], base_files['60°'], base_files['120°']]
    file_bias = base_files['bias']
    file_dark_A2 = base_files['dark_A2']
    file_dark_B2 = base_files['dark_B2']
    file_dark_C2 = base_files['dark_C2']
    ak_files = [base_files['ak 0°'], base_files['ak 60°'], base_files['ak 120°']]

    headers = [read_fits_header(filename) for filename in files]
    header = headers[0]
    bias_image = read_fits(file_bias)
    dark_image_A2 = read_fits(file_dark_A2)
    dark_image_B2 = read_fits(file_dark_B2)
    dark_image_C2 = read_fits(file_dark_C2)
    dark_image = calculate_darkbias(bias_image, dark_image_A2, dark_image_B2, dark_image_C2, header)
    images_noak = [np.array(read_fits(filename)) for filename in files]     # ciclo su tutti i files

    t_exps = [float(header['EXPTIME']) for header in headers]

    # tolgo valori saturati e poi il dark
    for i, image in enumerate(images_noak): 
        images_noak[i][image>16000] = np.nan
        images_noak[i] = image - dark_image 
        images_noak[i][images_noak[i]<0] = np.nan                           # forza i valori >0

    sigs = [np.sqrt(image) for image in images_noak]
    aks = [np.array(read_fits(filename)) for filename in ak_files]
    # filtra ak < 1e-13
    for i, ak in enumerate(aks):
        aks[i][ak<1e-13] = np.nan
    MSB = 2.018e20

    images = [image / (ak * t_exp * MSB) for image, ak, t_exp in zip(images_noak, aks, t_exps)]
    sigs = [sig / (ak * t_exp * MSB) for sig, ak, t_exp in zip(sigs, aks, t_exps)]
    bunits = [['MSB', 'MSB', 'MSB', 'MSB', '', 'rad'], ['MSB', 'MSB', 'MSB', 'MSB', '', 'rad']]
    dn = ''

    # definisco una maschera mask per considerare solo i dati fuori dall'occultatore nelle medie
    sun_x, sun_y = sun_center(files[0])
    mask = define_mask(images_noak[0], sun_y, sun_x, r_in=450, r_out=980)

    # set begin, average and end time of the observation sequence
    date_begs = [datetime.datetime.strptime(header['DATE-BEG'], '%Y-%m-%dT%H:%M:%S.%f') for header in headers]
    date_ends = [datetime.datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') for header in headers]
    date_beg = min(date_begs)
    date_end = max(date_ends)
    date_avg = date_beg + (date_end - date_beg)/2
    date_beg_txt = date_beg.strftime('%Y%m%d%H%M%S')
    id_seq = header['CYCLE_ID']

    ###########################
    ###### DEMODULAZIONE ######
    ###########################

    file_demod_matr = matrix_path
    filename_demod_matr = os.path.split(file_demod_matr)[-1]
    
    txt, ext = os.path.splitext(filename_demod_matr)
    txt = txt.replace('demod_matrix', '')
    print(f'usata matrice: {filename_demod_matr}')

    if ext == '.sav':
        data = readsav(file_demod_matr)
        #print(data.keys())
        demod_matr = data['mat_b']
        demod_matr = np.transpose(demod_matr, (2, 3, 0, 1))
        demod_matr_err = data['mat_b_dev']
        demod_matr_err = np.transpose(demod_matr_err, (2, 3, 0, 1))
    else:
        demod_matr = read_fits(file_demod_matr)
        demod_matr_err = read_fits_2(file_demod_matr)

    M = np.stack(images, axis=-1) 
    sig_M = np.stack(sigs, axis=-1) 

    # calcola il vett di stokes (einsum moltiplicazione matriciale più veloce di ciclo for px per px)
    S = np.einsum('ijkl,ijl->ijk', demod_matr, M) 
    sig_S = np.sqrt(np.einsum('ijkl,ijl->ijk', demod_matr_err**2, M**2) + 
            np.einsum('ijkl,ijl->ijk', demod_matr**2, sig_M**2))

    #print(np.nanmean(np.einsum('ijkl,ijl->ijk', demod_matr_err**2, M**2)[mask]))
    #print(np.nanmean(np.einsum('ijkl,ijl->ijk', demod_matr_nuova**2, sig_M**2)[mask]))

    # si estraggono I, Q, U
    I = S[..., 0]
    Q = S[..., 1]
    U = S[..., 2]

    sig_I = sig_S[..., 0]
    sig_Q = sig_S[..., 1]
    sig_U = sig_S[..., 2]

    # calcola grado di polarizzazione p, polarized brightness pb e angolo di polarizzazione psi con relativi errori
    p = np.sqrt(Q**2 + U**2) / I
    pb = np.sqrt(Q**2 + U**2)
    psi = 0.5*np.arctan2(U, Q)
    #psi[psi<=-0.5*np.pi] = psi[psi<=-0.5*np.pi] + np.pi
    #psi[psi>=0.5*np.pi] = psi[psi>=0.5*np.pi] - np.pi

    denominator = I * np.sqrt(Q**2 + U**2)  # = p * I²
    term1 = (p / I)**2 * sig_I**2
    term2 = (Q / denominator)**2 * sig_Q**2
    term3 = (U / denominator)**2 * sig_U**2
    sig_p = np.sqrt(term1 + term2 + term3)

    sig_pb = np.sqrt(Q**2 * sig_Q**2 + U**2 * sig_U**2) / pb

    denominator = 2 * (Q**2 + U**2)
    sig_psi = np.sqrt(U**2 * sig_Q**2 + Q**2 * sig_U**2) / denominator

    #####################
    ### manca il save ###
    #####################

    images = [[I, Q, U, pb, p, psi], 
              [sig_I, sig_Q, sig_U, sig_pb, sig_p, sig_psi]]
    names = [
        [f'aspiics_{name}{dn}_l2_{id_seq}_{date_beg_txt}.fits' for name in ('I', 'Q', 'U', 'pb', 'DoLP', 'AoLP')],
        [f'aspiics_{name}{dn}_l2_{id_seq}_{date_beg_txt}.fits' for name in ('sig_I', 'sig_Q', 'sig_U', 'sig_pb', 'sig_DoLP', 'sig_AoLP')]
    ]
    siglas = [['I', 'Q', 'U', 'pb', 'DoLP', 'AoLP'],['sig_I', 'sig_Q', 'sig_U', 'sig_pb', 'sig_DoLP', 'sig_AoLP']]
    btypes = [['VL total brightness',
              'Stokes Q', 
              'Stokes U',
              'VL polarized brightness',
              'Degree of linear polarization',
              'VL polarization angle'],
             ['Sigma VL total brightness',
              'Sigma Stokes Q', 
              'Sigma Stokes U',
              'Sigma VL polarized brightness',
              'Sigma degree of linear polarization',
              'Sigma VL polarization angle']
              ]

    if 'BLANK' in header:
        del header['BLANK']
    maps = [[],[]]
    for i, (image_row, btype_row, bunit_row, name_row) in enumerate(zip(images, btypes, bunits, names)):
        for image, btype, bunit, name in zip(image_row, btype_row, bunit_row, name_row):        
            header_keys = (files, btype, bunit, 'file_dark', filename_demod_matr, date_beg, date_avg, date_end)
            maps[i].append(create_maps(name, image, header, header_keys))
    print(f'Created maps...')

    maps_dict = {}

    for i in range(len(maps)): 
        for j in range(len(maps[i])):
            name = siglas[i][j]
            maps_dict[name] = maps[i][j]

    return maps_dict


def demodulate_merged(merged_maps, matrix_path):
    """
    Funzione che esegue la demodulazione delle immagini merged.

    Args:
        merged_maps: mappe già merged in ordine 0, 60, 120. (calibrate in MSB)
        matrix_path: stringa, con path matrice scelta.

    Returns:
        mappe in formato dizionario. 
    """


    images = [smap.data for smap in merged_maps]
    headers = [smap.meta for smap in merged_maps]
    header = headers[0]

    ####### da mettere a posto per header e sigs #######
    sigs = [np.ones_like(image) for image in images]     
    bunit = merged_maps[0].meta['bunit']       
    bunits = [[bunit, bunit, bunit, bunit, '', 'rad'], [bunit, bunit, bunit, bunit, '', 'rad']]
    dn = ''
    id_seq = header['CYCLE_ID']
    files = [f'merged_images_{id_seq}_0', f'merged_images_{id_seq}_60', f'merged_images_{id_seq}_120']
    file_dark = 'bias_A.fits'
    ####################################################

    # definisco una maschera mask per considerare solo i dati fuori dall'occultatore nelle medie
    #sun_x, sun_y = sun_center(files[0])
    #mask = define_mask(images_noak[0], sun_y, sun_x, r_in=450, r_out=980)

    # set begin, average and end time of the observation sequence
    date_begs = [datetime.datetime.strptime(header['DATE-BEG'], '%Y-%m-%dT%H:%M:%S.%f') for header in headers]
    date_ends = [datetime.datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') for header in headers]
    date_beg = min(date_begs)
    date_end = max(date_ends)
    date_avg = date_beg + (date_end - date_beg)/2
    date_beg_txt = date_beg.strftime('%Y%m%dT%H%M%S')

    ###########################
    ###### DEMODULAZIONE ######
    ###########################

    file_demod_matr = matrix_path
    filename_demod_matr = os.path.split(file_demod_matr)[-1]
    
    txt, ext = os.path.splitext(filename_demod_matr)
    txt = txt.replace('demod_matrix', '')
    print(f'usata matrice: {filename_demod_matr}')

    if ext == '.sav':
        data = readsav(file_demod_matr)
        #print(data.keys())
        demod_matr = data['mat_b']
        demod_matr = np.transpose(demod_matr, (2, 3, 0, 1))
        demod_matr_err = data['mat_b_dev']
        demod_matr_err = np.transpose(demod_matr_err, (2, 3, 0, 1))
    else:
        demod_matr = read_fits(file_demod_matr)
        #demod_matr_err = read_fits_2(file_demod_matr)

    M = np.stack(images, axis=-1) 
    sig_M = np.stack(sigs, axis=-1) 

    # calcola il vett di stokes (einsum moltiplicazione matriciale più veloce di ciclo for px per px)
    S = np.einsum('ijkl,ijl->ijk', demod_matr, M) 
    #sig_S = np.sqrt(np.einsum('ijkl,ijl->ijk', demod_matr_err**2, M**2) + 
    #        np.einsum('ijkl,ijl->ijk', demod_matr**2, sig_M**2))

    #print(np.nanmean(np.einsum('ijkl,ijl->ijk', demod_matr_err**2, M**2)[mask]))
    #print(np.nanmean(np.einsum('ijkl,ijl->ijk', demod_matr_nuova**2, sig_M**2)[mask]))

    # si estraggono I, Q, U
    I = S[..., 0]
    Q = S[..., 1]
    U = S[..., 2]

    #sig_I = sig_S[..., 0]
    #sig_Q = sig_S[..., 1]
    #sig_U = sig_S[..., 2]

    sig_I = np.sqrt(S[..., 0])
    sig_Q = np.sqrt(S[..., 1])
    sig_U = np.sqrt(S[..., 2])

    # calcola grado di polarizzazione p, polarized brightness pb e angolo di polarizzazione psi con relativi errori
    p = np.sqrt(Q**2 + U**2) / I
    pb = np.sqrt(Q**2 + U**2)
    psi = 0.5*np.arctan2(U, Q)
    #psi[psi<=-0.5*np.pi] = psi[psi<=-0.5*np.pi] + np.pi
    #psi[psi>=0.5*np.pi] = psi[psi>=0.5*np.pi] - np.pi

    denominator = I * np.sqrt(Q**2 + U**2)  # = p * I²
    term1 = (p / I)**2 * sig_I**2
    term2 = (Q / denominator)**2 * sig_Q**2
    term3 = (U / denominator)**2 * sig_U**2
    sig_p = np.sqrt(term1 + term2 + term3)

    sig_pb = np.sqrt(Q**2 * sig_Q**2 + U**2 * sig_U**2) / pb

    denominator = 2 * (Q**2 + U**2)
    sig_psi = np.sqrt(U**2 * sig_Q**2 + Q**2 * sig_U**2) / denominator

    #####################
    ### manca il save ###
    #####################

    images = [[I, Q, U, pb, p, psi], 
              [sig_I, sig_Q, sig_U, sig_pb, sig_p, sig_psi]]
    names = [
        [f'aspiics_{name}{dn}_l2_{id_seq}_{date_beg_txt}.fits' for name in ('I', 'Q', 'U', 'pb', 'DoLP', 'AoLP')],
        [f'aspiics_{name}{dn}_l2_{id_seq}_{date_beg_txt}.fits' for name in ('sig_I', 'sig_Q', 'sig_U', 'sig_pb', 'sig_DoLP', 'sig_AoLP')]
    ]
    siglas = [['I', 'Q', 'U', 'pb', 'DoLP', 'AoLP'],['sig_I', 'sig_Q', 'sig_U', 'sig_pb', 'sig_DoLP', 'sig_AoLP']]
    btypes = [['VL total brightness',
              'Stokes Q', 
              'Stokes U',
              'VL polarized brightness',
              'Degree of linear polarization',
              'VL polarization angle'],
             ['Sigma VL total brightness',
              'Sigma Stokes Q', 
              'Sigma Stokes U',
              'Sigma VL polarized brightness',
              'Sigma degree of linear polarization',
              'Sigma VL polarization angle']
              ]

    if 'BLANK' in header:
        del header['BLANK']
    maps = [[],[]]
    for i, (image_row, btype_row, bunit_row, name_row) in enumerate(zip(images, btypes, bunits, names)):
        for image, btype, bunit, name in zip(image_row, btype_row, bunit_row, name_row):        
            header_keys = (files, btype, bunit, file_dark, filename_demod_matr, date_beg, date_avg, date_end)
            maps[i].append(create_maps(name, image, header, header_keys))
    print(f'Created maps...')

    maps_dict = {}

    for i in range(len(maps)): 
        for j in range(len(maps[i])):
            name = siglas[i][j]
            maps_dict[name] = maps[i][j]

    return maps_dict

def demodulate_rob(maps,
                   polar_files, 
                   outputdir = './output/',
                   caldir = './rob_calib_data/',
                   docenter = True):
    # # receive input filenames from the command line. 
    map1, map2, map3 = maps[0], maps[1], maps[2]
    file1 =map1.meta['filename']  #sys.argv[1]
    file2 =map2.meta['filename']  #sys.argv[2]
    file3 =map3.meta['filename']  #sys.argv[3]
    #file1="input/polariz_testCalData_0.fits"
    #file2="input/polariz_testCalData_60.fits"
    #file3="input/polariz_testCalData_120.fits"

    file2write_I =     os.path.join(outputdir,os.path.splitext(os.path.basename(file1))[0]+'.totalB.fits')         # 'output_totalB.fits'
    file2write_pB =    os.path.join(outputdir,os.path.splitext(os.path.basename(file1))[0]+'.pB.fits')             # 'output_pB.fits'
    file2write_alpha = os.path.join(outputdir,os.path.splitext(os.path.basename(file1))[0]+'.alpha.fits')          # 'output_alpha.fits'

    print("    Reading calibration data from ",caldir)
    #polar1_filename = os.path.join(caldir,"aspiics_p1_angle.fits")  #filename = params["calib_data"]["nonlin"]
    #polar2_filename = os.path.join(caldir,"aspiics_p2_angle.fits")
    #polar3_filename = os.path.join(caldir,"aspiics_p3_angle.fits")
    polar1, polar1_head =am.read_fits_image_array(polar_files[0])
    polar2, polar2_head =am.read_fits_image_array(polar_files[1])
    polar3, polar3_head =am.read_fits_image_array(polar_files[2])
    #polar1[:,:]=0.
    #polar2[:,:]=60.
    #polar3[:,:]=120.
    print("    Polarization mean angles: ", np.mean(polar1), ", ", np.mean(polar2), ", ", np.mean(polar3), "  (before CROTA)")


    # # receive input filenames from the command line. 
    #file1 =args.file1  #sys.argv[1]
    #file2 =args.file2  #sys.argv[2]
    #file3 =args.file3  #sys.argv[3]
    #file1="input/polariz_testCalData_0.fits"
    #file2="input/polariz_testCalData_60.fits"
    #file3="input/polariz_testCalData_120.fits"

    data1, header1 =map1.data, header_from_sunpymap(map1.meta)
    data2, header2 =map2.data, header_from_sunpymap(map2.meta)
    data3, header3 =map3.data, header_from_sunpymap(map3.meta)
    print("    Input files:")
    print("     file1: ",file1, " Polar=",header1['POLAR'])
    print("     file2: ",file2, " Polar=",header2['POLAR'])
    print("     file3: ",file3, " Polar=",header3['POLAR'])

    # Re-center and rotate images.
    print("  ************************** file p1 ************************** ")
    print(file1)
    CROTA1 = header1['CROTA']                 # CROTA denotes how the solar image should be rotated (counter-clockwise, i.e. positive mathematical angle) to make solar north vertical
    #POLAR1 = header1['POLAR']                # now we read it from calibration file
    header1c=header1.copy()
    #smap = sunpy.map.Map(data1, header1)
    #smap.peek()
    if docenter:
        data1 = am.rotate_center1(data1,header1)   
        polar1= am.rotate_center1(polar1,header1c,verbose=True)+CROTA1   # we need to apply the same transformations to the polarization angles
    #smap = sunpy.map.Map(data1, header1)
    #smap.peek()
    ###polar1 = polar1 + CROTA1                  # during rotation we efficiently change the orientation of polarization
    #plt.imshow(polar1,vmin=5.26-1.0,vmax=5.26+1.0,cmap='RdBu',origin='lower')
    #plt.colorbar()

    print("  ************************** file p2 ************************** ")
    print(file2)
    CROTA2 = header2['CROTA']
    #POLAR2 = header2['POLAR']
    header2c=header2.copy()
    if docenter:
        data2 = am.rotate_center1(data2,header2)
        polar2= am.rotate_center1(polar2,header2c,verbose=True)+CROTA2 
    ##polar2 = polar2 + CROTA2
    #plt.imshow(polar2,vmin=64.63-1.0,vmax=64.63+1.0,cmap='RdBu',origin='lower')

    print("  ************************** file p3 ************************** ")
    print(file3)
    CROTA3 = header3['CROTA']
    #POLAR3 = header3['POLAR']
    header3c=header3.copy()
    if docenter:
        data3 = am.rotate_center1(data3,header3)
        polar3= am.rotate_center1(polar3,header3c,verbose=True)+CROTA3 
    ##polar3 = polar3 + CROTA3

    #print("Polarization angles: ", POLAR1, ", ", POLAR2, ", ", POLAR3)
    print("Polarization angles: ", np.nanmean(polar1), ", ", np.nanmean(polar2), ", ", np.nanmean(polar3))

    # Modulation matrix
    #M = 0.5 * np.array([[1.0, np.cos(2.0*np.deg2rad(POLAR1)), np.sin(2.0*np.deg2rad(POLAR1))],
    #                    [1.0, np.cos(2.0*np.deg2rad(POLAR2)), np.sin(2.0*np.deg2rad(POLAR2))],
    #                    [1.0, np.cos(2.0*np.deg2rad(POLAR3)), np.sin(2.0*np.deg2rad(POLAR3))]])
    #print("Modulation matrix")
    #print(M)
    ## Demodulation matrix   
    #Dem = np.linalg.inv(M)
    ## Stokes components
    #I = Dem[0,0]*data1 + Dem[0,1]*data2 + Dem[0,2]*data3              # should be 2/3*(1 + 1 + 1) in ideal case 0, 60, 120
    #Q = Dem[1,0]*data1 + Dem[1,1]*data2 + Dem[1,2]*data3              # should be 2/3*(2 - 1 - 1) in ideal case
    #U = Dem[2,0]*data1 + Dem[2,1]*data2 + Dem[2,2]*data3              #           2/3*(0 + 1.732 - 1.732)


    ############### per-pixel derivation of modulation and demodulation matrix ###############
    # tstart = time.time()
    # C1=np.cos(2.0*np.deg2rad(polar1))  ;  S1=np.sin(2.0*np.deg2rad(polar1))
    # C2=np.cos(2.0*np.deg2rad(polar2))  ;  S2=np.sin(2.0*np.deg2rad(polar2))
    # C3=np.cos(2.0*np.deg2rad(polar3))  ;  S3=np.sin(2.0*np.deg2rad(polar3))
    # M = np.zeros((3,3,2048,2048))
    # Dem=np.zeros((3,3,2048,2048))
    # for y in range(2048):
    #     for x in range(2048):
    #         M1 = 0.5 * np.array([[1.0, C1[y,x], S1[y,x]],
    #                              [1.0, C2[y,x], S2[y,x]],
    #                              [1.0, C3[y,x], S3[y,x]]])
    #         M[:,:,y,x] = M1
    #         Dem1 = np.linalg.inv(M1)
    #         Dem[:,:,y,x] = Dem1
    # tend = time.time()
    # print(tend - tstart)

    print("    Calculating demodulation matrixes (on the per-pixel basis):")
    ############### analytical formulas for demodulation matrix, the demodulation tensor is a linear combination of #########################
    ## polarization angles and coefficients. See "demodulation_matrix_derivation.1.jpg" for explanations. 
    C1=np.cos(2.0*np.deg2rad(polar1))  ;  S1=np.sin(2.0*np.deg2rad(polar1))
    C2=np.cos(2.0*np.deg2rad(polar2))  ;  S2=np.sin(2.0*np.deg2rad(polar2))
    C3=np.cos(2.0*np.deg2rad(polar3))  ;  S3=np.sin(2.0*np.deg2rad(polar3))
    ## we find analytically the inverse matrix by converting (ident|modul)-> into (demodul|ident) finding necessary transformations
    ## we need some coefficients, which are 2D arrays
    A=1./(C2-C1)  ;  B=1./(C3-C1)  ;  E=1./(B*(S3-S1)-A*(S2-S1))  ;  F=(S2-S1)/(C2-C1)
    ## The demodulation matrix(tensor) consists from 9 elements, each of which is a 2D array. Notation is DemXY
    Dem12=(B-A)*E*F-A   ;   Dem22=A*E*F+A   ;   Dem32=-B*E*F    # this is 2nd row (y=2) of the demodulation matrix
    Dem13=(A-B)*E       ;   Dem23=-A*E      ;   Dem33=B*E       # this is 3rd row (y=3)
    Dem11=np.ones((2048,2048))  ;  Dem21=np.zeros((2048,2048))  ;  Dem31=np.zeros((2048,2048))
    Dem11=Dem11-Dem12*C1-Dem13*S1
    Dem21=Dem21-Dem22*C1-Dem23*S1
    Dem31=Dem31-Dem32*C1-Dem33*S1
    ## The derivation was done for 2M. Need to take into account 1/2
    Dem11=Dem11*2.  ;  Dem21=Dem21*2.  ;  Dem31=Dem31*2.
    Dem12=Dem12*2.  ;  Dem22=Dem22*2.  ;  Dem32=Dem32*2.
    Dem13=Dem13*2.  ;  Dem23=Dem23*2.  ;  Dem33=Dem33*2.
    Dem_av=np.array([[np.nanmean(Dem11),np.nanmean(Dem21),np.nanmean(Dem31)],           #  for 0,60,120 should equal to 2/3*(1    1      1) 
                    [np.nanmean(Dem12),np.nanmean(Dem22),np.nanmean(Dem32)],           #                               2/3*(2   -1     -1)
                    [np.nanmean(Dem13),np.nanmean(Dem23),np.nanmean(Dem33)]])          #                               2/3*(0 1.732  1.732)


    print("Demodulation matrix")
    print(Dem_av)

    # Stokes components
    I = Dem11*data1 + Dem21*data2 + Dem31*data3              # should be 2/3*(1 + 1 + 1) in ideal case 0, 60, 120
    Q = Dem12*data1 + Dem22*data2 + Dem32*data3              # should be 2/3*(2 - 1 - 1) in ideal case
    #print("Q coeff: ", Dem[1,0], Dem[1,1], Dem[1,2])
    U = Dem13*data1 + Dem23*data2 + Dem33*data3              #           2/3*(0 + 1.732 - 1.732)
    #print("U coeff: ", Dem[2,0], Dem[2,1], Dem[2,2])

    pB = np.sqrt(Q**2 + U**2)
    alpha = 0.5*np.arctan2(U,Q)                                       # !!!!! syntax: np.arctan2(y,x) - stupid documentation!!!! 
    #alpha = 0.5*np.arctan(U/Q)

    ### ********** convert to 32bit float ************ ###
    I = I.astype(np.float32)
    pB = pB.astype(np.float32)
    alpha = alpha.astype(np.float32)

    ### ********** filling all the points with original NaN or Inf data with NaN in the output *************** ####
    ### ********** but probably is not needed here                                             *************** ####
    #good_mask = np.isfinite(data1) | np.isfinite(data2) | np.isfinite(data3)
    #I[ ~ good_mask ]  = np.nan
    #pB[ ~ good_mask ] = np.nan
    #alpha[ ~ good_mask ] = np.nan

    ### ************** prepare headers ****************** ###
    unit = header1['bunit']
    #header1.set('CRPIX1',1024.5, "[pixel] Pixel scale along axis x, arcsec")
    #header1.set('CRPIX2',1024.5, "[pixel] Pixel scale along axis y, arcsec")
    #header1.set('HISTORY',"Image has been centered before processing")
    header1.set('HISTORY',"The polarized data has been calculated using")
    header1.set('HISTORY',file1,", ")
    header1.set('HISTORY',file2,", ")
    header1.set('HISTORY',file3)
    header1.set('LEVEL','L3')

    header_I = header1.copy()
    header_I.set('PROD_ID',"Total brightness",after='LEVEL')
    header_I.set('BTYPE','Total brightness','for polarized data - B, pB, alpha',before='BUNIT')
    header_I.set('BUNIT',unit)
    header_I.set('FILTER','0, 60, 120','Spectral passband corresponds to WB')
    header_I.set('POLAR','                  ','Removed after pol.processing')

    header_pB = header1.copy()
    header_pB.set('PROD_ID',"Polarised brightness",after='LEVEL')
    header_pB.set('BTYPE','Polarized brightness','for polarized data - B, pB, alpha',before='BUNIT')
    header_pB.set('BUNIT',unit,'here MSB referes to B')
    header_pB.set('FILTER','0, 60, 120','Spectral passband corresponds to WB')
    header_pB.set('POLAR','                  ','Removed after pol.processing')

    header_alpha = header1.copy()
    header_alpha.set('PROD_ID',"Polarised angle",after='LEVEL')
    header_alpha.set('BTYPE','Polarization angle','for polarized data - B, pB, alpha',before='BUNIT')
    header_alpha.set('BUNIT','deg','Angle [deg] WRT horiz pixel')
    header_alpha.set('FILTER','0, 60, 120','Spectral passband corresponds to WB')
    header_alpha.set('POLAR','                  ','Removed after pol.processing')

    file2write_I     = os.path.splitext(os.path.basename(file1))[0]+'.totalB.fits'         # 'output_totalB.fits'
    file2write_pB    = os.path.splitext(os.path.basename(file1))[0]+'.pB.fits'             # 'output_pB.fits'
    file2write_alpha = os.path.splitext(os.path.basename(file1))[0]+'.alpha.fits'          # 'output_alpha.fits'
    file2write_I    =file2write_I.replace("l2","l3")
    file2write_pB   =file2write_pB.replace("l2","l3")
    file2write_alpha=file2write_alpha.replace("l2","l3")

    header_I['FILENAME'] = file2write_I
    header_alpha['FILENAME'] = file2write_alpha
    header_pB['FILENAME'] = file2write_pB

    map_I = sunpy.map.Map(I, header_I)
    map_alpha = sunpy.map.Map(alpha, header_alpha)
    map_pB = sunpy.map.Map(pB, header_pB)

    #### ************* write down the final Im into fits ****
    #hdu_I=fits.PrimaryHDU(I,header=header_I)
    #if os.path.isfile(file2write_I):
    #    print("% L3_polariz.2: Removing existing file "+file2write_I)
    #    os.remove(file2write_I)
    #print("% L3_polariz.2: Writing "+file2write_I)
    #hdu_I.writeto(file2write_I)
    #
    #hdu_pB=fits.PrimaryHDU(pB,header=header_pB)
    #if os.path.isfile(file2write_pB):
    #    print("% L3_polariz.2: Removing existing file "+file2write_pB)
    #    os.remove(file2write_pB)
    #print("% L3_polariz.2: Writing "+file2write_pB)
    #hdu_pB.writeto(file2write_pB)
    #
    #hdu_alpha=fits.PrimaryHDU(alpha,header=header_alpha)
    #if os.path.isfile(file2write_alpha):
    #    print("% L3_polariz.2: Removing existing file "+file2write_alpha)
    #    os.remove(file2write_alpha)
    #print("% L3_polariz.2: Writing "+file2write_alpha)
    #hdu_alpha.writeto(file2write_alpha)

    maps_dict = {
        'I': map_I,
        'pB': map_pB,
        'AoLP': map_alpha
    }

    return maps_dict

