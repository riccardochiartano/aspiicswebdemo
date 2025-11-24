from astropy.io import fits
import astropy.units as u
import numpy as np
import sunpy.map
import streamlit as st
import os
from pathlib import Path

from logic.process import calculate_darkbias, calculate_dark, calculate_bias
from logic.utils import calibrate, get_filter
from logic.rob import aspiics_misc as am
from logic.rob import aspiics_detector as det
from logic.rob import parameters as par

base_dir = Path(__file__).resolve().parent.parent


def read_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data

def read_fits_2(path):
    with fits.open(path) as hdul:
        return hdul[1].data

def read_fits_header(path):
    with fits.open(path) as hdul:
        return hdul[0].header
    
def merge_demodulation(base_files, unit):
    '''
    Effettua il merge di diverse sequenze, correggendole prima per dark+bias e flatfield.

    Args:
        base_files: dict con le chiavi come angoli ('0°', '60°', etc) e valori i filepath.
        unit: measure unit con cui calibrare le immagini

    Returns:
        tre mappe (0, 60, 120) dopo merging calibrate in 'unit'.

    to-do:  cambio header per dire che sono merged (tempo inizio e tempo fine e dire che sono calibrate in MSB)
            decidere come definire le mappe sig_..
    '''
    
    maps_finali = []
    file_bias_A = base_files['bias_A']
    file_bias_B = base_files['bias_B']
    file_dark_A2 = base_files['dark_A2']
    file_dark_B2 = base_files['dark_B2']
    file_dark_C2 = base_files['dark_C2']
    ak_files = [base_files['ak 0°'], base_files['ak 60°'], base_files['ak 120°']]

    bias_image_A = read_fits(file_bias_A)
    bias_image_B = read_fits(file_bias_B)
    dark_image_A2 = read_fits(file_dark_A2)
    dark_image_B2 = read_fits(file_dark_B2)
    dark_image_C2 = read_fits(file_dark_C2)
    aks = [np.array(read_fits(filename)) for filename in ak_files]

    params=par.readparams('./resources/rob_calib_data/calibr_data.json.real',[])

    # ak files must be only flat fields, calibration done separately
    if 'rob' not in ak_files[0]:
        aks[0] = aks[0]/1.01e-10
        aks[1] = aks[1]/1.00e-10
        aks[2] = aks[2]/0.906e-10

    # filtra ak < 1e-13
    #for i, ak in enumerate(aks):
    #    aks[i][ak<1e-13] = np.nan
    #MSB = 2.082e20

    for j, angle in enumerate(['files_0°', 'files_60°', 'files_120°']):
        
        files = base_files[angle]
        headers = [read_fits_header(filename) for filename in files]
        header = headers[0]
        
        gain = det.gain(header, params)
        #gain = 1
        
        images_noak = [np.array(read_fits(filename)) for filename in files]
        
        sat_masks = []
        final_image = np.zeros_like(images_noak[0])
        count_sat = np.zeros_like(images_noak[0])

        t_exps = [float(header['EXPTIME']) for header in headers]   # in sec

        saturated_value = 0.9*(2**14-1)

        # valori saturati e dark
        for i, image in enumerate(images_noak): 
            #dark_image = calculate_darkbias(bias_image, dark_image_A2, dark_image_B2, dark_image_C2, headers[i])
            dark_image = calculate_dark(dark_image_A2, dark_image_B2, dark_image_C2, headers[i]) / gain
            bias_image = calculate_bias(bias_image_A, bias_image_B, headers[i])
            images_noak[i][image>saturated_value] = np.nan
            images_noak[i] = (images_noak[i] - bias_image) / gain
            images_noak[i], _ =  det.get_nlcorr(images_noak[i],headers[i],params)
            images_noak[i] = images_noak[i] - dark_image
            images_noak[i][images_noak[i]<0] = np.nan               # forza i valori >0
            images_noak[i] = images_noak[i] * gain

        sigs = [np.sqrt(image) for image in images_noak]

        images = [image / aks[j] for image in images_noak]
        sat_masks = [(~np.isnan(image)) for image in images]
        
        for image, sat_mask, t_exp in zip(images, sat_masks, t_exps):
            final_image[sat_mask] += image[sat_mask]
            count_sat[sat_mask] += t_exp
        
        final_image = np.divide(final_image, count_sat)

        smap = sunpy.map.Map(final_image, header)

        filter = angle.split('_')[-1]
        calibrated_map = calibrate(smap, filter, unit)

        maps_finali.append(calibrated_map)
        #smap.save(f'/home/bob/Scuola/Dotts/aspiics/aspiics_gui/resources/{angle[:-1]}.fits', overwrite=True)

    return maps_finali

    
def merge(base_files, unit=None):
    '''
    Effettua il merge di diverse sequenze, correggendole prima per dark+bias e flatfield.

    Args:
        base_files: dict con le chiavi come files e valori i filepath.

    Returns:
        mappa dopo merging calibrate in DN.

    to-do:  cambio header per dire che sono merged (tempo inizio e tempo fine e dire che sono calibrate in MSB)
            decidere come definire le mappe sig_..
    '''
    
    file_bias_A = base_files['bias_A']
    file_bias_B = base_files['bias_B']
    file_dark_A2 = base_files['dark_A2']
    file_dark_B2 = base_files['dark_B2']
    file_dark_C2 = base_files['dark_C2']
    file_flat = base_files['flat']

    bias_image_A = read_fits(file_bias_A)
    bias_image_B = read_fits(file_bias_B)
    dark_image_A2 = read_fits(file_dark_A2)
    dark_image_B2 = read_fits(file_dark_B2)
    dark_image_C2 = read_fits(file_dark_C2)
    flat = read_fits(file_flat)

    files = base_files['files']
    headers = [read_fits_header(filename) for filename in files]
    header = headers[0]
    filter = header['FILTER']
    
    params=par.readparams(os.path.join(base_dir, 'resources', 'rob_calib_data', 'calibr_data.json.real'),[])
    if filter=='Fe XIV':
        params1=params['calib_data']['Fe XIV']
    elif filter=='He I':
        params1=params['calib_data']['He I']
    elif filter=='Wideband':
        params1=params['calib_data']['Wideband']
    elif filter=='Polarizer 0':
        params1=params['calib_data']['Polarizer 0']
    elif filter=='Polarizer 60':
        params1=params['calib_data']['Polarizer 60']
    elif filter=='Polarizer 120':
        params1=params['calib_data']['Polarizer 120']
    params['calib_data'].update(params1)
    gain = det.gain(header, params)
    #gain = 1

    images_noak = [np.array(read_fits(filename)) for filename in files]
    
    sat_masks = []
    final_image = np.zeros_like(images_noak[0])
    count_sat = np.zeros_like(images_noak[0])

    t_exps = [float(header['EXPTIME']) for header in headers]   # in sec

    saturated_value = 0.9*(2**14-1)

    # valori saturati e dark
    for i, image in enumerate(images_noak): 
        #dark_image = calculate_darkbias(bias_image, dark_image_A2, dark_image_B2, dark_image_C2, headers[i])
        dark_image = calculate_dark(dark_image_A2, dark_image_B2, dark_image_C2, headers[i]) / gain
        bias_image = calculate_bias(bias_image_A, bias_image_B, headers[i])
        images_noak[i][image>saturated_value] = np.nan
        images_noak[i] = (images_noak[i] - bias_image) / gain
        images_noak[i], _ =  det.get_nlcorr(images_noak[i],headers[i],params)
        images_noak[i] = images_noak[i] - dark_image
        images_noak[i][images_noak[i]<0] = np.nan               # forza i valori >0
        images_noak[i] = images_noak[i] * gain

    sigs = [np.sqrt(image) for image in images_noak]

    images = [image / flat for image in images_noak]
    sat_masks = [(~np.isnan(image)) for image in images]
    
    for image, sat_mask, t_exp in zip(images, sat_masks, t_exps):
        final_image[sat_mask] += image[sat_mask]
        count_sat[sat_mask] += t_exp
    
    final_image = np.divide(final_image, count_sat)

    if unit == 'DN/s':
        header.set('BUNIT', unit, "obtained from Digital Numbers measured")
    elif unit == 'ph/(s cm^2 sr)':
        conv_pho=params['calib_data']['CONV_PHO']
        final_image = final_image / conv_pho
        header.set('BUNIT', unit, "obtained from [DN/s] dividing by CONV_PHO")
    elif unit == 'MSB':
        Aphot=params['calib_data']['Aphot']
        final_image = final_image / Aphot
        header.set('BUNIT', unit, "obtained from [DN/s] dividing by Aphot")


    final_map = sunpy.map.Map(final_image, header)
    #final_map.meta['bunit'] = 'DN/s'
    
    if len(files) > 1:
        merged = True
    else:
        merged = False
    final_map.meta['filename'] = change_filename(header, merged)

    return final_map

def change_filename(header, merged):
    parts = header['filename'].split('_')
    parts[2] = 'l2'
    if merged:
        parts[3] = f"{header['cycle_id']}_merged"
    fname_change = '_'.join(parts)
    return fname_change
    

def merge_rob(
        files,
        outputdir='./output/',
        docenter=True,
        coalign=False,
        save_shifted=False,
        CRVAL1=None,
        CRVAL2=None
    ):

    forceCRVAL1 = False
    forceCRVAL2 = False
    if CRVAL1 != None:
        forceCRVAL1 = True
        CRVAL1 = CRVAL1
        print("   Using CRVAL1 value {:.2f} arcsec".format(CRVAL1))
    if CRVAL2 != None:
        forceCRVAL2 = True
        CRVAL2 = CRVAL2
        print("   Using CRVAL2 value {:.2f} arcsec".format(CRVAL2))


    nfiles=len(files)
    if nfiles == 1:
        fileA=files[0]
        dataA,headerA=read_fits_image_array(fileA)
        data01=dataA  ;  header01=headerA  ;  file01=fileA  ;  header1=header01  ;  file1=fileA  ;  header10=headerA  ;  file10=fileA
    elif nfiles == 2:
        fileA=files[0]
        fileB=files[1]
        dataA,headerA=read_fits_image_array(fileA)
        dataB,headerB=read_fits_image_array(fileB)
        exptimes=[headerA['EXPTIME'],headerB['EXPTIME']]
        idx=np.argsort(exptimes)
        data3=[dataA,dataB]         ;  data3=np.array(data3)  # this is a numpy array
        header3=[headerA,headerB]   ;                         #  this is a list
        data3=data3[idx,:,:]        ;  data01=np.squeeze(data3[0,:,:])   ;   data1=np.squeeze(data3[1,:,:])
        header01=header3[idx[0]]    ;  header1=header3[idx[1]]
        file01 = files[idx[0]]      ;  file1=files[idx[1]]                                        ;  header10=header1  ;  file10=file1
    elif nfiles == 3:
        fileA=files[0]
        fileB=files[1]
        fileC=files[2]
        dataA,headerA=read_fits_image_array(fileA)
        dataB,headerB=read_fits_image_array(fileB)
        dataC,headerC=read_fits_image_array(fileC)
        exptimes=np.array([headerA['EXPTIME'],headerB['EXPTIME'],headerC['EXPTIME']])
        idx=np.argsort(exptimes)
        data3=[dataA,dataB,dataC]           ;  data3=np.array(data3)  # this is a numpy array
        header3=[headerA,headerB,headerC]                             #  this is a list
        data3=data3[idx,:,:]       ;   data01=np.squeeze(data3[0,:,:])  ;  data1=np.squeeze(data3[1,:,:])  ;  data10=np.squeeze(data3[2,:,:])
        header01=header3[idx[0]]   ;   header1=header3[idx[1]]          ;  header10=header3[idx[2]]
        file01 = files[idx[0]]     ;   file1=files[idx[1]]              ;  file10=files[idx[2]]
    else:
        st.error("% L3_Merge: we expect 1--3 input files ... Exiting!")
        exit()

    if coalign == True:
    #    if nfiles != 3:
    #        print("   to co-align we need 3 input files. Exiting ...")
    #        exit()
        if docenter == True:
            print("   Centering [default?] and co-aligning of images are requested. We do co-aligning only.")
            docenter = False

    # receive input filenames from the command line. We assume original exposure time was 1<2<3, which is encoded by 01, 1 and 10 in var names
    #file01 =args.file1  #sys.argv[1]    #file10='tile_map/ASPIICS_synthetic_T30S3000_10.0sec_filterWB.fits'
    #file1  =args.file2  #sys.argv[2]    #file1='tile_map/ASPIICS_synthetic_T30S3000_01.0sec_filterWB.fits'
    #file10 =args.file3  #sys.argv[3]    #file01='tile_map/ASPIICS_synthetic_T30S3000_00.1sec_filterWB.fits'
    print("    Input files:")
    print("     file01: ",file01)
    if nfiles >= 2:
        print("     file1:  ",file1)
    if nfiles == 3:
        print("     file10: ",file10)


    if forceCRVAL1:
        header01['CRVAL1']=CRVAL1       ;  header01.set('HISTORY','Forced CRVAL1 {:.2f} arcsec'.format(CRVAL1))
        if nfiles >= 2:
            header1['CRVAL1']=CRVAL1    ;  header1.set('HISTORY','Forced CRVAL1 {:.2f} arcsec'.format(CRVAL1))
        if nfiles == 3:
            header10['CRVAL1']=CRVAL1   ;  header10.set('HISTORY','Forced CRVAL1 {:.2f} arcsec'.format(CRVAL1))

    if forceCRVAL2:
        header01['CRVAL2']=CRVAL2       ;  header01.set('HISTORY','Forced CRVAL2 {:.2f} arcsec'.format(CRVAL2))
        if nfiles >= 2:
            header1['CRVAL2']=CRVAL2    ;  header1.set('HISTORY','Forced CRVAL2 {:.2f} arcsec'.format(CRVAL2))
        if nfiles == 3:
            header10['CRVAL2']=CRVAL2   ;  header10.set('HISTORY','Forced CRVAL2 {:.2f} arcsec'.format(CRVAL2))

    #data01, header01=am.read_fits_image_array(file01)
    #data1,  header1 =am.read_fits_image_array(file1)
    #if nfiles == 3:
    #    data10, header10=am.read_fits_image_array(file10)

    if docenter == True: 
        print("    Re-centering images:")
        print("  ************************** file01 ************************** ")
        data01 = am.rotate_center1(data01,header01,verbose=True)
        headerRef=header01.copy()
        if nfiles >= 2:
            print("  ************************** file1 ************************** ")
            data1 = am.rotate_center1(data1,header1,verbose=True)
            headerRef=header1.copy()
        if nfiles == 3:
            print("  ************************** file10 ************************** ")
            data10 = am.rotate_center1(data10,header10,verbose=True)
            headerRef=header10.copy()

    if coalign == True:
        if nfiles == 2:
            print("  ************************** file01 ************************** ")
            data01 = am.shift_image(data01,header01,header1,verbose=True)
            headerRef=header1.copy()
        if nfiles == 3:
            print("  ************************** file01 ************************** ")
            data01 = am.shift_image(data01,header01,header10,verbose=True)
            print("  ************************** file1 ************************** ")
            data1 = am.shift_image(data1,header1,header10,verbose=True)
            headerRef=header10.copy()
    

    # Merging two or three images. By default take the data from data10, but substitute the NaN/Inf pixels via the pixels from smaller exposure file
    if nfiles == 3:
        Im_out = data10
        mask = ~np.isfinite(Im_out)
        Im_out[mask] = data1[mask]
        mask = ~np.isfinite(Im_out)
        Im_out[mask] = data01[mask]
    elif nfiles == 2:
        Im_out = data1
        mask = ~np.isfinite(Im_out)
        Im_out[mask] = data01[mask]
    else:
        Im_out = data01

    BadMask = Im_out < 0
    Im_out[BadMask] = 1e-11
    BadMask = ~np.isfinite(Im_out)
    #Im_out[BadMask] = np.nan
    DATAMIN = np.min(Im_out[ ~ BadMask])
    DATAMAX = np.max(Im_out[ ~ BadMask])
    DATAMEAN = np.mean(Im_out[ ~ BadMask])
    DATAMEDN = np.median(Im_out[ ~ BadMask])


    ### ********** convert to 32bit float ************ ###
    Im_out = Im_out.astype(np.float32)

    headerM = headerRef.copy()
    headerM.set('LEVEL','L3','data processing level')
    headerM.set('PROD_ID','Merged',after='LEVEL')
    headerM.set('DATAMIN', DATAMIN, "minimum valid physical value")
    headerM.set('DATAMAX', DATAMAX, "maximum valid physical value")
    headerM.set('DATAMEAN', DATAMEAN, "average pixel value across the image")
    headerM.set('DATAMEDN', DATAMEDN, "median pixel value across the image")
    headerM.set('CREATOR',"Sergei's l3_merge", "FITS creation software")

    #header1.set('CRPIX1',1024.5,'[pix] (1..2048) The image has been ...')
    #header1.set('CRPIX2',1024.5,'[pix]   re-centered')
    #header1.set('CROTA',0.0,"[deg] The image has been de-rotated")
    #header1.set('CRVAL1',0.0,"[arcsec] reference value on axis 1")
    #header1.set('CRVAL2',0.0,"[arcsec] reference value on axis 1")
    #header1.set('PC1_1',1.0)
    #header1.set('PC1_2',0.0)
    #header1.set('PC2_1',0.0)
    #header1.set('PC2_2',1.0)
    #header1.set('FLT_TEST',float("{:.3f}".format(109.123456568)))

    headerM.set('HISTORY',"File01: "+os.path.basename(file01))
    headerM.set('HISTORY','  IO position {:.2f}/{:.2f}'.format(header01['X_IO'],header01['Y_IO']))
    if nfiles >= 2:
        headerM.set('HISTORY',"File1: " +os.path.basename(file1))
        headerM.set('HISTORY','  IO position {:.2f}/{:.2f}'.format(header1['X_IO'],header1['Y_IO']))
    if nfiles == 3:
        headerM.set('HISTORY',"File 10 "+os.path.basename(file10))
        headerM.set('HISTORY','  IO position {:.2f}/{:.2f}'.format(header10['X_IO'],header10['Y_IO']))

    ## ************* write down the final Im into fits ****
    hdu=fits.PrimaryHDU(Im_out,header=headerM)
    #newname=os.path.splitext(os.path.basename(file10))[0]+'.merged.fits'
    newname=os.path.splitext(headerM['FILENAME'])[0]+'.merged.fits'
    newname=newname.replace("l2","l3")
    #file2write = './output/'+os.path.splitext(os.path.basename(file1))[0]+'.merged.fits'
    #file2write = os.path.join(outputdir,newname)
    #if os.path.isfile(file2write):
    #    print("% L3_Merge. Removing existing file "+file2write)
    #    os.remove(file2write)
    #print("% L3_Merge. Writing "+file2write)
    #hdu.writeto(file2write)
    
    headerM.set('FILENAME', newname)

    smap = sunpy.map.Map(Im_out, headerM)
    return smap

def read_fits_image_array(filename):
    """See https://docs.astropy.org/en/stable/io/fits/ for more info"""
    with fits.open(filename, do_not_scale_image_data=True) as hdul:    
       imagedata = hdul[0].data
       header    = hdul[0].header
    return imagedata, header
