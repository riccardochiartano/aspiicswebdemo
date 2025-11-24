from astropy.io import fits
import astropy.units as u
import os
import numpy as np
import sunpy.map
import streamlit as st
from scipy import ndimage
from pathlib import Path

from logic.process import calculate_darkbias
from logic.utils import calibrate, get_filter

import logic.rob.aspiics_detector  as det
import logic.rob.aspiics_optics    as optics
import logic.rob.aspiics_get_opse  as opse
import logic.rob.parameters        as par

base_dir = Path(__file__).resolve().parent.parent


def read_fits_image_array(filename):
    """See https://docs.astropy.org/en/stable/io/fits/ for more info"""
    with fits.open(filename, do_not_scale_image_data=True) as hdul:    
       imagedata = hdul[0].data
       header    = hdul[0].header
    return imagedata, header

def read_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data

def read_fits_2(path):
    with fits.open(path) as hdul:
        return hdul[1].data

def read_fits_header(path):
    with fits.open(path) as hdul:
        return hdul[0].header
    
def calibrate_rob(
        filename, 
        calib_files,
        calib_files_name,
        unit=None,
        diff='None', 
        save_diff=False, 
        cal=os.path.join(base_dir, 'resources', 'rob_calib_data', 'calibr_data.json.real'),
        #cal='/resources/rob_calib_data/calibr_data.json.real',
        filter='None',
        outputdir='./output/',
        mark_io=False,
        mark_sun=False,
        manyfiles = False
    ):
    
    Im, header = read_fits_image_array(filename)
    
    if filter != "None":
        filter=filter.strip()
        header.set('FILTER',filter)
        #print("%    Forcing filter **",filter,"**",sep='')
    else:
        filter=header['FILTER']            # filter='Wideband' # should be retrieved from the header
    #print("filter :  **",filter,"**")
    if 'Unknown' in filter:
        filter='Fe XIV'
        header.set('FILTER',"Fe XIV")
        #print("%    'Unknown' is in fits header. Forcing to 'Fe XIV'")
    #print("%    Using filter:  **",filter,"**",sep='')
    t_exp =header['EXPTIME']           # t_exp =1.0        # should be retrieved from the header
    pixscale = header['CDELT1']        #
    if 'BLANK' not in header:
        header.set('BLANK',-32768)
    BLANK =header['BLANK']
    if 'BZERO' not in header:
        header.set('BZERO',0.0)
    if 'BSCALE' not in header:
        header.set('BSCALE',1.0)


    ##### *** this exists only in sergeis test data ***
    #DIFFR_filename = header['DIFFR1'] + '.' + header['DIFFR2']
    #print("% L2_MASTER. Filename: "+filename)
    #print("       header diffraction: "+DIFFR_filename)
    #DIFFR_filename="ID_phiF.z144348.A1048576pts.rho50.IO1748mmH.C2425.LA235.JJ4096.T00_S0000.cart9.fits"
    if diff != 'None':
        DIFFR_filename=diff.strip()
        #print("%    Using diffraction file:    ",DIFFR_filename,sep='')

    ### *** trying to match to git-version of l2_master ***
    #params = par.readparams("calibr_data.json",[])
    params=par.readparams(cal,[])
    #print("%    Using calibration file:    ",cal,sep='')

    #  read filter-specific parameters and put then into main dictionary
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

    ### *** this should be retrieved from the repository, as Level-1 data still has no this info *** 
    #Aphot =2.1718226e+10  # for WBF.  Here calculated as 2.08e+20[photon/s/cm2/sr] * T_f * dS_pix * Apert * QE_f * gain # T=0.393 QE=0.65
    ##Aphot =3.4428314e+8   # for Fe XIV                    3.742e+18[photon/s/cm2/sr] * T_f * dS_pix * Apert * QE_f * gain # T=0.369 QE=0.61
    ##Aphot =1.3899817e+9   # for He I                     1.548e+19[photon/s/cm2/sr] * T_f * dS_pix * Apert * QE_f * gain # T=0.338 QE=0.65


    #x_IO = 1024.5-1.0 # center of IO in pix, converting from FITS (1->2048) to python (0->2047) standard
    #y_IO = 1024.5-1.0 
    #R_IO = 1.748      # radius in mm
    #eta1  = 2.0       # additional margin for the saturation. Currently comes from nowhere !!!!
    #eta2  = 4.0       # margin for the comparing of optical signal with its median value
    #params = {'calib_data': {'Aphot': Aphot, 'x_IO': x_IO, 'y_IO': y_IO, 'R_IO': R_IO, 'eta1': eta1, 'eta2': eta2, 'gain': 0.119, 'readout_noise': 6.013}}

    #params['calib_data']['Aphot']=Aphot
    #params['calib_data']['x_IO']=x_IO
    #params['calib_data']['y_IO']=y_IO
    Aphot=params['calib_data']['Aphot']    
    x_IO = params['calib_data']['x_IO']
    y_IO = params['calib_data']['y_IO']
    R_IO = params['calib_data']['R_IO']
    eta1=params['calib_data']['eta1']  #=eta1
    eta2=params['calib_data']['eta2']  #=eta2
    header.set('CDELT1',params['calib_data']['pixscale'])
    header.set('CDELT2',params['calib_data']['pixscale'])


    Im_orig = Im      # temporary, to compare with the processed 
    nlcorr_msg="No nonlinearity" ; dc_msg="No dark current" ; bias_msg="No bias" ; flat_msg="No flat"

    ### $$$ ***      PLEASE DO NOT SPLIT THIS PART OF CODE INTO PIECES/FILES/PROCEDURES      *** $$$ ###
    ### $$$ ***  VEUILLEZ NE PAS DIVISER CETTE PARTIE DU CODE EN MORCEAUX/FILES/PROCEDURES   *** $$$ ###
    ### $$$ *** SPLITS DIT DEEL VAN DE CODE ALSTUBLIEFT NIET IN STUKKEN/BESTANDEN/PROCEDURES *** $$$ ###

    # ***************** this part should correspond to the lines 1--6 of the pseudo-code of DPM *******************
    filefake='test'           
    gain = det.gain(header,params)                                             # 
    dc,   dc_msg   = det.darkcurrent(header,calib_files)
    dc             = dc/gain                                                   # original calibration data is in [DN]. We need [el] here
    bias, bias_msg = det.bias(header,calib_files)
    if not(manyfiles):
        flat, flat_msg = det.flat(header,calib_files,calib_files_name)
    else:
        flat, flat_msg = det.flat_many(header,params)
    HotPixels      = det.hot_pixels(header,params,1.0)                               
    Vread = np.ones((2048,2048))*(det.readout_noise(header,params)/gain)**2    # readout noise variance in DN
    Vdc   = (dc*t_exp)  # np.sqrt(dc*t_exp)                                    # dark current variance, el/sec -> el         ?? -> sqrt()  ??
    #Vphot = np.full((2048,2048),1.0/gain)                                      # wrong: variance of photons = number of photons ~ Im/gain (since qe~1), but Im contains noise also
    Vphot = np.full((2048,2048),1.0)                                           # variance of photons = number of photons = numbers of photoelectrons. Which is calculated from Im in correspondent place
                                                                            # Vphot==1 and will go as a factor to the photovariance;
    #vign  = optics.vignetting(pixscale,x_IO,y_IO,R_IO,verbose=False)
    vign, vign_msg = optics.vignetting2(header,params)                         # vignetting takes into account variation of R_IO with polar angle
    BlankIdx = (Im == BLANK)

    # ***************** this part should correspond to the lines 8--25 of the pseudo-code of DPM *******************
    Imax = np.full((2048,2048),pow(2,14))                                 # the maximum value, 2D array. We need it to identify saturated pixels.
    Imax = (Imax - bias)/gain                                             # convert everything into el.
    Im   = (Im - bias)/gain
    Im,nlcorr_msg = det.get_nlcorr(Im,header,params)
    Var  = Vread + Vdc + np.multiply(Vphot,Im)*(Im > 0)                   # Variance map for the signal, 2D array
    Imax = Imax - eta1*np.sqrt(Var)                                       
    OvrExpIdx = Im > Imax                                                 # Boolean 2D array with overexposed pixels
    Im   = Im - dc*t_exp                                                  # In the original version the DC was subtracted before overexposed pixels, 
                                                                        #   but in this case at long t_exp many pixels within saturated zone were marked as normal

    #### Correction of hot pixels was moved after the diffraction subtraction. The diffraction pattern has too steep gradients
    #Med  = ndimage.median_filter(Im,size=(3,3))                           # Median-filtered array for bad/hot pixels identification
    #BadPixIdx = np.absolute(Im-Med) > eta2*np.sqrt(Var)      #eta2        # It fails and selects bright structures in the corona for synthetic and eclipse data
    #BadPixIdx_mask = np.zeros((2048,2048)) ; BadPixIdx_mask[BadPixIdx] = 1
    ###plt.imshow(BadPixIdx_mask,origin='lower') ; plt.colorbar() ; plt.show()
    #### Think about union of current BadPixIdx with repository-based HotPixels 
    ##Im[BadPixIdx] = Med[BadPixIdx] 
    ##Im[HotPixels] = Med[HotPixels]

    ### When the optical part is commented out we should multiply by gain (introduced during DC subtraction)
    ### Do not forget to comment vignetting below                                                    ###
    #Im = Im*gain
    #Im = np.divide(Im,flat)
    ### This is the optical part: radiometric calibration, subtraction of diffraction/ghost/scattering ###
    ### Do not forget to uncomment vignetting below                                                    ###
    Im   = np.divide(Im,            flat*(Aphot/gain*t_exp))           # here convert units to [MSB] (analog of [photon s-1 cm-2 sr-1]), as the gain was taken into account before
    Var  = np.divide(Var, np.square(flat*Aphot/gain*t_exp))
    tmpY,tmpZ = opse.aspiics_get_opse(Im,header,params,verbose=False,save_image=True)        # find position of the OPSE LEDs in the image
    #sys.exit('exiting after OPSE')
    #Im   = np.subtract(Im,optics.ghost(header,params))
    #Im   = np.subtract(Im,np.multiply(optics.scatter(header,params),vign))
    #Im   = np.subtract(Im,optics.diffraction(header,params,verbose=False))
    #Im   = np.subtract(Im,optics.diffraction(header,params,verbose=False,diff_filename=DIFFR_filename))       # old-style, diffraction file passed by parameters
    if save_diff:
        diff=optics.diffraction(header,params,verbose=False,diff_filename=DIFFR_filename)
        hdu=fits.PrimaryHDU(diff,header=header)
        diff_newfilename=os.path.basename(filename)
        diff_newfilename=diff_newfilename.replace("l0","l2")
        diff_newfilename=diff_newfilename+".diffraction.fits"
        print("%    Diffraction filename: **",diff_newfilename,"**",sep='')
        file2write = os.path.join(outputdir,diff_newfilename)
        print("%    Saving diffraction to "+file2write)
        hdu.writeto(file2write,checksum=True,overwrite=True)

    Med  = ndimage.median_filter(Im,size=(3,3))                           # Median-filtered array for bad/hot pixels identification
    BadPixIdx = np.absolute(Im-Med) > eta2*np.sqrt(Var)      #eta2        # It fails and selects bright structures in the corona for synthetic and eclipse data
    BadPixIdx_mask = np.zeros((2048,2048)) ; BadPixIdx_mask[BadPixIdx] = 1 ;    #plt.imshow(BadPixIdx_mask,origin='lower') ; plt.colorbar() ; plt.show()    # Show masked
    #plt.imshow(Med,origin='lower') ; plt.colorbar() ; plt.show()
    #### Think about union of the current BadPixIdx with the repository-based HotPixels 
    Im[BadPixIdx] = Med[BadPixIdx] 
    #Im[HotPixels] = Med[HotPixels]

    ### Vignetting                                                                                     ###
    #vign[vign < 1e-3] = 1e-3 ;  Im   = np.divide(Im,vign)

    # here we should mark overexposed pixels
    Im[OvrExpIdx] = np.inf #1.5e+30
    Im[BlankIdx]  = np.nan
    # finding min-max-mean-median
    BadIdx = OvrExpIdx | BlankIdx                                         # mask for all the bad pixels
    DATAMIN = np.min(Im[ ~ BadIdx])
    DATAMAX = np.max(Im[ ~ BadIdx])
    DATAMEAN = np.mean(Im[ ~ BadIdx])
    DATAMEDN = np.median(Im[ ~ BadIdx])

    if mark_io:
        xio1=np.rint(x_IO).astype(int)  ;  yio1=np.rint(y_IO).astype(int)
        Im[yio1,xio1-20:xio1+20] = DATAMAX
        Im[yio1-20:yio1+20,xio1] = DATAMAX
        print("Marking IO in the image: ", xio1, yio1)
        

    if mark_sun:
        CRPIX1=header['CRPIX1']-1.  ;  CRPIX2=header['CRPIX2']-1.  ;  CRVAL1=header['CRVAL1']  ;  CRVAL2=header['CRVAL2']  ;  CDELT1=header['CDELT1']  ;  CDELT2=header['CDELT2']
        CROTA=header['CROTA']       ;  PC1_1=header['PC1_1']       ;  PC1_2=header['PC1_2']    ;  PC2_1=header['PC2_1']  ;  PC2_2=header['PC2_2']
        xSun = PC1_1*(-CRVAL1)/CDELT1 + PC2_1*(-CRVAL2)/CDELT2 + CRPIX1  # this is manual transformation (x,y)^ = rot()^  @  (dx/CDELT1, dy/CDELT2)^ + (CRPIX1,CRPIX2)^
        ySun = PC1_2*(-CRVAL1)/CDELT1 + PC2_2*(-CRVAL2)/CDELT2 + CRPIX2  # following Thompson 2006,  ^ -- transposition, @ - matrix multiplication
        xSunI=np.rint(xSun).astype(int)
        ySunI=np.rint(ySun).astype(int)
        Im[ySunI,xSunI-40:xSunI+40] = DATAMAX
        Im[ySunI-40:ySunI+40,xSunI] = DATAMAX
        print("Marking solar center in the image: ", xSun, ySun)
        header.set("SOLC_X",float("{:.2f}".format(xSun+1)),'expected position of x-Sun center (1..2048)')
        header.set("SOLC_Y",float("{:.2f}".format(ySun+1)),'expected position of y-Sun center (1..2048)')
        

    ### $$$ *********                                                                  ********* $$$ ###
    ### $$$ *********                        OK, NOW YOU CAN DO IT                     ********* $$$ ###
    ### $$$ *********                                                                  ********* $$$ ###

    ### ************ updating keywords ************** ###
    del header['BLANK']
    del header['BZERO']
    del header['BSCALE']
    header.set("HISTORY", bias_msg)
    header.set("HISTORY", dc_msg)
    header.set("HISTORY", nlcorr_msg)
    header.set("HISTORY", flat_msg)
    #header.set("HISTORY", vign_msg)
    header.set("LEVEL", "L2")
    header.set('VERS_CAL', params['calib_data']['VERS_CAL'], "version of set of calibration files")
    header.set('BUNIT', "MSB", "obtained from [DN/s] dividing by A_PHOT")
    header.set('A_phot', Aphot, "["+params['calib_data']['Aphot_units']+"] mean radiometric sensitivity")
    header.set('X_IO', x_IO+1.0, "[pix] X position of the IO (1..2048)")
    header.set('Y_IO', y_IO+1.0, "[pix] Y position of the IO (1..2048)")
    header.set('R_IO', R_IO, "[mm] IO radius")
    header.set('CONV_PHO', params['calib_data']['CONV_PHO'], "[DN/s]/CONV_PHO gives photon/s/cm2/sr")
    header.set('CONV_WAT', params['calib_data']['CONV_PHO']/params['calib_data']['Photon_energy']/1.0e+4, "Conversion from DN/s to W/m2/sr")
    header.set('CONV_ERG', params['calib_data']['CONV_PHO']/(params['calib_data']['Photon_energy']*1e+7), "Conversion from DN/s to erg/s/cm2/sr")
    #header.set('DATAMIN', DATAMIN, "minimum valid physical value")
    #header.set('DATAMAX', DATAMAX, "maximum valid physical value")
    #header.set('DATAMEAN', DATAMEAN, "average pixel value across the image")
    #header.set('DATAMEDN', DATAMEDN, "median pixel value across the image")
    header.set('HISTORY',"MSB equals "+"{:10.4e}".format(params['calib_data']['MSB'])+" "+params['calib_data']['MSB_UNITS'])

    ### ********** calibrate to unit ************ ###
    if unit == 'DN/s':
        Im = Im * Aphot
        header.set('BUNIT', unit, "obtained from Digital Numbers measured")
    elif unit == 'ph/(s cm^2 sr)':
        conv_pho=params['calib_data']['CONV_PHO']
        Im = Im * Aphot / conv_pho
        header.set('BUNIT', unit, "obtained from [DN/s] dividing by CONV_PHO")

    ### ********** convert to 32bit float ************ ###
    Im = Im.astype(np.float32)


    #if __name__ == '__main__':
    #    parser = argparse.ArgumentParser()
    #    parser.add_argument("-i", "--input", action="store", dest="inputfile")
    #    parser.add_argument("-c", "--calibrationfile", action="store", dest="calibrationfile")
    #    parser.add_argument("-o", "--outputdir", action="store", dest="outputdir")
    #    parser.add_argument("-r", "--run_id", action="store", dest="run_id")
    #    parser.add_argument("--opse", action="store_true")
    #    parser.add_argument("--diffraction")
    #    parser.add_argument("--ghost")
    #    args = parser.parse_args()

    ### ********** setup output dir *******************
    #outputdir = './output/'
    #if os.path.exists(args.outputdir):
    #  outputdir = args.outputdir


    ### ************* write down the final Im into fits ****
    hdu=fits.PrimaryHDU(Im,header=header)
    filename = header['filename']
    filename=os.path.basename(filename)
    newname=filename.replace("l0","l2")
    newname=filename.replace("l1","l2")
    header['filename'] = newname
    #print("File to write: **",filename_parts[1],"**",filename_parts[2],"**")
    
    # don't save
    if False:
        file2write = os.path.join(outputdir,newname)
        if os.path.isfile(file2write):
            print("% L2_MASTER. Output file "+file2write+" exists. Removing it")
            os.remove(file2write)
        print("% L2_MASTER. Writing "+file2write)
        hdu.writeto(file2write,checksum=True)
    
    calibrated_map = sunpy.map.Map(Im, header)
    return calibrated_map
