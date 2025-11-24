import numpy as np
from astropy.io import fits
from scipy import interpolate
import os

##### older version - in DN ###########
#def darkcurrent(filename,**kwargs):
#    """Returns dark current 2D map (currently default), units are [DN/sec]"""
#    ### currently we are using sshestov test data, which is expressed in el./sec
#    filename="./calibration/detector_dark_current.fits"
#    print("% aspiics_detector.darkcurrent: using standard file "+filename)
#    print("%      which has units [el/sec]. Converting it to [DN/sec] using gain")
#    with fits.open(filename) as hdul:             
#       imagedata = hdul[0].data
#    ### hence need to convert el/sec into DN/sec
#    imagedata = imagedata*gain()    
#
#    return imagedata

def darkcurrent(header,calib_files,**kwargs):
    """Returns dark current 2D map [DN/sec], using model
          dark = A + B*temp + C*temp^2 (from processing of the on-ground calibration data"""
    
    version_msg = "aspiics_detector.darkcurrent() v1.0 ASPIICSWEB"
    
    #fileA = os.path.join(params["calib_data"]["dark_current"],"dark_A2.fits")  
    #fileB = os.path.join(params["calib_data"]["dark_current"],"dark_B2.fits")  
    #fileC = os.path.join(params["calib_data"]["dark_current"],"dark_C2.fits")  
    fileA = calib_files['dark_A2']  
    fileB = calib_files['dark_B2']  
    fileC = calib_files['dark_C2']  

    #print("% aspiics_detector.darkcurrent: using files dark_{A,B,C}2.fits from the")
    #print("   "+params["calib_data"]["dark_current"]+" folder. Resulting units [DN/sec].")
   
    with fits.open(fileA, do_not_scale_image_data=True) as hdul:             
      darkA = hdul[0].data
    with fits.open(fileB, do_not_scale_image_data=True) as hdul:             
      darkB = hdul[0].data
    with fits.open(fileC, do_not_scale_image_data=True) as hdul:             
      darkC = hdul[0].data

    if 'APS_TEMP' in header:
      temp=header["APS_TEMP"]
      print("   APS_TEMP="+'{:05.2f}'.format(temp)+" C")
    else:
      temp=-13.0
      print("   no APS_TEMP found in the header, using standard "+'{:05.2f}'.format(temp)+" C")

    dark = darkA + darkB*temp + darkC*temp**2
    return dark, version_msg


def bias(header,calib_files,**kwargs):
    """Returns bias 2D map [DN], using the model  bias = A + B*temp
          The 2D-arrays A and B are derived during processing of the calibration data"""

    version_msg = "aspiics_detector.bias() v1.0 ASPIICSWEB"
    
    #fileA =os.path.join(params["calib_data"]["bias"],"bias_A.fits")
    #fileB =os.path.join(params["calib_data"]["bias"],"bias_B.fits")
    fileA = calib_files['bias_A']  
    fileB = calib_files['bias_B']

    #print("% aspiics_detector.bias:        using files bias_{A,B}.fits from the ")
    #print("   "+params["calib_data"]["bias"]+" folder. Resulting units [DN].")
    
    with fits.open(fileA, do_not_scale_image_data=True) as hdul:             
      biasA = hdul[0].data
      biasA_header = hdul[0].header
    with fits.open(fileB, do_not_scale_image_data=True) as hdul:             
      biasB = hdul[0].data
 
    if 'VERSION' in biasA_header:
      version_msg = version_msg+"; v_cal: "+biasA_header['VERSION']
 
    if 'APS_TEMP' in header:
      temp=header["APS_TEMP"]
      print("   APS_TEMP="+'{:05.2f}'.format(temp)+" C")
    else:
      temp=-13.0
      print("   no APS_TEMP found in the header, using standard "+'{:05.2f}'.format(temp)+" C")

    texp = header.get("EXPTIME",0.0)
    
    bias = biasA + biasB*temp    
    #if (temp < 0.0) and (texp < 3.0):
    #  bias=bias+6.0

    return bias, version_msg


#def flat(filter,**kwargs):
#    """Returns flat field 2D map (currently default), expressed in fraction"""
#    # can receive optional input:
#    ## filter - used filter
#    filename="./calibration_r/detector_flat.fits"
#    print("% aspiics_det_flat:             using standard file "+filename)
#    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
#       imagedata = hdul[0].data
#    return imagedata

def flat(header,calib_files,calib_files_name,**kwargs):
    """Returns flat field 2D map, dimensionless"""
    
    version_msg = "aspiics_detector.flat() v1.0 ASPIICSWEB"
  
    #filter=header['FILTER']
    #if filter=='Fe XIV':
    #   file='flatfield_Fe.fits'
    #elif filter=='He I':
    #   file='flatfield_He.fits'
    #elif filter=='Wideband':
    #   file='flatfield_WB.fits'
    #elif filter=='Polarizer 0':
    #   file='flatfield_P1.fits'
    #elif filter=='Polarizer 60':
    #   file='flatfield_P2.fits'
    #elif filter=='Polarizer 120':
    #   file='flatfield_P3.fits'
    #
    #print("% aspiics_detector.flat:        using file "+file)
    #fullfile=os.path.join(params['calib_data']['flat'],file)
    fullfile=calib_files['flat']

    with fits.open(fullfile, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data

    return imagedata, version_msg+" with "+calib_files_name['flat']

def flat_many(header,params,**kwargs):
    """Returns flat field 2D map, dimensionless"""
    
    version_msg = "aspiics_detector.flat() v1.0"
  
    filter=header['FILTER']
    if filter=='Fe XIV':
       file='flatfield_Fe.fits'
    elif filter=='He I':
       file='flatfield_He.fits'
    elif filter=='Wideband':
       file='flatfield_WB.fits'
    elif filter=='Polarizer 0':
       file='flatfield_P1.fits'
    elif filter=='Polarizer 60':
       file='flatfield_P2.fits'
    elif filter=='Polarizer 120':
       file='flatfield_P3.fits'
    
    print("% aspiics_detector.flat:        using file "+file)
    fullfile=os.path.join(params['calib_data']['flat'],file)
    #fullfile=calib_files['flat']

    with fits.open(fullfile, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data

    return imagedata, version_msg+" with "+file



def gain(header,params,**kwargs):
    """Returns gain of the detector, units are [DN/el.-]"""
    res=params['calib_data']['gain'] 
    return res


def hot_pixels(header,params,value,**kwargs):
    """Returns 2D array with marked hot pixels"""
    #filename="./calibration_r/hotpixels_list.fits"
    filename=params['calib_data']['hotpixel'] 
    print("% aspiics_detecotr.hot_pixels:  reading file "+filename,end='')
    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       data = hdul[0].data
    x_idx=data[0,:]
    y_idx=data[1,:]
    print(". There are "+'{:0.0f}'.format(len(x_idx))+" pixels.")
    image=np.zeros((2048,2048),dtype=bool)
    image[y_idx,x_idx]=True
    return image


def get_nlcorr(Im,header,params,**kwargs):
    """Returns image Im with corrected nonlinearity. Input Im - in [electron] after bias subtraction.
         The conversion function can be either in [DN] or [electron]."""

    filename=params['calib_data']['nonlin'] #"./calibration_r/detector_nonlin.fits"

    version_msg = "aspiics_detector.get_nlcorr() v1.0 with "+os.path.basename(filename)

    print("% aspiics_getector.get_nlcorr:  using file "+filename+",")
    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       response  = hdul[0].data
       nl_header = hdul[0].header
     
    print("    which is a transfer function from ["+nl_header['UNITS']+"] to ["+nl_header['UNITS']+"]. We need [electron] to [electron],", end="")
    if nl_header['UNITS'].find("electron") != -1:
        print(" thus no conversion is needed.")
        x = response[0,:]
        y = response[1,:]
    else:
        print(" thus we need to convert by dividing by gain.") 
        x = response[0,:]/gain(header,params)
        y = response[1,:]/gain(header,params)

    # The pixels with too high values (outside y range) are normally extrapolated ... and get negative values
    # We are creating a mask and restore original signal for such pixels. They will be marked afterwards as overexposed.
    OvrExpIdx = Im > np.max(y)
 
    tck = interpolate.splrep(y,x,s=0)
    Im1 = interpolate.splev(Im, tck, der=0)    
    
    if np.count_nonzero(OvrExpIdx)>0:
        print("    We found "+'{:0.0f}'.format(np.count_nonzero(OvrExpIdx))+" pixels with the value outside available data ("+'{:0.1f}'.format(np.max(y))+" [electron]). We keep them unmodified, they will be marked later on as overexposed.")
        Im1[OvrExpIdx]=Im[OvrExpIdx]

    return Im1, version_msg


def readout_noise(header,params,**kwarg):
    """Returns std.dev. of the read-out noise, units are [DN]"""
    res=params['calib_data']['readout_noise'] #6.013  ### res=50.53
    return res
   

def qe(filter):
    """Returns average (per whole image) quantum efficiency of the detector
       Units are [el./photon]. Filter-dependent
    """
    if filter == 'Wideband':
      return 0.65
    if filter == 'Fe XIV':
      return 0.61
    if filter == 'He I':
      return 0.65
    if filter == 'Polarizer 0':
      return 0.65
    if filter == 'Polarizer 60':
      return 0.65
    if filter == 'Polarizer 120':
      return 0.65
