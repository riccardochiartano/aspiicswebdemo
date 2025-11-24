import numpy as np
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt
import os

def read_fits_image_array(filename):
    """See https://docs.astropy.org/en/stable/io/fits/ for more info"""
    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data
       header    = hdul[0].header
    return imagedata, header


def aspiics_header2tiltshift(header,params):
    """Calculates the solar shift and the telescope tilt from
       the header parameters. Following the notations from Shestov&Zhukov 2018
       Returns: Tilt_x,Tilt_y,Shift_x,Shift_y in arcsec as would be projected to the detector"""
    OPLF_X=header['OPLF_X'] ; OPLF_Y=header['OPLF_Y'] ; OPLF_Z=header['OPLF_Z']  # m, position of the OPLF center wrt CPLF center
    CRPIX1=header['CRPIX1']-1.0 ; CRPIX2=header['CRPIX2']-1.0                    # pix, location of the solar center in the image. Convert to range 0..2047 (retrieved from the header)
    pixscale = header['CDELT1']                                                  # arcsec/pixel, pixel plate scale
    x_IO=params['calib_data']['x_IO']                                            # pix, coordinates of the IO in the image. Here should be in the range 0..2047
    y_IO=params['calib_data']['y_IO']

    print("%       header2tiltshift: OPLF_X="+'{:9.4f}'.format(OPLF_X)+"; OPLF_Y="+'{:8.4f}'.format(OPLF_Y)+"; OPLF_Z="+'{:8.4f}'.format(OPLF_Z)+"; [m] everywhere")
    print("%       header2tiltshift: CRPIX1="+'{:8.3f}'.format(CRPIX1)+";  CRPIX2="+'{:8.3f}'.format(CRPIX2)+"; [pix] using 0..2047")
    #                                   '{:02.0f}'.format(Tilt_x)   and before I used str(OPLF_X)
    ISD1 = np.sqrt(OPLF_X**2 + OPLF_Y**2 + OPLF_Z**2)

    # By its definition in Shestov&Zhukov 2018, the main axis connects the center of the aperture and the center of the OPLF.
    # Thus the telescope tilt is the angle between the CPLF and the main axis.
    # The coordinates of the OPLF center on the detector are:
    x_OPLF = x_IO + np.degrees(np.arctan(OPLF_Z/ISD1))*3600.0/pixscale           #         x_image = - (z_DRF); image-x-axis is reversed WRT z_DRF
    y_OPLF = y_IO - np.degrees(np.arctan(OPLF_Y/ISD1))*3600.0/pixscale           # however y_image = + (y_DRF). Thus +y in DRF points towards negative in solar RF
    Tilt_x = np.degrees(np.arctan(OPLF_Z/ISD1))*3600.0                          # arcsec, =def= (x_OPLF - x_IO)*pixscale. Positive direction of CPLF rotation around Oy shifts projected position of the OPLF to the right
    Tilt_y = np.degrees(np.arctan(OPLF_Y/ISD1))*3600.0                          # arcsec, =def= (y_OPLF - y_IO)*pixscale

    print("%       header2tiltshift: X_OPLF="+'{:8.3f}'.format(x_OPLF)+";  Y_OPLF="+'{:8.3f}'.format(y_OPLF)+"; [pix] (as projected on detector)")
 
    # By its definition in Shestov&Zhukov 2018, the main axis connects the center of the aperture and the center of the OPLF
    Shift_x = (CRPIX1 - x_OPLF)*pixscale
    Shift_y = (CRPIX2 - y_OPLF)*pixscale
    return Tilt_x, Tilt_y, Shift_x, Shift_y


def ghost(header,params):
    """Returns either pre-calculated ghost image (if available) or zero array
    """
    #path = "/home/sergeis/Projects/ASPIICS/Pipeline/output/"
    path=params['config']['ghost_images']
    ghost_file = header['GHOST'] ; ghost_file=ghost_file.strip()
    if ghost_file.lower() != 'none':
       if os.path.isfile(path+ghost_file):
           print("%Ghost: reading from file "+path+ghost_file)
           data, header = read_fits_image_array(path+ghost_file)    
       else:
           print("%Ghost: the file "+path+ghost_file+" is unavailable. Returning zeros")
           data=np.zeros((2048,2048))
    else:
        print("%Ghost: no ghosts were used in the input file. Returning zeros")
        data=np.zeros((2048,2048))
    return data


def scatter(header,params):
    """Returns scattering image using input data, current RSun and interpolation"""
    scat_file="/home/sergeis/Projects/ASPIICS/my_diffraction_Oprim_Rsun/aspiics_scattering_R16.fits"
    RSun_arcmin = header['RSUN_ARC']/60.0
    
    print("% aspiics_optics.scatter: filename = "+scat_file)
    data, header = read_fits_image_array(scat_file)    
    
    RSun_arr =[15.73,    15.79,    15.85,    15.91,    15.97,    16.03,   16.09,   16.15,   16.21,   16.26]
    coeff_arr=[ 0.757576, 0.797980, 0.838384, 0.888889, 0.939394, 1.01010, 1.08687, 1.18182, 1.29798, 1.45253]
   
    coeff=np.interp(RSun_arcmin,RSun_arr,coeff_arr)
    print("% aspiics_optics.scatter: RSun = "+str(RSun_arcmin)+" arcmin;  coeff = "+str(coeff))
   
    return data*coeff


def diffraction(header, params, **kwargs):
##def diffraction(filename, pixscale, **kwargs):
    """Returns diffraction pattern read from filename (calculated with Sergei's Fortran code)
       rescaled to the pixscale"""
#    path = "/home/sergeis/Projects/ASPIICS/my_diffraction_Fortran_param/diffr_light_images/"
    path = params['config']['diffr_light_images']
#    ### temporary ####
#    #filename = "ID_phiF.z144348.A1048576pts.rho50.IO1748mmH.C2425.LA235.JJ4096.T00_S0000.cart9.fits"
    
#    filename = path+filename
#    x_IO=1023.5                    # might not be needed, see xnew,ynew
#    y_IO=1023.5 
#    print("% aspiics_optics.diffraction: x_IO="+str(x_IO)+"; y_IO="+str(y_IO)+";  pixscale="+str(pixscale))

    x_IO=params['calib_data']['x_IO']                    # might not be needed, see xnew,ynew
    y_IO=params['calib_data']['y_IO']
    pixscale=header['CDELT1']                            # used to be =2.82

    print("% aspiics_optics.diffraction: x_IO="+str(x_IO)+"; y_IO="+str(y_IO)+";  pixscale="+str(pixscale))
    
    diff_filename=kwargs.get('diff_filename', 'none')
    if diff_filename != 'none' :
       print("% aspiics_optics.diffraction: received diffraction filename as a parameter.")
       filename = diff_filename
    else: 
       Tilt_x, Tilt_y, Shift_x, Shift_y = aspiics_header2tiltshift(header,params)
       print("% aspiics_optics.diffraction: received following tilts/shifts from the header:")
       print("%                        Tilt_x="+'{:6.3f}'.format(Tilt_x)+"; Tilt_y="+'{:6.3f}'.format(Tilt_y)+"; Shift_x="+'{:6.3f}'.format(Shift_x)+"; Shift_y="+'{:6.3f}'.format(Shift_y))
       Tilt_x = int(np.rint(Tilt_x/5.0))*5 
       Tilt_y = int(np.rint(Tilt_y/5.0))*5
       Shift_x = int(np.rint(Shift_x/5.0))*5
       Shift_y = int(np.rint(Shift_y/5.0))*5          # round tilt and shift 
       print("%       after rounding:  Tilt_x="+str(Tilt_x)+"; Tilt_y="+str(Tilt_y)+"; Shift_x="+str(Shift_x)+"; Shift_y="+str(Shift_y))
       TS=("T"+'{:02.0f}'.format(Tilt_x)+"_S"+'{:02.0f}'.format(Shift_x)+'{:02.0f}'.format(Shift_y))
       print("%       using TS string: "+TS)
       filename=("ID_phiF.z144348.A1048576pts.rho50.IO1748mmH.C2425.LA235.JJ4096."+TS+".cart9.fits")

    print("%       filename = "+filename)
    print("%       path     = "+path)
    filename=path+filename 

    Im, headerD = read_fits_image_array(filename)
    JJ      = headerD['JJ']         # the diffraction image size is JJ x JJ
    pixsizeB= headerD['PSCALE_B']   # pixel size in mkm in B in diffraction image
    fPO     = 330.341               # focal length of the PO in diffraction image, should also be read from the diffraction header

    pixscaleB = np.degrees(np.arctan(pixsizeB*1e-3/fPO))*3600.0 # pixel plate scale in B 

    x = (np.linspace(0,JJ-1,num=JJ)-(JJ-1)/2.0)*pixscaleB  # original coordinates, centered around (JJ-1)/2.0
    y = (np.linspace(0,JJ-1,num=JJ)-(JJ-1)/2.0)*pixscaleB
    f = interpolate.interp2d(x, y, Im, kind='linear')
    
    xnew = (np.linspace(0,2047,num=2048)-x_IO)*pixscale   # new coordinates in the ASPIICS image
    ynew = (np.linspace(0,2047,num=2048)-y_IO)*pixscale   # might need to have x_IO->1023.5 if the displacement of IO was already taken into account
    znew = f(xnew, ynew)
   
    ####### testing #######
    ## by manual comparison with IDL sshestov has established, that IDL's interpolate() with bilinear interpolation (default, no cubic keyword)
    ## matches best to the interp2d(kind='linear')
    #fc= interpolate.interp2d(x, y, Im, kind='cubic')
    #fq= interpolate.interp2d(x, y, Im, kind='quintic')
    #flog = interpolate.interp2d(x, y, np.log(Im), kind='linear')
    #zcnew = fc(xnew, ynew)
    #zqnew = fq(xnew, ynew)
    #zlognew = np.exp(flog(xnew, ynew))
    #hdu=fits.PrimaryHDU(znew)
    #hdu.writeto("test_diff_py_lin.fits")
    #hdu=fits.PrimaryHDU(zcnew)
    #hdu.writeto("test_diff_py_cub.fits")
    #hdu=fits.PrimaryHDU(zqnew)
    #hdu.writeto("test_diff_py_quin.fits")
    #hdu=fits.PrimaryHDU(zlognew)
    #hdu.writeto("test_diff_py_loglin.fits")
    ####### testing #######


    verbose=kwargs.get('verbose', False)
    if verbose:
        RSun=16.0*60.0
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(x, Im[:,2048], 'r-', label='Pre-calculated')        # 'ro-'
        ax.plot(xnew, znew[:,1024], 'b-', label='Interpolated, 2048x2048 linear')
        #ax.plot(xnew, zcnew[:,1024], 'g-', label='Interpolated, 2048x2048 cubic')
        #ax.plot(xnew, zqnew[:,1024], 'm-', label='Interpolated, 2048x2048 quintic')
        #ax.plot(xnew, zlognew[:,1024], 'c-', label='Interpolated, 2048x2048 log-linear')
        ax.set_xlabel('Coordinate, arcsec')
        ax.set_ylabel('Diffraction, MSB')
        ax.set_title('Diffraction')
        plt.yscale('log')
        plt.xlim(600,1700)
        plt.legend()
        secax = ax.secondary_xaxis('top', functions=(lambda x: x/(16.0*60.0), lambda x: x*(16.0*60.0)))
        secax.set_xlabel('Coordinate, R_Sun')
        plt.draw()
        plt.show()
  
    return znew

def vignetting(pixscale, x_IO, y_IO, R_IO, **kwargs):
    """Returns vignetting function. Should receive pixscale [arcsec/pix], x_IO [pix], y_IO [pix], R_IO [mm]
    """
    #### now these are received as positional parameters
    #pixscale = 2.82          # atan(dx/f)*!radeg*3600.00  # angular size of a pixel in arcsec
    #R_IO= 1.748              # mm
    #x_IO= 1023.5
    #y_IO= 1023.5

    r   = 25.0     # mm, radius of the aperture of the telescope
    ISD = 144348.0 # mm, NOMINAL (not actual) ISD size
    R_IO1 = R_IO*ISD/330.348 # mm, size of IO projected to the conjugate plane
    w_min = np.degrees(np.arctan((R_IO1-r)/ISD))*3600.0   # minimum angle in arcsec. Currently should be w_min=1055.6947" = 17.5949'=1.0997RSun
    w_max = np.degrees(np.arctan((R_IO1+r)/ISD))*3600.0   # maximum angle in arcsec. Currently should be w_max=1127.1398" = 18.7857'=1.1741RSun
    ### creating xx and yy 2D arrays with angular coordinates; 
    # in fact should use meshgrid for that
    xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-x_IO) * pixscale
    yy = np.outer(np.linspace(0,2047,num=2048)-y_IO,np.ones(2048)) * pixscale
    rr=np.sqrt( np.add(np.square(xx),np.square(yy)) )
    # radial coordinates across det. WRT IO
    idx=((rr >= w_min) & (rr <= w_max)) ## boolean indexing of the zone of vignetting
    ## simple linear dependence
    vign = np.ones((2048,2048))
    vign[idx]=(rr[idx]-w_min) * 1.0 / (w_max-w_min)
    # more precise formula
    vign1 = np.ones((2048,2048)) ; h = np.ones((2048,2048)) ; beta = np.ones((2048,2048))
    h[idx] = r - np.tan( np.radians((rr[idx]-w_min)/3600.0) )*ISD
    beta[idx] = 2*np.arccos(h[idx]/r)
    vign1[idx] = (beta[idx]-np.sin(beta[idx]))/2/np.pi
    ## zero out inside w_min
    #idx=where(rr le w_min)
    #vign[idx] =0.0
    #vign1[idx]=0.0

    verbose=kwargs.get('verbose', False)
    if verbose:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(yy[:,1024],vign1[:,1024],label='Serpent area')
        ax.plot(yy[:,1024],vign[:,1024],label='Linear')
        ax.set_xlabel('Coordinate, arcsec')
        ax.set_ylabel('Transmission')
        ax.set_title('Vignetting')
        plt.xlim(1000,1150)
        #plt.xlabel("arcsec")
        plt.legend()
        secax = ax.secondary_xaxis('top', functions=(lambda x: x/(16.0*60.0), lambda x: x*(16.0*60.0)))
        secax.set_xlabel('Coordinate, R_Sun') 
        plt.show()

    return vign1

def vignetting2(header, params, **kwargs):
    """Returns vignetting function 2048x2048 pix centered on IO and takes into account dependence of R_IO 
       with polar angle from vign_polar_fit.fits. 
    """

    version_msg = "aspiics_optics.vignetting2() v1.0, takes into account variable R_IO"

    r   = 25.0     # mm, radius of the aperture of the telescope
    ISD = 144348.0 # mm, NOMINAL (not actual) ISD size, which is distance to the IO conjugate plane
    x_IO=params['calib_data']['x_IO']
    y_IO=params['calib_data']['y_IO']
    pixscale=header['CDELT1']                            

    #R_IO1 = R_IO*ISD/(330.348+0.76) # mm, size of IO projected to the conjugate plane
    #w_min = np.degrees(np.arctan((R_IO1-r)/ISD))*3600.0   # minimum angle in arcsec. Currently should be w_min=1055.6947" = 17.5949'=1.0997RSun
    #w_max = np.degrees(np.arctan((R_IO1+r)/ISD))*3600.0   # maximum angle in arcsec. Currently should be w_max=1127.1398" = 18.7857'=1.1741RSun
    
    xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-x_IO) * pixscale
    yy = np.outer(np.linspace(0,2047,num=2048)-y_IO,np.ones(2048)) * pixscale
    rr =np.sqrt( np.add(np.square(xx),np.square(yy)) )
    phi=np.arctan2(yy,xx)*180.0/np.pi
    idx=(yy < 0)
    phi[idx] += 360.0


    #here we are interpolating R_IO1 with taking into account R_IO(phi)    
    filename=os.path.join(params['calib_data']['vign'],'vign_polar_fit.fits')
    R_IO, header = read_fits_image_array(filename)
    f = interpolate.interp1d(R_IO[0,:],R_IO[1,:],fill_value='extrapolate')   # syntax -- i.interp1d(x,y)
    R_IO1 = f(phi)
    R_IO1 = R_IO1*1.748*ISD/(330.348+0.76)
    w_min = np.degrees(np.arctan((R_IO1-r-6.1)/ISD))*3600.0                  # <--  hack to final distance FFP
    w_max = np.degrees(np.arctan((R_IO1+r)/ISD))*3600.0   
    
    # radial coordinates across det. WRT IO
    idx=((rr >= w_min) & (rr <= w_max)) ## boolean indexing of the zone of vignetting
    vign1 = np.ones((2048,2048)) ; h = np.ones((2048,2048)) ; beta = np.ones((2048,2048))
    h[idx] = r - np.tan( np.radians((rr[idx]-w_min[idx])/3600.0) )*ISD*0.91               # <--  hack to final distance FFP
    hidx=(h < (-r)) ; h[hidx]=-25.0 
    beta[idx] = 2*np.arccos(h[idx]/r)
    vign1[idx] = (beta[idx]-np.sin(beta[idx]))/2/np.pi
 
    return vign1, version_msg 
