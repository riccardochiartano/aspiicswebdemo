import numpy as np
from scipy import ndimage
from astropy.io import fits

def read_fits_image_array(filename):
    """See https://docs.astropy.org/en/stable/io/fits/ for more info"""
    with fits.open(filename, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data
       header    = hdul[0].header
    return imagedata, header

def write_fits(imagedata, inputfile, outputfile):
    with fits.open(inputfile) as hdul:
       hdul[0].data = imagedata
       hdul.writeto(outputfile)

def rotate_center(image,header):
    """Performs rotation and re-centering of an image based on its CRPIX1, CRPIX2, CROTA keywords"""
    CRPIX1=header['CRPIX1']-1.0     # pix, location of the solar center in the image. Convert to range 0..2047 (retrieved from the header)
    CRPIX2=header['CRPIX2']-1.0     #                             ^^^ this is wrong (even according to the standard). In L1 files ..
                                    #          the CRPIX1/2 keywords denote reference pixels, which corresponds to some coordinates (CRVAL1/2) on the Sun
    CROTA =header['CROTA']          # deg, rotation angle wrt WCS
    CROTA = CROTA/180.0*np.pi
    Ts1  = np.array([[1.,0.,-CRPIX2],[0.,1.,-CRPIX1],[0.,0.,1.]])                                      # shift coordinates to have center (0,0)
    Trot = np.array([[np.cos(CROTA),np.sin(CROTA),0.],[-np.sin(CROTA),np.cos(CROTA),0.],[0.,0.,1.]])   # rotate coordinates by ?CROTA
    Ts   = np.array([[1.,0.,1023.5],[0.,1.,1023.5],[0.,0.,1]])                                         # shift coordinates to have center (1023.5,1023.5)
    T = Ts @ Trot @ Ts1
    print(CRPIX1, CRPIX2, CROTA)
    print(T)
    image = ndimage.affine_transform(image,np.linalg.inv(T),mode='constant',cval=float("NaN"),order=3,prefilter=False)

    # updating position of the IO center in the image
    X_IO=header['X_IO']-1.
    Y_IO=header['Y_IO']-1.
    MRot=np.array([[np.cos(CROTA),-np.sin(CROTA)],[np.sin(CROTA),np.cos(CROTA)]])
    pos_IO1=np.array([[X_IO-CRPIX1],[Y_IO-CRPIX2]])
    new_IO=MRot @ pos_IO1 + np.array([[1023.5],[1023.5]])
    print("New cooordinates of the IO:", new_IO)
    header.set('X_IO1',new_IO[0,0]+1,'X_IO coord after CROTA (1..2048)')
    header.set('Y_IO1',new_IO[1,0]+1,'Y_IO coord after CROTA (1..2048)')

    return image


def rotate_center1(image,header, **kwargs):
    """Performs de-rotation and re-centering of an image based on its CRPIX1/2, CRVAL1/2, CROTA keywords"""
    """Similar to rotate_center, but in fact CRPIX1/2 not necessarily denote the solar center, just a reference pixel"""

    verbose=kwargs.get('verbose',False)

    CRPIX1=header['CRPIX1']-1.0     # pix, location of the solar center in the image. Convert to range 0..2047 (retrieved from the header)
    CRPIX2=header['CRPIX2']-1.0
    CROTA =header['CROTA']          # deg, rotation angle wrt WCS
    CROTA = CROTA/180.0*np.pi
    pixscale=header['CDELT1']
    CRVAL1=header['CRVAL1']
    CRVAL2=header['CRVAL2']
    Ts1  = np.array([[1.,0.,-(CRPIX2)],[0.,1.,-(CRPIX1)],[0.,0.,1.]])                                  # shift coordinates to have reference pixel at (0,0)
    Trot = np.array([[np.cos(CROTA),np.sin(CROTA),0.],[-np.sin(CROTA),np.cos(CROTA),0.],[0.,0.,1.]])   # rotate coordinates by ?CROTA
    Tcent= np.array([[1.,0.,CRVAL2/pixscale],[0.,1.,CRVAL1/pixscale],[0.,0.,1.]])                      # shift once again to have solar center at (0,0)
    Ts   = np.array([[1.,0.,1023.5],[0.,1.,1023.5],[0.,0.,1]])                                         # shift coordinates to have center (1023.5,1023.5)
    T = Ts @ Tcent @ Trot @ Ts1
    if verbose:
        np.set_printoptions(suppress=True,precision=3)
        print("   CRPIX1,CRPIX2,CRVAL1,CRVAL2,CROTA:", CRPIX1, CRPIX2, CRVAL1, CRVAL2, np.degrees(CROTA))
        print("   Direct transformation matrix:")
        print(T)
    image = ndimage.affine_transform(image,np.linalg.inv(T),mode='constant',cval=float("NaN"),order=3,prefilter=False)
    header.set('CRPIX1',1024.5,'[pix] (1..2048) Location of the reference pixel')
    header.set('CRPIX2',1024.5,'[pix] The image has been re-centered => 1024.5')
    header.set('CRVAL1',0.0,"[arcsec] reference value on axis 1")
    header.set('CRVAL2',0.0,"[arcsec] reference value on axis 2")
    header.set('CROTA',0.0,"[deg] The image has been de-rotated wrt WCS")
    header.set('PC1_1',1.0,"Should be straight in WCS - [1,0,0,1]")
    header.set('PC1_2',0.0)
    header.set('PC2_1',0.0)
    header.set('PC2_2',1.0)

    #### ************ old version -- update only IO information *******************
    ## updating position of the IO center in the image
    #X_IO=header['X_IO']-1.
    #Y_IO=header['Y_IO']-1.
    #MRot=np.array([[np.cos(CROTA),-np.sin(CROTA)],[np.sin(CROTA),np.cos(CROTA)]])
    #MCRPIX=np.array([[X_IO-CRPIX1],[Y_IO-CRPIX2]])
    #MCRVAL=np.array([[CRVAL1/pixscale],[CRVAL2/pixscale]])
    #new_IO=MCRVAL + MRot @ MCRPIX + np.array([[1023.5],[1023.5]])
    #if verbose:
    #    print("   CRPIX/ROTA correction: ", np.transpose(MRot @ MCRPIX))
    #    print("   CRVAL correction: ", np.transpose(MCRVAL))
    #    print("   New cooordinates of the IO:", np.transpose(new_IO))
    #header.set('X_IO',float("{:.2f}".format(new_IO[0,0]+1)),'X_IO after re-center/de-rot (1..2048)')
    #header.set('Y_IO',float("{:.2f}".format(new_IO[1,0]+1)),'Y_IO after re-center/de-rot (1..2048)')
    #header.set('HISTORY',"IO position before re-centering {:.2f}/{:.2f} (1..2048)".format(X_IO+1,Y_IO+1))
    
    #### ************ new version -- update set of various coordinates, uses T-matrix *******************
    #### ************    to calculate the new coordinates                             *******************
    params=[('X_IO','Y_IO'),('LED0_X','LED0_Y'),('LED1_X','LED1_Y'),('LED2_X','LED2_Y'),('SOLC_X','SOLC_Y')]
    for par in params:
        if par[0] in header and par[1] in header: 
            header.set('HISTORY',"Updating "+par[0]+"/"+par[1]+" (old values {:.2f}/{:.2f})".format(header[par[0]],header[par[1]]))
            x0=header[par[0]]-1.
            y0=header[par[1]]-1.
            y1=T[0,0]*y0+T[0,1]*x0+T[0,2]
            x1=T[1,0]*y0+T[1,1]*x0+T[1,2]
            header.set(par[0],float("{:.2f}".format(x1+1.)))
            header.set(par[1],float("{:.2f}".format(y1+1.)))
            if verbose:
                print("New coordinates for "+par[0]+"/"+par[1]+": {:.2f}/{:.2f}".format(x1,y1))

    return image


def shift_image(image,header,header_ref,**kwargs):
    """Rotates&shifts the image to have the coordinates given by the header_ref (its CRPIX1/2, CRVAL1/2, CROTA keywords)"""
    """Should be used if three images have slightly different CRVAL1/2 CROTA, but we want to stick to the one with the longest t_exp"""
    """   but without centering of all the images."""

    verbose=kwargs.get('verbose',False)

    rCRPIX1=header_ref['CRPIX1']-1.0
    rCRPIX2=header_ref['CRPIX2']-1.0
    rCROTA=header_ref['CROTA']
    rCROTA=rCROTA/180.0*np.pi
    rCRVAL1=header_ref['CRVAL1']
    rCRVAL2=header_ref['CRVAL2']

    CRPIX1=header['CRPIX1']-1.0     # pix, location of the solar center in the image. Convert to range 0..2047 (retrieved from the header)
    CRPIX2=header['CRPIX2']-1.0
    CROTA =header['CROTA']          # deg, rotation angle wrt WCS
    CROTA = CROTA/180.0*np.pi
    pixscale=header['CDELT1']
    CRVAL1=header['CRVAL1']
    CRVAL2=header['CRVAL2']

    dCROTA=CROTA-rCROTA 
    dCRVAL1=(CRVAL1-rCRVAL1)        
    dCRVAL2=(CRVAL2-rCRVAL2)
  
    Ts1  = np.array([[1.,0.,-(CRPIX2)],[0.,1.,-(CRPIX1)],[0.,0.,1.]])                                      # shift coordinates to have reference pixel at (0,0)
    Trot1 = np.array([[np.cos(CROTA),np.sin(CROTA),0.],[-np.sin(CROTA),np.cos(CROTA),0.],[0.,0.,1.]])      # de-rotate coordinates by CROTA
    Trot2 = np.array([[np.cos(rCROTA),-np.sin(rCROTA),0.],[np.sin(rCROTA),np.cos(rCROTA),0.],[0.,0.,1.]])  # rotate again coordinates by rCROTA
    Tcent = np.array([[1.,0.,dCRVAL2/pixscale],[0.,1.,dCRVAL1/pixscale],[0.,0.,1.]])                       # shift once again to compensate difference in CRVAL1 rCRVAL1 etc
    Ts   = np.array([[1.,0.,rCRPIX2],[0.,1.,rCRPIX1],[0.,0.,1]])                                           # shift coordinates to have center as in the reference image
    T = Ts @ Trot2 @ Tcent @ Trot1 @ Ts1
    Tinv=np.linalg.inv(T)
    if verbose:
        np.set_printoptions(suppress=True,precision=3)
        print("   Direct transformation matrix:")
        print(T)
        print("   linalg.inv(T) transformation matrix (the one provided to ndimage.affine_transform):")
        print(Tinv)
    image = ndimage.affine_transform(image,Tinv,mode='constant',cval=float("NaN"),order=3,prefilter=False)
    header.set('HISTORY',"The image was co-aligned with the reference image (FILE_REF)")
    header.set('FILE_REF',header_ref['FILENAME'].replace("l1","l2"),"File used as a reference for co-alignment",after="CRVAL2",before="LONPOLE")   
    header.set('CRPIX1',rCRPIX1+1.,'[pix] (1..2048) Location of reference value')
    header.set('CRPIX2',rCRPIX2+1.,'[pix] Must correspond to those of FILE_REF')
    header.set('CRVAL1',rCRVAL1,"[arcsec] Reference value on axes 1,2")
    header.set('CRVAL2',rCRVAL2,"[arcsec] Must correspond to those of FILE_REF")
    header.set('CROTA',rCROTA*180./np.pi,"[deg] Must correspond to that of FILE_REF",after='CRVAL2')
    header.set('PC1_1',header_ref['PC1_1'],"pixel coord matrix. See comment for CROTA",after='CROTA')
    header.set('PC1_2',header_ref['PC1_2'],after='PC1_1')
    header.set('PC2_1',header_ref['PC2_1'],after='PC1_2')
    header.set('PC2_2',header_ref['PC2_2'],after='PC2_1')

    params=[('X_IO','Y_IO'),('LED0_X','LED0_Y'),('LED1_X','LED1_Y'),('LED2_X','LED2_Y'),('SOLC_X','SOLC_Y')]
    for par in params:
        if par[0] in header and par[1] in header: 
            header.set('HISTORY',"Updating "+par[0]+"/"+par[1]+" (old values {:.2f}/{:.2f})".format(header[par[0]],header[par[1]]))
            x0=header[par[0]]-1.
            y0=header[par[1]]-1.
            y1=T[0,0]*y0+T[0,1]*x0+T[0,2]
            x1=T[1,0]*y0+T[1,1]*x0+T[1,2]
            header.set(par[0],float("{:.2f}".format(x1+1.)))
            header.set(par[1],float("{:.2f}".format(y1+1.)))
            if verbose:
                print("New coordinates for "+par[0]+"/"+par[1]+": {:.2f}/{:.2f}".format(x1,y1))

    return image

def rotate_center1header(image,header, **kwargs):
    """Performs de-rotation and re-centering of an image based on its CRPIX1/2, CRVAL1/2, CROTA keywords"""
    """Similar to rotate_center, but in fact CRPIX1/2 not necessarily denote the solar center, just a reference pixel"""
    """Same as rotate_center1 but return also the header"""

    verbose=kwargs.get('verbose',False)

    CRPIX1=header['CRPIX1']-1.0     # pix, location of the solar center in the image. Convert to range 0..2047 (retrieved from the header)
    CRPIX2=header['CRPIX2']-1.0
    CROTA =header['CROTA']          # deg, rotation angle wrt WCS
    CROTA = CROTA/180.0*np.pi
    pixscale=header['CDELT1']
    CRVAL1=header['CRVAL1']
    CRVAL2=header['CRVAL2']
    Ts1  = np.array([[1.,0.,-(CRPIX2)],[0.,1.,-(CRPIX1)],[0.,0.,1.]])                                  # shift coordinates to have reference pixel at (0,0)
    Trot = np.array([[np.cos(CROTA),np.sin(CROTA),0.],[-np.sin(CROTA),np.cos(CROTA),0.],[0.,0.,1.]])   # rotate coordinates by ?CROTA
    Tcent= np.array([[1.,0.,CRVAL2/pixscale],[0.,1.,CRVAL1/pixscale],[0.,0.,1.]])                      # shift once again to have solar center at (0,0)
    Ts   = np.array([[1.,0.,1023.5],[0.,1.,1023.5],[0.,0.,1]])                                         # shift coordinates to have center (1023.5,1023.5)
    T = Ts @ Tcent @ Trot @ Ts1
    if verbose:
        np.set_printoptions(suppress=True,precision=3)
        print("   CRPIX1,CRPIX2,CRVAL1,CRVAL2,CROTA:", CRPIX1, CRPIX2, CRVAL1, CRVAL2, np.degrees(CROTA))
        print("   Direct transformation matrix:")
        print(T)
    image = ndimage.affine_transform(image,np.linalg.inv(T),mode='constant',cval=float("NaN"),order=3,prefilter=False)
    header.set('CRPIX1',1024.5,'[pix] (1..2048) Location of the reference pixel')
    header.set('CRPIX2',1024.5,'[pix] The image has been re-centered => 1024.5')
    header.set('CRVAL1',0.0,"[arcsec] reference value on axis 1")
    header.set('CRVAL2',0.0,"[arcsec] reference value on axis 2")
    header.set('CROTA',0.0,"[deg] The image has been de-rotated wrt WCS")
    header.set('PC1_1',1.0,"Should be straight in WCS - [1,0,0,1]")
    header.set('PC1_2',0.0)
    header.set('PC2_1',0.0)
    header.set('PC2_2',1.0)

    #### ************ old version -- update only IO information *******************
    ## updating position of the IO center in the image
    #X_IO=header['X_IO']-1.
    #Y_IO=header['Y_IO']-1.
    #MRot=np.array([[np.cos(CROTA),-np.sin(CROTA)],[np.sin(CROTA),np.cos(CROTA)]])
    #MCRPIX=np.array([[X_IO-CRPIX1],[Y_IO-CRPIX2]])
    #MCRVAL=np.array([[CRVAL1/pixscale],[CRVAL2/pixscale]])
    #new_IO=MCRVAL + MRot @ MCRPIX + np.array([[1023.5],[1023.5]])
    #if verbose:
    #    print("   CRPIX/ROTA correction: ", np.transpose(MRot @ MCRPIX))
    #    print("   CRVAL correction: ", np.transpose(MCRVAL))
    #    print("   New cooordinates of the IO:", np.transpose(new_IO))
    #header.set('X_IO',float("{:.2f}".format(new_IO[0,0]+1)),'X_IO after re-center/de-rot (1..2048)')
    #header.set('Y_IO',float("{:.2f}".format(new_IO[1,0]+1)),'Y_IO after re-center/de-rot (1..2048)')
    #header.set('HISTORY',"IO position before re-centering {:.2f}/{:.2f} (1..2048)".format(X_IO+1,Y_IO+1))
    
    #### ************ new version -- update set of various coordinates, uses T-matrix *******************
    #### ************    to calculate the new coordinates                             *******************
    params=[('X_IO','Y_IO'),('LED0_X','LED0_Y'),('LED1_X','LED1_Y'),('LED2_X','LED2_Y'),('SOLC_X','SOLC_Y')]
    for par in params:
        if par[0] in header and par[1] in header: 
            header.set('HISTORY',"Updating "+par[0]+"/"+par[1]+" (old values {:.2f}/{:.2f})".format(header[par[0]],header[par[1]]))
            x0=header[par[0]]-1.
            y0=header[par[1]]-1.
            y1=T[0,0]*y0+T[0,1]*x0+T[0,2]
            x1=T[1,0]*y0+T[1,1]*x0+T[1,2]
            header.set(par[0],float("{:.2f}".format(x1+1.)))
            header.set(par[1],float("{:.2f}".format(y1+1.)))
            if verbose:
                print("New coordinates for "+par[0]+"/"+par[1]+": {:.2f}/{:.2f}".format(x1,y1))

    return image, header