from math import pi
import numpy             as np
import matplotlib.pyplot as plt
import os

def method1(im):
    w2=int(len(im)/2)
    xx = np.outer(np.ones(2*w2),np.linspace(0,2*w2-1,num=2*w2))
    yy = np.outer(np.linspace(0,2*w2-1,num=2*w2),np.ones(2*w2)) 
    xc = np.mean(np.mean(im*xx,axis=1)/np.mean(im,axis=1))
    yc = np.mean(np.mean(im*yy,axis=0)/np.mean(im,axis=0))
    return xc, yc

def method2(im):
    w2=int(len(im)/2)
    xx = np.outer(np.ones(2*w2),np.linspace(0,2*w2-1,num=2*w2))
    yy = np.outer(np.linspace(0,2*w2-1,num=2*w2),np.ones(2*w2)) 
    xc = (np.mean(im*xx)/np.mean(im))
    yc = (np.mean(im*yy)/np.mean(im))
    return xc, yc

def aspiics_get_opse(Image_in,header,params,**kwargs):
    Im = Image_in   # create a local copy of image since I don't remember whether the input array is overwritten    
    verbose=kwargs.get('verbose',False)
    save_image=kwargs.get('save_image',False)

    print("% aspiics_get_opse: ")

    ## expected positions of the LEDs in the image WRT (x_IO;y_IO)
    #x0=1053-1023.5  ; y0=1078-1023.5
    #x1=978 -1023.5  ; y1=1078-1023.5
    ##x2=1054-1023.5  ; y2=971 -1023.5      ; in the beginning there was a mistake due to crazy Z-orientation in the CPLF 
    #x2=978-1023.5   ; y2=971 -1023.5
    #led_coords=np.array([[x0,x1,x2],[y0,y1,y2]])
    ##print(f"provisional coords hardcoded: ")
    ##print(led_coords)    


    measured_x=np.zeros((3))    
    measured_y=np.zeros((3))    

    x_IO=params['calib_data']['x_IO'] #- 1.0    this is derived from the calibration data where I use 0..2047 notation. No need to use FITS-like style
    y_IO=params['calib_data']['y_IO'] #- 1.0
    pixscale=header['CDELT1']
    ISD     =header['ISD']*1000.0
    #ISD = 144348.0

    # expected positions of the LEDs in the image WRT (x_IO;y_IO) derived from geometrical parameters
    LEDA_Z = -0.0857 ; LEDA_Y = 0.1
    LEDB_Z =  0.0647 ; LEDB_Y = 0.1
    LEDC_Z =  0.0647 ; LEDC_Y =-0.1
    led_coords=np.array([[-LEDA_Z,-LEDB_Z,-LEDC_Z],[LEDA_Y,LEDB_Y,LEDC_Y]])   # calculate expected LEDs coordinates in pix from meters
    ### During commissioning we have realized the images are 180deg rotated
    #led_coords=-led_coords                                                    # During commissioning we have realized the images are 180deg rotated
    ## Instead rotate the image
    print("  Rotating the image to take into account current L0/L1 processing\n   (with the rotation the image corresponds to the actual scene seen by ASPIICS)")
    Im = np.rot90(Im,2)
    x_IO=2047.0-x_IO  
    y_IO=2047.0-y_IO
    led_coords=led_coords*1000.0/ISD                                          # tangent
    led_coords=np.arctan(led_coords)*180.0/pi*3600.0                          # angle in arcsec
    led_coords=led_coords/pixscale                                            # in pixels
    #print("calculated coords (WRT IO)")
    #print(led_coords)    
    #print("calculated coords +(x_IO,y_IO)")
    #print(led_coords+np.array([[x_IO],[y_IO]]))    


    w2=30 # half-width of ROI

    #xx = np.outer(np.ones(2*w2),np.linspace(0,2*w2-1,num=2*w2))
    #yy = np.outer(np.linspace(0,2*w2-1,num=2*w2),np.ones(2*w2)) 
    #rr=np.sqrt( np.add(np.square(xx),np.square(yy)) )

    linestyle='-.' ; w=0.5
    if verbose:
      #plt.clf()
      plt.figure(figsize=(12, 12))
      #ax=fig.add_subplot(111)
      plt.imshow(Im,origin='lower',vmin=1e-11,vmax=1e-6)
      filename=header['FILENAME'].replace("l1","l2")
      plt.title(filename)
      plt.plot((x_IO,x_IO),(y_IO-20,y_IO+20),color='orange')
      plt.plot((x_IO-20,x_IO+20),(y_IO,y_IO),color='orange')

    for i in range(0,3):
      #x0 = int(led_coords[0,i]+1023)-w2 ; x1 = x0+2*w2  ## -1 - the sub-indexing in python is idiotic!!!!
      #y0 = int(led_coords[1,i]+1023)-w2 ; y1 = y0+2*w2
      x0 = int(led_coords[0,i]+x_IO)-w2 ; x1 = x0+2*w2  ## -1 - the sub-indexing in python is idiotic!!!!
      y0 = int(led_coords[1,i]+y_IO)-w2 ; y1 = y0+2*w2
      ROI = Im[y0:y1,x0:x1]
      xc1, yc1 = method1(ROI) ; xc1=xc1+x0 ; yc1=yc1+y0
      xc2, yc2 = method2(ROI) ; xc2=xc2+x0 ; yc2=yc2+y0
      measured_x[i]=xc2                                 ## method 2 works better (tested with recent APSIICSE_eclipse_OPSE images
      measured_y[i]=yc2
      if verbose: 
         print(f"i={i:2}. ROI coords: {x0:4},{y0:4},{x1:4},{y1:4}. ", end='')
         print(f"Measured coordinates x1/y1, x2/y2: {xc1:7.2f}/{yc1:7.2f}, {xc2:7.2f}/{yc2:7.2f}")
         #plt.plot((xc1-10,xc1+10),(yc1,yc1), color='r', label='method 1')
         #plt.plot((xc1,xc1), (yc1-10,yc1+10), color='r')
         plt.plot((xc2-10,xc2+10),(yc2,yc2), color='m', label='method 2')
         plt.plot((xc2,xc2), (yc2-10,yc2+10), color='m')
         plt.text(xc2+10, yc2, i, color='orange')
         plt.plot((xc2-w2,xc2+w2),(yc2-w2,yc2-w2),color='w',ls=linestyle,linewidth=w) ; plt.plot((xc2+w2,xc2+w2),(yc2-w2,yc2+w2),color='w',ls=linestyle,linewidth=w) ; plt.plot((xc2-w2,xc2+w2),(yc2+w2,yc2+w2),color='w',ls=linestyle,linewidth=w) ; plt.plot((xc2-w2,xc2-w2),(yc2-w2,yc2+w2),color='w',ls=linestyle,linewidth=w)

    ### second iteration, to have higher precision
    w2=14
    if verbose:
      print()
      print("Second iteration, to get higher precision")
    for i in range(0,3):
      x0 = int(measured_x[i])-w2 ; x1 = x0+2*w2  ## -1 - the sub-indexing in python is idiotic!!!! 
      y0 = int(measured_y[i])-w2 ; y1 = y0+2*w2
      ROI = Im[y0:y1,x0:x1]
      xc1, yc1 = method1(ROI) ; xc1=xc1+x0 ; yc1=yc1+y0
      xc2, yc2 = method2(ROI) ; xc2=xc2+x0 ; yc2=yc2+y0
      measured_x[i]=xc2                                 ## method 2 works better (tested with recent APSIICSE_eclipse_OPSE images
      measured_y[i]=yc2
      if verbose: 
         print(f"i={i:2}. ROI coords: {x0:4},{y0:4},{x1:4},{y1:4}. ", end='')
         print(f"Measured coordinates x2/y2:                         {xc2:7.2f}/{yc2:7.2f}")
         #plt.plot((xc1-10,xc1+10),(yc1,yc1), color='y', label='method 1 - 2nd step')
         #plt.plot((xc1,xc1), (yc1-10,yc1+10), color='y')
         plt.plot((xc2-10,xc2+10),(yc2,yc2), color='g', label='method 2 - 2nd step')
         plt.plot((xc2,xc2), (yc2-10,yc2+10), color='g')
         plt.text(xc2+10, yc2, i, color='green')
         plt.plot((xc2-w2,xc2+w2),(yc2-w2,yc2-w2),color='w',ls=linestyle,linewidth=w) ; plt.plot((xc2+w2,xc2+w2),(yc2-w2,yc2+w2),color='w',ls=linestyle,linewidth=w) ; plt.plot((xc2-w2,xc2+w2),(yc2+w2,yc2+w2),color='w',ls=linestyle,linewidth=w) ; plt.plot((xc2-w2,xc2-w2),(yc2-w2,yc2+w2),color='w',ls=linestyle,linewidth=w)

    # Save into header measured LED coordinates
    for i in range(0,3):
        xS=2047.0-measured_x[i]
        yS=2047.0-measured_y[i]
        #print('keyword name: "LED{:1d}_X"'.format(i))
        #print('value: {:6.1f}'.format(xS))
        header.set( "LED{:1d}_X".format(i) , float("{:6.1f}".format(xS+1)) , " [pix] (1..2048) - measured x-coordinate")
        header.set( "LED{:1d}_Y".format(i) , float("{:6.1f}".format(yS+1)) , " [pix] (1..2048) -        y-coordinate")
        #header.set('OPLFOY',float("{:.4f}".format(OPLFOY)),"[m] OPLF coordinate from OPSE",after='OPLFNZ')

    # convert coordinates of LEDs from pixels into meters in CPLF
    CPLF_z =-ISD * np.tan( (measured_x-x_IO)*pixscale/3600.0*pi/180.0 ) / 1000.0
    CPLF_y = ISD * np.tan( (measured_y-y_IO)*pixscale/3600.0*pi/180.0 ) / 1000.0
    #### During commissioning we have realized the images are 180deg rotated
    #CPLF_z =-ISD * np.tan( -(measured_x-x_IO)*pixscale/3600.0*pi/180.0 ) / 1000.0
    #CPLF_y = ISD * np.tan( -(measured_y-y_IO)*pixscale/3600.0*pi/180.0 ) / 1000.0
    for i in range(0,3):
       print(f"   LED {i:1}: Y={CPLF_y[i]:7.4f} m, Z={CPLF_z[i]:7.4f} m.")
    # calculate position of the OPLF center using LEDs position in CPLF
    # see sketch "Position of the LEDs in OPLF-CPLF.png" -- derivation of the OPLF center from LEDs and explanation
    c1 = -0.0647/(-0.0857-0.0647)     #=0.43018617     # coefficient for BA
    c2 = -0.1/(-0.1-0.1)              #=0.5            # coefficient for BC
    y_BA = CPLF_y[0] - CPLF_y[1]                    # this is vector BA
    z_BA = CPLF_z[0] - CPLF_z[1]
    y_BC = CPLF_y[2] - CPLF_y[1]                    # this is vector BC
    z_BC = CPLF_z[2] - CPLF_z[1]
    y_BO1 = c1*y_BA + c2*y_BC       # this is vector BO'
    z_BO1 = c1*z_BA + c2*z_BC
    OPLFOY =  CPLF_y[1] + y_BO1     #  OPLFOY = y_O1 = y_B + y_BO1            # this is coordinate of the point O'
    OPLFOZ =  CPLF_z[1] + z_BO1     #  OPLFOZ = z_O1 = z_B + z_BO1
    print(f"   OPLFOZ={OPLFOZ:8.5f} m;  OPLFOY={OPLFOY:8.5f} m;")
    #print(f"   or approximately {np.arctan(OPLFOZ/(ISD/1000.0))/pi*180.0*3600.0/pixscale:8.4f} pix, {np.arctan(OPLFOY/(ISD/1000.0))/pi*180.0*3600.0/pixscale:8.4f} pix")
  
    if verbose:
      for i in range(0,3):
          plt.annotate(f" LED {i:1}: Z={CPLF_z[i]:7.4f} m, Y={CPLF_y[i]:7.4f} m; ({measured_x[i]:6.1f},{measured_y[i]:6.1f} pix).",(100,60-i*25),xycoords='figure pixels')
      plt.legend()
      plt.xlim((950,1150))
      plt.ylim((900,1100))
      plt.draw()
      if save_image:
          filename=os.path.basename(header['FILENAME'])
          filename=filename.replace("fits","zoom_OPSE.png")
          filename=filename.replace("l1","l2")
          print("Saving image "+filename)
          plt.savefig(filename,format='png')
      plt.show()

    len_BA_mm=1000.*np.sqrt(y_BA**2+z_BA**2)
    len_BC_mm=1000.*np.sqrt(y_BC**2+z_BC**2)
    header.set('OPLFOY',float("{:.4f}".format(OPLFOY)),"[m] OPLF coordinate from OPSE",after='OPLFNZ')
    header.set('OPLFOZ',float("{:.4f}".format(OPLFOZ)),"[m] OPLF coordinate from OPSE",after='OPLFOY')
    header.set('OPLF_BAL',float("{:6.2f}".format(len_BA_mm)),"[mm] measured distance A-B LEDs",after='OPLFOZ')
    header.set('OPLF_BCL',float("{:6.2f}".format(len_BC_mm)),"[mm] measured distance B-C LEDs",after='OPLF_BAL')
    header.set("HISTORY","OPLF center and distance between LEDs were measured using aspiics_get_opse.py")
    return OPLFOY, OPLFOZ
