05-06-2025  -- calib_data3

This is a copy of calib_data2 with small modification of filter names. Due to wrong filter names in the on-ground calibration data, the originally delivered (fall 2022) 
calibration files implied P1==Polarizer0, P2==Polarizer60, and P3=Polarizer120. However, since FILT_POS was preserved through years, we have for the in-flight data p1==Polarizer60, p2==Polarizer0, p3=Polarizer120.

Thus angle-files were renamed:
  --  aspiics_p1_ corresponds to 'FILT_POS=filter1' == 'POLAR=60' == 64.65 deg average
  --  aspiics_p2_ corresponds to 'FILT_POS=filter2' == 'POLAR=0' == 5.26 deg average
  --  aspiics_p3 as before
The flat field files were re-created with up-to-date filter/name; flatfield filenames are changed from "Ak_WB_flatfield_nonlin.fits" to "flatfield_wb.fits" etc.


27-01-2025  -- calib_data2

This is a copy of the calib_data1 and on-ground calibration data processing
BUT
with bias model derived from the first in-flight data, registered on 20th Dec 2024. Cycle_ID=1308007, Fe XIV files with t_exp = 0.1, 0.1, 0.5, 1,2,5,10,20. The satellites in Stack, pointed towards Sun, the HDD is closed, Fe XIV filter. Simplified no-temperature dependence model (==> bias_B=0) is used.


*********************************************************************************************************************
*********************************************************************************************************************
Meaning of the files:
*********************************************************************************************************************
calibr_data.json.real      -  A json calibration/configuration file with such parameters as radiometric sensitivity, position of the IO
                              path's to other files

bias_A.fits & bias_B.fits  -  Bias temperature dependent model bias=A+B*temp
                              Sometimes there are files with additional _linfit.fits or _ladfit.fits suffixes.
                              These were created using different linfit/ladfit extrapolations to t_exp=0. There is no significant difference, though.
                              
dark_A2.fits, dark_B2.fits -  Dark current quadratic (in temperature) model. Dark=(dark_A2+dark_B2*temp+dark_C2*temp^2)*t_exp.
    dark_C2.fits              This is the default model. 

dark_A.fits & dark_B.fits  -  Dark current linear (in temperature) model. Currently not used.

aspiics_p1_angle.fits      -  ROB model for orientation of polarizers. 2D maps. See description from 05-06-2025
aspiics_p2_angle.fits         
aspiics_p3_angle.fits

flatfield_XX.fits          -  Flatfields for all 6 channels. There exist versions which take nonlinearity into account (default) or do not take,
                              and correct flat field in the vignetting zone after vignetting correction (dafault) or do not correct.
                              
detector_nonlin.fits       -  Detector nonlinearity

vign_polar_fit.fits        -  Variation of the IO radius with polar angle

hotpixels_list.fits        -  Hot pixels. These are broken in almost every image, thus their values is substituded.

unusual_list.fits          -  Some pixels occasionally show unusual behavior - their intensity is higher by 20-40 (200-400?) DNs. They are situated 
                              in columns in checker-board order. Under investigation. Currently the list is not used.


*********************************************************************************************************************
The following files were prepared by INAF OATo, they might be not deliviered in every version of the calibration data
a0_filter{1,2,3}.fits      -  2D maps of Malus curve fits
a2_filter{1,2,3}.fits
ak_pol{1,2,3}.fits

modulation_matrix_ij.fits
demodulation_matrix_ij.fits

t_pol{1,2,3}.fits
*********************************************************************************************************************
