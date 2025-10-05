# the second version uses real calibration data stored in 2D maps and calculates per-pixel demodulation matrix

import numpy as np
from scipy import ndimage
from astropy.io import fits
import matplotlib.pyplot as plt
#import aspiics_detector  as det
#import aspiics_optics    as optics
import aspiics_misc       as am
import argparse
import os
import sys
import time

print("%*******************************************************************************")
print("% L3_polariz.2: processing (uses per-pixel polarization orientation) ...")



# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("file1", help="input p1 ASPIICS fits")
parser.add_argument("file2", help="input p2 ASPIICS fits ")
parser.add_argument("file3", help="input p3 ASPIICS fits")
parser.add_argument("--outdir", help="force output dir (default './output/')", default='./output/')
parser.add_argument("--caldir", help="force folder with calibration files", default='./calib_data3/')
parser.add_argument("--center", help="re-center/de-rotate the images", default=True, action=argparse.BooleanOptionalAction)
# Parse arguments
args = parser.parse_args()

outputdir = args.outdir
docenter  = args.center
caldir    = args.caldir

# # receive input filenames from the command line. 
file1 =args.file1  #sys.argv[1]
file2 =args.file2  #sys.argv[2]
file3 =args.file3  #sys.argv[3]
#file1="input/polariz_testCalData_0.fits"
#file2="input/polariz_testCalData_60.fits"
#file3="input/polariz_testCalData_120.fits"

file2write_I =     os.path.join(outputdir,os.path.splitext(os.path.basename(file1))[0]+'.totalB.fits')         # 'output_totalB.fits'
file2write_pB =    os.path.join(outputdir,os.path.splitext(os.path.basename(file1))[0]+'.pB.fits')             # 'output_pB.fits'
file2write_alpha = os.path.join(outputdir,os.path.splitext(os.path.basename(file1))[0]+'.alpha.fits')          # 'output_alpha.fits'

print("    Reading calibration data from ",caldir)
polar1_filename = os.path.join(caldir,"aspiics_p1_angle.fits")  #filename = params["calib_data"]["nonlin"]
polar2_filename = os.path.join(caldir,"aspiics_p2_angle.fits")
polar3_filename = os.path.join(caldir,"aspiics_p3_angle.fits")
polar1, polar1_head =am.read_fits_image_array(polar1_filename)
polar2, polar2_head =am.read_fits_image_array(polar2_filename)
polar3, polar3_head =am.read_fits_image_array(polar3_filename)
#polar1[:,:]=0.
#polar2[:,:]=60.
#polar3[:,:]=120.
print("    Polarization mean angles: ", np.mean(polar1), ", ", np.mean(polar2), ", ", np.mean(polar3), "  (before CROTA)")


# # receive input filenames from the command line. 
file1 =args.file1  #sys.argv[1]
file2 =args.file2  #sys.argv[2]
file3 =args.file3  #sys.argv[3]
#file1="input/polariz_testCalData_0.fits"
#file2="input/polariz_testCalData_60.fits"
#file3="input/polariz_testCalData_120.fits"

data1, header1 =am.read_fits_image_array(file1)
data2, header2 =am.read_fits_image_array(file2)
data3, header3 =am.read_fits_image_array(file3)
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
if docenter:
    data1 = am.rotate_center1(data1,header1)   
    polar1= am.rotate_center1(polar1,header1c,verbose=True)+CROTA1   # we need to apply the same transformations to the polarization angles
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
Dem_av=np.array([[np.mean(Dem11),np.mean(Dem21),np.mean(Dem31)],           #  for 0,60,120 should equal to 2/3*(1    1      1) 
                 [np.mean(Dem12),np.mean(Dem22),np.mean(Dem32)],           #                               2/3*(2   -1     -1)
                 [np.mean(Dem13),np.mean(Dem23),np.mean(Dem33)]])          #                               2/3*(0 1.732  1.732)


#print("Demodulation matrix")
#print(Dem)

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
header_I.set('BUNIT','MSB')
header_I.set('FILTER','0, 60, 120','Spectral passband corresponds to WB')
header_I.set('POLAR','                  ','Removed after pol.processing')

header_pB = header1.copy()
header_pB.set('PROD_ID',"Polarised brightness",after='LEVEL')
header_pB.set('BTYPE','Polarized brightness','for polarized data - B, pB, alpha',before='BUNIT')
header_pB.set('BUNIT','MSB','here MSB referes to B')
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
file2write_I    =os.path.join(outputdir,file2write_I)
file2write_pB   =os.path.join(outputdir,file2write_pB)
file2write_alpha=os.path.join(outputdir,file2write_alpha)

#### ************* write down the final Im into fits ****
hdu_I=fits.PrimaryHDU(I,header=header_I)
if os.path.isfile(file2write_I):
    print("% L3_polariz.2: Removing existing file "+file2write_I)
    os.remove(file2write_I)
print("% L3_polariz.2: Writing "+file2write_I)
hdu_I.writeto(file2write_I)

hdu_pB=fits.PrimaryHDU(pB,header=header_pB)
if os.path.isfile(file2write_pB):
    print("% L3_polariz.2: Removing existing file "+file2write_pB)
    os.remove(file2write_pB)
print("% L3_polariz.2: Writing "+file2write_pB)
hdu_pB.writeto(file2write_pB)

hdu_alpha=fits.PrimaryHDU(alpha,header=header_alpha)
if os.path.isfile(file2write_alpha):
    print("% L3_polariz.2: Removing existing file "+file2write_alpha)
    os.remove(file2write_alpha)
print("% L3_polariz.2: Writing "+file2write_alpha)
hdu_alpha.writeto(file2write_alpha)

#hdu_Q=fits.PrimaryHDU(Q,header=header_alpha)
#if os.path.isfile("output_Q.fits"):
#    print("% L3_polariz. Output file 'output_Q.fits' exists. Removing it")
#    os.remove("output_Q.fits")
#print("% L3_polariz. Writing output_Q")
#hdu_Q.writeto("output_Q.fits")
#
#hdu_U=fits.PrimaryHDU(U,header=header_alpha)
#if os.path.isfile("output_U.fits"):
#    print("% L3_polariz. Output file 'output_U.fits' exists. Removing it")
#    os.remove("output_U.fits")
#print("% L3_polariz. Writing output_U")
#hdu_U.writeto("output_U.fits")


