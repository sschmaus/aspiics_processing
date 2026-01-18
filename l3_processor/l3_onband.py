import numpy as np
from scipy import ndimage
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt
import aspiics_misc      as am
from l3_fcorona import f_corona
import argparse
import os
import sys


#def l3_onband(inputNBF,inputK,params):
print("%*******************************************************************************")
print("% L3_onband: processing ")

# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("fileNBF", help="input NBF Level-3 ASPIICS image")
parser.add_argument("fileWBF", help="input WBF Level-3 ASPIICS image)")
parser.add_argument("--outdir", help="force output dir", default='./output/')
# Parse arguments
args = parser.parse_args()

inputNBF   = args.fileNBF
inputK   = args.fileWBF
outputdir = args.outdir

dataNBF, headerNBF = am.read_fits_image_array(inputNBF)
dataK,   headerK   = am.read_fits_image_array(inputK)

### ********* needed to create F-corona model *************** ###
pixscale = headerNBF['CDELT1']
CRPIX1   = headerNBF['CRPIX1']-1.0  # !!!! to put back !!!! header['CRPIX1']-1.0            # these are center of the Sun in the image, re-centered during l3_merge
CRPIX2   = headerNBF['CRPIX2']-1.0  # header['CRPIX2']-1.0
RSUN_ARC = headerNBF['RSUN_ARC'] 


### ************* these are calibration coefficients ****************** ###
### ****** to be retrieved from Level-3 calibration data rep ********** ###
### ****** The coefficients IMHO should strictly depend on units of data; for MSB they are equal to 1 ********* ###
### ********** Here we use 6 coefficients. There are more coefficients in the DPM ******** ###
### ********** according to Laurent's definition. We introduce 6 (important) obviously re-defining them ******* ###
C_gr_G = 1.0 ; C_K_G=1.0  ; C_F_G=1.0
C_D3_D3= 1.0 ; C_K_D3=1.0 ; C_F_D3=1.0 


### ************* should be a copy from l3_fcorona ******************** ###
#pixscale=2.8125 ; x_IO=1023.5 ; y_IO=1023.5 ; RSun=16.0*60.0
xx = np.outer(np.ones(2048),np.linspace(0,2047,num=2048)-CRPIX1) * pixscale / RSUN_ARC
yy = np.outer(np.linspace(0,2047,num=2048)-CRPIX2,np.ones(2048)) * pixscale / RSUN_ARC

### use model='simple_sh' for verbose ###
Fcor,Fcor_msg,Fcor_kind = f_corona(xx,yy,model='simple_sh')  ### --- Sergei's data were created with Allen model ### ,verbose=True --- with plots


#### Original approach with which uses K- as an input and removes K- and F- corona
#filt=headerNBF['FILTER']
#if filt=='Fe XIV':
#    data = (dataNBF - dataK*C_K_G - Fcor*C_F_G)/C_gr_G
#    prod_id_str = 'Green line'
#elif filt=='He I':
#    data = (dataNBF - dataK*C_K_D3- Fcor*C_F_D3)/C_D3_D3
#    prod_id_str = 'He I D3 line'
#
#headerNBF.set('PROD_ID',prod_id_str,after='FILTER')
#headerNBF.set('PARENT1',os.path.basename(inputNBF),after='PROD_ID',comment='Input NBF file')
#headerNBF.set('PARENT2',os.path.basename(inputK),after='PARENT1',comment='Input K-corona file')
#headerNBF.set('PARENT3',Fcor_msg,after='PARENT2',comment='Used F-corona model')
#headerNBF.set('HISTORY',"On-band calibration l3_onband.py")
#headerNBF.set('HISTORY',"K-corona from "+inputK)
#headerNBF.set('HISTORY','F-corona model: '+Fcor_msg+'; interpolation method (kind): '+Fcor_kind)
#if 'F_COR' in headerNBF:
#    del headerNBF['F_COR']
#if 'K_COR' in headerNBF:
#    del headerNBF['K_COR']
#### END: Original approach with which uses K- as an input and removes K- and F- corona

### Temporary approach which uses WBF- as an input and removes WBF- corona
filt=headerNBF['FILTER']
if filt=='Fe XIV':
    data = (dataNBF - dataK*C_K_G)/C_gr_G
    prod_id_str = 'Green line'
elif filt=='He I':
    data = (dataNBF - dataK*C_K_D3)/C_D3_D3
    prod_id_str = 'He I D3 line'

headerNBF.set('PROD_ID',prod_id_str,after='FILTER')
headerNBF.set('PARENT1',os.path.basename(inputNBF),after='PROD_ID',comment='Input NBF file')
headerNBF.set('PARENT2',os.path.basename(inputK),after='PARENT1',comment='Input WBF-corona file')
headerNBF.set('HISTORY',"On-band calibration l3_onband.py")
#headerNBF.set('HISTORY',"K-corona from "+inputK)
#headerNBF.set('HISTORY','F-corona model: '+Fcor_msg+'; interpolation method (kind): '+Fcor_kind)
if 'F_COR' in headerNBF:
    del headerNBF['F_COR']
if 'K_COR' in headerNBF:
    del headerNBF['K_COR']
### END: Temporary approach which uses WBF- as an input and removes WBF- corona

# writing main file with removed corona
hdu = fits.PrimaryHDU(data,headerNBF)
file2write = os.path.join(outputdir,os.path.splitext(os.path.basename(inputNBF))[0]+'.onband.fits')
if os.path.isfile(file2write):
   print("% L3_onband. Removing existing file "+file2write)
   os.remove(file2write)
print("% L3_onband. Writing "+file2write)
hdu.writeto(file2write)


#if __name__ == '__main__':
#    inputNBF = sys.argv[1]
#    inputK   = sys.argv[2]
#    l3_onband(inputNBF,inputK)
 
