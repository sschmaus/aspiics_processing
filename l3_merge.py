#!/bin/python3
import numpy as np
from scipy import ndimage
from skimage import morphology
from astropy.io import fits
import matplotlib.pyplot as plt
#import aspiics_detector  as det
#import aspiics_optics    as optics
import aspiics_misc       as am
import argparse
import os
from pathlib import Path

print("%*******************************************************************************")
print("% L3_Merge: processing ")

# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
#parser.add_argument("file1", help="input Level-2 ASPIICS fits (min t_exp)")
#parser.add_argument("file2", help="input Level-2 ASPIICS fits (middle t_exp)")
#parser.add_argument("file3", help="input Level-2 ASPIICS fits (max t_exp)")
parser.add_argument("files", help="1-to-3 input Level-2 ASPIICS fits (min,med,max t_exp)", nargs='+')
parser.add_argument("--outdir", help="force output dir (default './output/')", default='./output/')
parser.add_argument("--center", help="re-center/de-rotate the images", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--coalign", help="co-align images such they match max t_exp image", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--save_shifted", help="save two shifted L2 images in addition to merged L3", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--CRVAL1", help="force CRVAL1 value for the files", type=float)
parser.add_argument("--CRVAL2", help="force CRVAL2 value for the files", type=float)
# Parse arguments
args = parser.parse_args()

outputdir = args.outdir
docenter  = args.center
coalign   = args.coalign
save_shifted = args.save_shifted
forceCRVAL1 = False
forceCRVAL2 = False
if args.CRVAL1 != None:
    forceCRVAL1 = True
    CRVAL1 = args.CRVAL1
    print("   Using CRVAL1 value {:.2f} arcsec".format(CRVAL1))
if args.CRVAL2 != None:
    forceCRVAL2 = True
    CRVAL2 = args.CRVAL2
    print("   Using CRVAL2 value {:.2f} arcsec".format(CRVAL2))


files=args.files
nfiles=len(files)
if nfiles == 1:
    fileA=files[0]
    dataA,headerA=am.read_fits_image_array(fileA)
    data01=dataA  ;  header01=headerA  ;  file01=fileA  ;  header1=header01  ;  file1=fileA  ;  header10=headerA  ;  file10=fileA
elif nfiles == 2:
    fileA=files[0]
    fileB=files[1]
    dataA,headerA=am.read_fits_image_array(fileA)
    dataB,headerB=am.read_fits_image_array(fileB)
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
    dataA,headerA=am.read_fits_image_array(fileA)
    dataB,headerB=am.read_fits_image_array(fileB)
    dataC,headerC=am.read_fits_image_array(fileC)
    exptimes=np.array([headerA['EXPTIME'],headerB['EXPTIME'],headerC['EXPTIME']])
    idx=np.argsort(exptimes)
    data3=[dataA,dataB,dataC]           ;  data3=np.array(data3)  # this is a numpy array
    header3=[headerA,headerB,headerC]                             #  this is a list
    data3=data3[idx,:,:]       ;   data01=np.squeeze(data3[0,:,:])  ;  data1=np.squeeze(data3[1,:,:])  ;  data10=np.squeeze(data3[2,:,:])
    header01=header3[idx[0]]   ;   header1=header3[idx[1]]          ;  header10=header3[idx[2]]
    file01 = files[idx[0]]     ;   file1=files[idx[1]]              ;  file10=files[idx[2]]
else:
    print("% L3_Merge: we expect 1--3 input files ... Exiting!")
    exit()

if coalign:
#    if nfiles != 3:
#        print("   to co-align we need 3 input files. Exiting ...")
#        exit()
    if docenter:
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

if docenter: 
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

if coalign:
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

def soft_merge(data_under, data_over, distance=30, plot=False):
    """ Merge two images using soft transition over given distance (in pixels).
        data_high: high-exposure image (to be used where valid)
        data_low: low-exposure image (to be used to fill invalid pixels in high-exposure image)
        distance: distance (in pixels) over which the weights change from 0 to 1
    """
    data_under_mask = np.isfinite(data_under) # all valid pixels
    data_over_mask = ~np.isnan(data_over) # all non-NaN pixels, inf is ok

    # all over areas where under data is invalid
    fill_mask = data_over_mask * (~data_under_mask)
    # remove small islands in the fill_mask with remove_small_objects because borders may not overlap perfectly after the coregistration 
    # fill_mask = morphology.remove_small_objects(fill_mask.astype(bool), min_size=64, connectivity=1)

    # calculate soft gradient with distance transform
    data_over_alpha = ndimage.distance_transform_edt(~fill_mask)
    # normalize to distance - this could be improved by using the slope of the image to determine blending distance (e.g., larger distance for shallower slopes)
    data_over_alpha = 1-np.clip(data_over_alpha/distance,0,1)
    # mask any area of the soft mask which doesn't have valid data_over
    data_over_alpha = data_over_alpha * data_over_mask * data_under_mask

    # now do the merging
    Im_out = data_under
    Im_out[fill_mask] = data_over[fill_mask]
    Im_out[data_over_alpha>0] = data_over[data_over_alpha>0] * data_over_alpha[data_over_alpha>0] + data_under[data_over_alpha>0] * (1 - data_over_alpha[data_over_alpha>0])

    if plot:
        fig, ax = plt.subplots(1,5, figsize=(15,5))
        ax[0].imshow(data_under_mask, cmap='gray', vmin=0, vmax=1)
        ax[0].set_title('data_under valid mask')
        ax[1].imshow(data_over_mask, cmap='gray', vmin=0, vmax=1)
        ax[1].set_title('data_over valid mask')
        ax[2].imshow(fill_mask, cmap='gray', vmin=0, vmax=1)
        ax[2].set_title('fill_mask for data_over')
        ax[3].imshow(data_over_alpha, cmap='gray')
        ax[3].set_title('soft blending edges for data_over')
        ax[4].imshow(Im_out, cmap='gray')
        ax[4].set_title('merged image')
        plt.tight_layout()
        plt.show()

    return Im_out


# Merging two or three images. By default take the data from data10, but substitute the NaN/Inf pixels via the pixels from smaller exposure file
if nfiles == 3:
    # pixels over which the weights change from 0 to 1
    distance = 30  
    Im_out = soft_merge(data1, data01, distance=distance, plot=False)
    Im_out = soft_merge(data10, Im_out, distance=distance, plot=False)

elif nfiles == 2:
    Im_out = soft_merge(data1, data01, distance=30, plot=False)
else:
    Im_out = data01

BadMask = Im_out < 0
Im_out[BadMask] = 1e-11
BadMask = ~np.isfinite(Im_out)
#Im_out[BadMask] = np.nan
DATAMIN = np.nanmin(Im_out[ ~ BadMask])
DATAMAX = np.nanmax(Im_out[ ~ BadMask])
DATAMEAN = np.nanmean(Im_out[ ~ BadMask])
DATAMEDN = np.nanmedian(Im_out[ ~ BadMask])



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

### ************* write down the final Im into fits ****
if not os.path.exists(outputdir):
    print("% L2_MASTER. Output directory "+outputdir+" does not exist. Creating it.")
    Path(outputdir).mkdir(parents=True, exist_ok=True)
hdu=fits.PrimaryHDU(Im_out,header=headerM)
newname=os.path.splitext(os.path.basename(file10))[0]+'.merged.fits'
newname=newname.replace("l2","l3")
#file2write = './output/'+os.path.splitext(os.path.basename(file1))[0]+'.merged.fits'
file2write = os.path.join(outputdir,newname)
if os.path.isfile(file2write):
    print("% L3_Merge. Removing existing file "+file2write)
    os.remove(file2write)
print("% L3_Merge. Writing "+file2write)
hdu.writeto(file2write)

#save_shifted=True        ##  for debugging
if save_shifted:
    header01.set('LEVEL','L2-coaligned','data processing level')
    header01.set('PROD_ID','L2-coaligned',after='LEVEL')
    header01.set('CREATOR',"Sergei's l3_merge", "FITS creation software")
    header01.set('FILENAME',header01['FILENAME'].replace("l1","l3"))
    header01.set('FILENAME',header01['FILENAME'].replace("l2","l3"))
    header1.set('LEVEL','L2-coaligned','data processing level')
    header1.set('PROD_ID','L2-coaligned',after='LEVEL')
    header1.set('CREATOR',"Sergei's l3_merge", "FITS creation software")
    header1.set('FILENAME',header1['FILENAME'].replace("l1","l3"))
    header1.set('FILENAME',header1['FILENAME'].replace("l2","l3"))
    hdu01=fits.PrimaryHDU(data01,header=header01)
    newname01=os.path.splitext(os.path.basename(file01))[0]+'.shifted.fits'
    newname01=newname01.replace("l2","l3")
    newname01=os.path.join(outputdir,newname01)
    hdu1=fits.PrimaryHDU(data1,header=header1)
    newname1=os.path.splitext(os.path.basename(file1))[0]+'.shifted.fits'
    newname1=newname1.replace("l2","l3")
    newname1=os.path.join(outputdir,newname1)
    if os.path.isfile(newname01):
        os.remove(newname01)
    if os.path.isfile(newname1):
        os.remove(newname1)
    print("% L3_Merge: saving modified 2 other files")
    hdu01.writeto(newname01)
    hdu1.writeto(newname1)
        
