import numpy as np
from astropy.io import fits
from scipy import interpolate
from scipy.ndimage import median_filter, vectorized_filter, distance_transform_edt, binary_dilation
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

def darkcurrent(header,params,**kwargs):
    """Returns dark current 2D map [DN/sec], using model
          dark = A + B*temp + C*temp^2 (from processing of the on-ground calibration data"""
    
    version_msg = "aspiics_detector.darkcurrent() v1.0"
    
    fileA = os.path.join(params["calib_data"]["dark_current"],"dark_A2.fits")  
    fileB = os.path.join(params["calib_data"]["dark_current"],"dark_B2.fits")  
    fileC = os.path.join(params["calib_data"]["dark_current"],"dark_C2.fits")  
  
    print("% aspiics_detector.darkcurrent: using files dark_{A,B,C}2.fits from the")
    print("   "+params["calib_data"]["dark_current"]+" folder. Resulting units [DN/sec].")
   
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


def bias(header,params,**kwargs):
    """Returns bias 2D map [DN], using the model  bias = A + B*temp
          The 2D-arrays A and B are derived during processing of the calibration data"""

    version_msg = "aspiics_detector.bias() v1.0"
    
    fileA =os.path.join(params["calib_data"]["bias"],"bias_A.fits")
    fileB =os.path.join(params["calib_data"]["bias"],"bias_B.fits")
    
    print("% aspiics_detector.bias:        using files bias_{A,B}.fits from the ")
    print("   "+params["calib_data"]["bias"]+" folder. Resulting units [DN].")
    
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

def flat(header,params,**kwargs):
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

    with fits.open(fullfile, do_not_scale_image_data=True) as hdul:             
       imagedata = hdul[0].data

    return imagedata, version_msg+" with "+file

def fill_nan_mirror(a):
    """Fill NaN values by mirroring across nearest valid value using distance transform."""
    a = a.copy()
    mask = np.isnan(a)
    if not mask.any():
        return a

    # Find nearest valid indices
    nearest_idx = distance_transform_edt(mask, return_distances=False, return_indices=True)

    # Mirror across nearest valid value
    mirrored_idx = tuple(
        np.clip(2 * nearest_idx[dim] - np.indices(a.shape)[dim], 0, a.shape[dim]-1)
        for dim in range(a.ndim)
    )

    return a[mirrored_idx]

def fast_nanmedian(img, size, *args, **kwargs):
    """Applies a windowed median filter to an image, handling NaN values appropriately.
    Uses vectorized_filter with np.nanmedian for images with NaNs or of uint16 type,
    otherwise uses faster standard median_filter.

    Parameters:
    -----------
    img : numpy array
        The input image to be filtered.
    *args, **kwargs :
        Additional arguments to be passed to the median filter functions.

    Returns:
    --------
    filtered_img : numpy array
        The median filtered image.
    """
    if np.ndim(img) != 2:
        raise ValueError("fast_nanmedian only supports 2D images.")

    if np.isnan(img).all():
        # if all values are NaN, return as is
        # print("All NaN image passed to fast_nanmedian, returning as is.")
        return img
    elif np.isnan(img).any():
        mask = np.isfinite(img)
        
        r0 = mask.any(axis=1).argmax()
        r1 = mask.shape[0] - mask.any(axis=1)[::-1].argmax()
        c0 = mask.any(axis=0).argmax()
        c1 = mask.shape[1] - mask.any(axis=0)[::-1].argmax()
        cropped = img[r0:r1, c0:c1]

        # print(f"Cropped region for vectorized nanmedian: rows {r0}-{r1}, cols {c0}-{c1}")

        # fill NaNs in cropped region with nearest valid value
        cropped = fill_nan_mirror(cropped)

        # apply vectorized nanmedian to cropped region
        cropped_filtered = median_filter(cropped, size=size, *args, **kwargs)

        # place filtered cropped region back into all-NaN image
        img_filtered = np.full_like(img, np.nan)
        img_filtered[r0:r1, c0:c1] = cropped_filtered

        return img_filtered
    elif img.dtype == np.uint16:
        # if uint16 type, use vectorized nanmedian
        # print("Image with NaNs or uint16 type passed to vectorized nanmedian.")

        mask = np.isfinite(img)

        r0 = mask.any(axis=1).argmax()
        r1 = mask.shape[0] - mask.any(axis=1)[::-1].argmax()
        c0 = mask.any(axis=0).argmax()
        c1 = mask.shape[1] - mask.any(axis=0)[::-1].argmax()
        cropped = img[r0:r1, c0:c1]

        # print(f"Cropped region for vectorized nanmedian: rows {r0}-{r1}, cols {c0}-{c1}")

        # apply vectorized nanmedian to cropped region
        cropped_filtered = vectorized_filter(cropped, size=size, function=np.nanmedian, *args, **kwargs)

        # place filtered cropped region back into all-NaN image
        img_filtered = np.full_like(img, np.nan)
        img_filtered[r0:r1, c0:c1] = cropped_filtered

        return img_filtered
    else:
        # if no NaNs, use faster median filter
        return median_filter(img, size=size, *args, **kwargs)

def correct_banding(img, filter_2d=15, filter_1d=None, structures_threshold=None, plotting=True):
    """
    Corrects banding along the first dimension in an image using specified filtering and averaging methods.

    Parameters:
    -----------
    img : 2D numpy array
        The input image to be corrected.
    saturation_mask : 2D numpy array
        A boolean mask indicating saturated pixels in the image.
    filter_2d : int, optional
        Size of the median filter applied in 2D to remove large structures.
    filter_1d : int | False, optional
        Size of the median filter applied in 1D to remove outliers.
        Default is 4 * filter_2d.
        If False, no 1D filtering is applied and banding profile is computed for the entire line with np.nanmedian.
    structures_threshold : float, optional
        Threshold in standard deviation units to identify remaining structures.
        Default is 60 / filter_2d.

    Returns:
    --------
    corrected_img : 2D numpy array
        The banding-corrected image.

    """
    if structures_threshold is None:
        structures_threshold = 60 / filter_2d # threshold varies with filter size
    if filter_1d is None:
        filter_1d = filter_2d * 4  # make 1d filter size proportional to 2d filter size

    img_ref = np.float32(img.copy())

    # img_ref = np.nan_to_num(img_ref, nan=np.nanmedian(img_ref))

    # subtract median filtered image to remove large structures and isolate banding
    img_med = fast_nanmedian(img_ref, size=filter_2d)
    # img_nan = img_nan | median_filter(img_nan.astype(np.uint8), size=filter_2d).astype(bool)
    img_ref = img_ref - img_med

    if filter_1d is False:
        # calculate median along axes to get banding profiles
        # this is more robust against picking up corona detail than local median filtering
        banding_vert = np.nanmedian(img_ref, axis=0)
        banding_vert = np.repeat(banding_vert[np.newaxis, :], img_ref.shape[0], axis=0)
    else:
        # calculate banding with a moving 1d median filter
        # can remove local banding variations better than global median
        # but is more prone to picking up and removing aligned corona detail

        #calculate standard deviation of the reference image to use for thresholding of remaining structures
        banding_vert = np.nanmedian(img_ref, axis=0)
        banding_vert = np.repeat(banding_vert[np.newaxis, :], img_ref.shape[0], axis=0)

        std_ref = np.nanstd(img_ref)

        # zero-out remaining structures above a certain amplitude
        remaining_structures = np.abs(img_ref-banding_vert) > std_ref * structures_threshold
        # dilate the remaining structures mask because not every pixel is caught by the threshold
        structure = np.ones((3,3), dtype=bool)
        remaining_structures = binary_dilation(remaining_structures, structure=structure)
        # fill these areas with the global median banding profile
        img_ref_ = img_ref.copy() # continue with copy so original is preserved for plotting
        img_ref_[remaining_structures] = banding_vert[remaining_structures]

        # img_ref = median_filter(img_ref, size=(1,filter_size))

        # 1d median along columns to remove outliers
        banding_vert = fast_nanmedian(img_ref_, size=filter_1d, axes=0)

    corrected_img = img - banding_vert
    
    return corrected_img

def correct_banding_dyadic(img, layers_2d=3, filter_1d=None, plotting=True):
    """
    Corrects banding along the first dimension in an image using specified filtering and averaging methods.

    Parameters:
    -----------
    img : 2D numpy array
        The input image to be corrected.
    saturation_mask : 2D numpy array
        A boolean mask indicating saturated pixels in the image.
    layers_2d : int, optional
        Number of dyadic layers of 2D median filtering to apply.
        3 layers corresponds to filter sizes of 3, 7, and 15.
    filter_1d : list of int | None, optional
        List of 1D filter sizes corresponding to each 2D layer.
        Default is None, which sets filter_1d to [filter_2d*7 for each layer].
    Returns:
    --------
    corrected_img : 2D numpy array
        The banding-corrected image.
    """

    if len(filter_1d) != layers_2d:
        raise ValueError("filter_1d list length must match layers_2d")
        
    for layer in range(layers_2d):
        # dyadic filter sizes:
        filter_2d = 2 * (2 ** (layer+1)) - 1  # dyadic progression of filter sizes: 3,7,15,...
        filter_1d_layer = filter_1d[layer] if filter_1d is not None else filter_2d * 7
        print(f"Applying banding correction to layer {layer+1}/{layers_2d} with 2D filter size {filter_2d}")
        corrected_img = correct_banding(img, filter_2d=filter_2d, filter_1d=filter_1d_layer, plotting=False)
        img = corrected_img

    return img

def correct_banding_splitrows(img, *args, **kwargs):
    img = img.copy()
    img[0::2, :] = correct_banding(img[0::2, :], *args, **kwargs) # top left
    img[1::2, :] = correct_banding(img[1::2, :], *args, **kwargs) # top right
    return img

def correct_banding_splitrows_dyadic(img, *args, **kwargs):
    img = img.copy()
    img[0::2, :] = correct_banding_dyadic(img[0::2, :], *args, **kwargs) # top left
    img[1::2, :] = correct_banding_dyadic(img[1::2, :], *args, **kwargs) # top right
    return img


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
    satlevel = np.max(y)
 
    tck = interpolate.splrep(y,x,s=0)
    Im1 = interpolate.splev(Im, tck, der=0)    
    
    if np.count_nonzero(OvrExpIdx)>0:
        print("    We found "+'{:0.0f}'.format(np.count_nonzero(OvrExpIdx))+" pixels with the value outside available data ("+'{:0.1f}'.format(np.max(y))+" [electron]). We clip them to the maximum possible value from the look-up table.")
        Im1[OvrExpIdx]=np.max(y)

    return Im1, satlevel, version_msg


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
