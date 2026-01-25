from scipy.ndimage import median_filter, distance_transform_edt, binary_dilation
from skimage.restoration import denoise_nl_means
import sunkit_image.enhance as enhance
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def split_rows(Im):
    """Split image into odd and even rows for further processing"""
    Im_odd  = Im[0::2,:]   # rows 0,2,4,...  (1,3,5,... in FITS standard)
    Im_even = Im[1::2,:]   # rows 1,3,5,...  (2,4,6,... in FITS standard)
    return Im_odd, Im_even

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

    Fills NaNs by mirroring across nearest valid value before filtering.
    Uses vectorized_filter with np.nanmedian for images with uint16 type,
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
        img_filtered[~mask] = np.nan

        return img_filtered
    # elif img.dtype == np.uint16:
    #     # if uint16 type, use vectorized nanmedian
    #     print("Image with uint16 type passed to vectorized nanmedian.")

    #     mask = np.isfinite(img)

    #     r0 = mask.any(axis=1).argmax()
    #     r1 = mask.shape[0] - mask.any(axis=1)[::-1].argmax()
    #     c0 = mask.any(axis=0).argmax()
    #     c1 = mask.shape[1] - mask.any(axis=0)[::-1].argmax()
    #     cropped = img[r0:r1, c0:c1]

    #     # print(f"Cropped region for vectorized nanmedian: rows {r0}-{r1}, cols {c0}-{c1}")

    #     # apply vectorized nanmedian to cropped region
    #     cropped_filtered = vectorized_filter(cropped, size=size, function=np.nanmedian, *args, **kwargs)

    #     # place filtered cropped region back into all-NaN image
    #     img_filtered = np.full_like(img, np.nan)
    #     img_filtered[r0:r1, c0:c1] = cropped_filtered

    #     return img_filtered
    else:
        # if no NaNs, use faster median filter
        return median_filter(img, size=size, *args, **kwargs)

def _calculate_banding_from_reference(img_ref, filter_1d=False, axis=0):
    """Corrects banding along the first dimension in an image using specified filtering and averaging methods.

    Parameters:
    -----------
    img_ref : 2D numpy array
        A prefiltered version of the input image to use for banding profile calculation.
        Usually obtained by subtracting a large-scale median filtered version of img.
    filter_1d : int | False, optional
        Size of the median filter applied in 1D to remove outliers.
        Default is 4 * filter_2d.
        If False, no 1D filtering is applied and banding profile is computed for the entire line with np.nanmedian.
    Returns:
    --------
    banding : 2D numpy array
        The calculated banding profile to be subtracted from the input image.

    """

    if filter_1d is False:
        # calculate median along axes to get banding profiles
        # this is more robust against picking up corona detail than local median filtering
        # print("Calculating banding profile with global median (no 1D filtering)")
        banding = np.nanmedian(img_ref, axis=axis, keepdims=True)

        banding = np.repeat(banding, img_ref.shape[axis], axis=axis)
    else:
        # calculate banding with a moving 1d median filter
        # can remove local banding variations better than global median
        # but is more prone to picking up and removing aligned corona detail

        # 1d median along columns to remove outliers
        # print(f"Calculating banding profile with 1D median filter of size {filter_1d}")
        banding = fast_nanmedian(img_ref, size=filter_1d, axes=axis)
    
    return banding

def _integer_downsample(img, factor):
    """Downscale image by integer factor using average pooling."""
    if factor == 1:
        return img
    elif factor < 1:
        raise ValueError("Downscale factor must be an integer >= 1.")
    else:
        return np.nanmean(img.reshape((img.shape[0]//factor, factor, img.shape[1]//factor, factor)), axis=(1,3))

def _integer_upsample(img, factor):
    """Upscale image by integer factor using bilinear interpolation."""
    if factor == 1:
        return img
    elif factor < 1:
        raise ValueError("Upscale factor must be an integer >= 1.")
    else:
        return ndimage.zoom(img, int(factor), order=1, prefilter=False)


def correct_banding(img, filter_2d=15, filter_1d=None, ref_method="median", ref_downscale=1, structures_threshold=None, split_rows=False, plotting=False):
    """
    Corrects banding along the first dimension in an image using specified filtering and averaging methods.

    Parameters:
    -----------
    img : 2D numpy array
        The input image to be corrected.
    filter_2d : int, optional
        Size of the median filter applied in 2D to remove large structures.
    filter_1d : int | False, optional
        Size of the median filter applied in 1D to remove outliers.
        Default is 4 * filter_2d.
        If False, no 1D filtering is applied and banding profile is computed for the entire line with np.nanmedian.
    ref_method : str, optional
        Method to use for reference image generation. Options are "median" or "nlmeans".
    ref_downscale : int, optional
        Factor by which to downscale the reference image for banding profile calculation for faster processing.
        Default is 1 (no downscaling).
    structures_threshold : float, optional
        Threshold in standard deviation units to identify remaining structures.
        Default is 60 / filter_2d.
    split_rows : bool, optional
        If True, process odd and even rows separately to account for row-wise readout differences.
    plotting : bool, optional
        If True, generates diagnostic plots of the banding correction process.
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

    if ref_downscale > 1:
        # use average resampling to downscale image for faster processing
        img_ref = _integer_downsample(img_ref, ref_downscale)
        filter_2d = max(3, filter_2d // ref_downscale)
        if filter_1d is not False:
            filter_1d = max(3, filter_1d // ref_downscale)
        print(f"Downscaling reference image by factor {ref_downscale} for banding calculation, new filter sizes: 2D={filter_2d}, 1D={filter_1d}")


    if ref_method == "median":
        # use a median filtered image as reference for the banding profile calculation
        # this is prone to residual structures around high contast features
        img_med = fast_nanmedian(img_ref, size=filter_2d)
        img_ref = img_ref - img_med

    elif ref_method == "nlmeans":
        # use non-local means denoising to isolate banding profile
        # this is more robust against residual structures than median filtering

        print("Using non-local means denoising for banding reference")

        sigma = np.nanstd(img_ref)
        # fill NaNs before applying nlmeans to avoid edge artifacts and NaN bleeding
        img_ref_denoised = fill_nan_mirror(img_ref)
        img_ref_denoised = denoise_nl_means(img_ref_denoised, h=0.8*sigma, fast_mode=True,
                                            patch_size=filter_2d//2, patch_distance=filter_2d//2, preserve_range=True)
        img_ref = img_ref - img_ref_denoised


    if structures_threshold is not False:
        # zero-out remaining structures above a certain amplitude
        std_ref = np.nanstd(img_ref)
        remaining_structures = np.abs(img_ref) > std_ref * structures_threshold
        # dilate the remaining structures mask because not every pixel is caught by the threshold
        structure = np.ones((3,3), dtype=bool)
        remaining_structures = binary_dilation(remaining_structures, structure=structure)
        # fill these areas with the global median banding profile

        img_ref[remaining_structures] = np.nanmedian(img_ref)


    banding = np.zeros_like(img)

    # calculate vertical banding profile, accounting for split rows and reference downscaling
    if not split_rows:
        banding_vert = _calculate_banding_from_reference(img_ref, filter_1d=filter_1d, axis=0)
        if ref_downscale > 1:
            banding_vert = _integer_upsample(banding_vert, ref_downscale)
    else:
        banding_vert = np.zeros_like(img)
        if filter_1d is not False:
            filter_1d = max(3, filter_1d // 2)
        banding_vert_odd = _calculate_banding_from_reference(img_ref[0::2, :], filter_1d=filter_1d, axis=0)
        banding_vert_even = _calculate_banding_from_reference(img_ref[1::2, :], filter_1d=filter_1d, axis=0)
        if ref_downscale > 1:
            banding_vert_odd  = _integer_upsample(banding_vert_odd,  ref_downscale)
            banding_vert_even = _integer_upsample(banding_vert_even, ref_downscale)
        banding_vert[0::2, :] = banding_vert_odd
        banding_vert[1::2, :] = banding_vert_even
    banding += banding_vert

    # calculate horizontal banding profile
    banding_horiz = _calculate_banding_from_reference(img_ref, filter_1d=filter_1d, axis=1)
    if ref_downscale > 1:
        banding_horiz = _integer_upsample(banding_horiz, ref_downscale)
    banding += banding_horiz

    # subtract banding profile from original image
    corrected_img = img - banding

    if plotting:
        # plotting for debugging
        subtext = f"Reference method: {ref_method}, 2D filter size: {filter_2d}, Reference downscale: {ref_downscale}, 1D filter size: {filter_1d}"
        wd = (256,768,256,768)
        img_ref = _integer_upsample(img_ref, ref_downscale)
        _plot_banding_correction_reference(img, img_ref, banding, split_rows, wd, title=f"{ref_method} banding correction", subtext=subtext)
    
    return corrected_img

def correct_banding_multiscale(img, ref_method ="median", layers_2d=[3,7,15], ref_downscale=None, filter_1d=None, split_rows=False, plotting=True):
    """
    Corrects banding along the first dimension in an image using specified filtering and averaging methods.

    Parameters:
    -----------
    img : 2D numpy array
        The input image to be corrected.
    ref_method : str, optional
        Method to use for reference image generation. Options are "median" or "nlmeans".
        Default is "median".
    saturation_mask : 2D numpy array
        A boolean mask indicating saturated pixels in the image.
    layers_2d : int, optional
        List of filter sizes for the 2D median filtering layers to apply.
        Default is [3,7,15]
    filter_1d : list of int | None, optional
        List of 1D filter sizes corresponding to each 2D layer.
        Default is None, which sets filter_1d to [filter_2d*7 for each layer].
    ref_downscale : list of int, optional
        Factor by which the reference image is downscaled before banding calculation.
        Default is 1 (no downscaling).
    Returns:
    --------
    corrected_img : 2D numpy array
        The banding-corrected image.
    """

    if len(filter_1d) != len(layers_2d):
        raise ValueError("filter_1d list length must match layers_2d")
    
    if plotting:
        img_orig = img.copy()

    if ref_downscale is None:
        ref_downscale = [1 for _ in layers_2d]
        
    for n, filter_2d in enumerate(layers_2d):
        # multiscale filter sizes:
        filter_1d_layer = filter_1d[n] if filter_1d is not None else filter_2d * 7
        print(f"Applying banding correction with 2D filter size {filter_2d}")
        corrected_img = correct_banding(img, ref_method=ref_method, filter_2d=filter_2d, filter_1d=filter_1d_layer, ref_downscale=ref_downscale[n], split_rows=split_rows, plotting=False)
        img = corrected_img

    if plotting:
        # plotting for debugging
        subtext = f"Reference method: {ref_method}, 2D filter sizes: {layers_2d}, Reference downscale: {ref_downscale}, 1D filter sizes: {filter_1d}"
        wd = (256,768,256,768)
        img_ref = _integer_upsample(img_ref, ref_downscale)
        _plot_banding_correction_reference(img, img_ref, banding, split_rows, wd, title=f"{ref_method} banding correction", subtext=subtext)

    if plotting:
        # plotting for debugging
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        # wd = (256,512,512,1024)
        wd = (256,768,256,768)

        aspect = 1
        if split_rows:
            wd = (wd[0]*2, wd[1]*2, wd[2], wd[3])
            #set y axis scale to 0.5
            aspect = 0.5

        img_med = fast_nanmedian(img_orig, size=max(layers_2d))
        img_ref_ = img_orig - img_med

        std_ref = np.nanstd(img_ref_)
        mean_ref = np.nanmean(img_ref_)
        
        axes[0].imshow(img_ref_[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin=mean_ref - 1*std_ref, vmax=mean_ref + 3*std_ref)
        axes[0].set_title("Reference image for banding calculation")
        
        axes[1].imshow((corrected_img-img_med)[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin=mean_ref - 1*std_ref, vmax=mean_ref + 3*std_ref)
        axes[1].set_title("Reference image after banding correction")
        
        sigma = [2, 4, 16, 64]
        weights = [0.125, 0.25, 0.5, 0.5]
        k = 0.1
        gamma = 3.5

        mgn = enhance.mgn(corrected_img[wd[0]:wd[1],wd[2]:wd[3]], h=0, sigma=sigma, k=k, weights=weights, gamma=gamma)
        axes[2].imshow(mgn, aspect=aspect)
        axes[2].set_title("Corrected image with MGN filter")

        axes[3].imshow((img_orig-corrected_img)[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin= -1*std_ref, vmax= 3*std_ref)
        axes[3].set_title("Banding profile")

        # add subtext with parameters used
        subtext = f"Reference method: {ref_method}, 2D filter sizes: {layers_2d}, Reference downscale: {ref_downscale}, 1D filter sizes: {filter_1d}"
        fig.text(0.5, 0.0, subtext, ha="center", fontsize=10)
        
        plt.tight_layout()
        plt.show()

    return img



def correct_banding_splitrows(img, *args, **kwargs):
    """Wrapper to apply correct_banding separately to odd and even rows.
    
    Using correct_banding with split_rows=True is preferred over this function.
    """
    img = img.copy()
    img[0::2, :] = correct_banding(img[0::2, :], *args, **kwargs) # top left
    img[1::2, :] = correct_banding(img[1::2, :], *args, **kwargs) # top right
    return img

def correct_banding_splitrows_multiscale(img, *args, **kwargs):
    """Wrapper to apply correct_banding_multiscale separately to odd and even rows.#
    
    Using correct_banding_multiscale with split_rows=True is preferred over this function.
    """
    img = img.copy()
    img[0::2, :] = correct_banding_multiscale(img[0::2, :], *args, **kwargs) # top left
    img[1::2, :] = correct_banding_multiscale(img[1::2, :], *args, **kwargs) # top right
    return img

def _mgn_enhance(img):
    """Helper function to apply MGN enhancement to an image for plotting."""

    sigma = [2, 4, 16, 64]
    weights = [0.125, 0.25, 0.5, 0.5]
    k = 0.1
    gamma = 3.5

    nan_mask = np.isnan(img)
    img = fill_nan_mirror(img)
    img = np.nan_to_num(img, nan=np.nanmedian(img))
    mgn = enhance.mgn(img, h=0, sigma=sigma, k=k, weights=weights, gamma=gamma)
    mgn[nan_mask] = np.nan
    
    return mgn

def _plot_banding_correction_reference(img_orig, img_ref, banding_profile, split_rows=False, window=(256,768,256,768), title="", subtext=""):
    """Helper function to plot banding correction results with the banding reference image.

    Parameters:
    -----------
    img_orig : 2D numpy array
        The original input image before banding correction.
    img_ref : 2D numpy array
        The reference image used for banding profile calculation.
    banding_profile : 2D numpy array
        The calculated banding profile that was subtracted from the input image.
    split_rows : bool, optional
        If True, indicates that odd and even rows were processed separately.
    window : tuple, optional
        A tuple specifying the (row_start, row_end, col_start, col_end) of the image region to display.
    title : str, optional
        Title for the plot.
    subtext : str, optional
        Subtext to display below the plot with parameters used.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    # wd = (256,512,512,1024)
    # wd = (256,768,256,768)
    wd = window

    aspect = 1
    if split_rows:
        wd = (wd[0]*2, wd[1]*2, wd[2], wd[3])
        #set y axis scale to 0.5
        aspect = 0.5

    std_ref = np.nanstd(img_ref)
    mean_ref = np.nanmean(img_ref)
    
    axes[0].imshow(img_ref[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin=mean_ref - 1*std_ref, vmax=mean_ref + 3*std_ref)
    axes[0].set_title("Reference image for banding calculation")
    
    axes[1].imshow((img_ref-banding_profile)[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin=mean_ref - 1*std_ref, vmax=mean_ref + 3*std_ref)
    axes[1].set_title("Reference image after banding correction")
    
    mgn = _mgn_enhance((img_orig-banding_profile)[wd[0]:wd[1],wd[2]:wd[3]])
    axes[2].imshow(mgn, aspect=aspect)
    axes[2].set_title("Corrected image with MGN filter")

    axes[3].imshow(banding_profile[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin= -1*std_ref, vmax= 3*std_ref)
    axes[3].set_title("Banding profile")

    # add subtext with parameters used
    fig.text(0.5, 0.0, subtext, ha="center", fontsize=10)
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def _plot_banding_correction_before_after(img_orig, img_corrected, split_rows=False, window=(256,768,256,768), title="", subtext=""):
    """Helper function to plot banding correction results from before and after images.

    Parameters:
    -----------
    img_orig : 2D numpy array
        The original input image before banding correction.
    img_corrected : 2D numpy array
        The banding corrected image.
    split_rows : bool, optional
        If True, indicates that odd and even rows were processed separately.
    window : tuple, optional
        A tuple specifying the (row_start, row_end, col_start, col_end) of the image region to display.
    title : str, optional
        Title for the plot.
    subtext : str, optional
        Subtext to display below the plot with parameters used.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    # wd = (256,512,512,1024)
    # wd = (256,768,256,768)
    wd = window

    aspect = 1
    if split_rows:
        wd = (wd[0]*2, wd[1]*2, wd[2], wd[3])
        #set y axis scale to 0.5
        aspect = 0.5

    banding_profile = img_orig - img_corrected
    img_orig = _mgn_enhance(img_orig[wd[0]:wd[1],wd[2]:wd[3]])
    img_corrected = _mgn_enhance(img_corrected[wd[0]:wd[1],wd[2]:wd[3]])

    std_mgn = np.nanstd(img_orig)

    pct99_banding = np.nanpercentile(np.abs(banding_profile), 99.9)
    
    axes[0].imshow(img_orig, aspect=aspect, vmin=-3*std_mgn, vmax=3*std_mgn)
    axes[0].set_title("Original image (MGN filtered)")
    
    axes[1].imshow(img_corrected, aspect=aspect, vmin=-3*std_mgn, vmax=3*std_mgn)
    axes[1].set_title("Corrected image (MGN filtered)")

    axes[2].imshow((banding_profile)[wd[0]:wd[1],wd[2]:wd[3]], aspect=aspect, vmin=-pct99_banding, vmax=pct99_banding)
    axes[2].set_title("Banding profile")

    # add subtext with parameters used
    fig.text(0.5, 0.0, subtext, ha="center", fontsize=10)
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()