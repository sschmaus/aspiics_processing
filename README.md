# Improved ASPIICS processing

Based on S.Shestov's version of ASPIICS Level 2 and 3 processor

<https://gitlab-as.oma.be/P3SC>

## How to use

1. Place the contents of the calibration [data release](https://gitlab-as.oma.be/P3SC/p3sc_calibration_data_repository/-/releases) 1.0 in the calibration-data directory
2. Use process_l2_batch.ipynb to process the L2 files
3. Use process_l3_batch.ipynb to process the L3 files

## Modifications

1. Fixed an issue with the saturated pixels not getting marked correctly. Now all out of range pixels from the nonlinearity correction will be flagged
2. Added smooth 30 px wide blending between layers during L3 product generation to smooth the transition
3. Added median based banding correction algorithm to correct bias residuals. It works in a multiscale approach, removing fine small noise first and then iteratively larger scale. Needs some fixes because it leaves residuals, both high frequency in bright regions of the image, and low frequency globally.

## Further thoughts

* The bias correction leaves obvious residuals. These are mostly static between frames but some lines vary a lot over time.
* The flatfield seems to contain a lot of bias. This results in stronger banding in bright areas. Apparently there is also a gain component to the banding component to the flat, because correcting the banding of the Flat makes things worse.
* There is another low frequency banding like artefact, but orthogonal to the normal banding. It seems to vary depending on which areas of the sensor are saturated. No idea for a correction yet.