This is  S.Shestov's version of ASPIICS Level-3 processor

Running:

  cd Level-2-processor
  python3 l3_merge.py /PATH/TO/INPUT/file1.fits /PATH/TO/INPUT/file2.fits /PATH/TO/INPUT/file3.fits --outdir /PATH/TO/OUTPUT
  python3 l3_polariz.2.py /PATH/TO/INPUT/file_pol0.fits /PATH/TO/INPUT/file_pol60.fits /PATH/TO/INPUT/file_pol120.fits --outdir /PATH/TO/OUTPUT
  ... etc ...


Calibration data should be available for some of the modules, in particular polarization.

There are various command-line parameters, get help with ie python3 l3_merge.py --help
   --outdir OUTDIR
   --CRVAL1  100500.66  -- force CRVAL1
   --CRVAL2  200600.66  -- force CRVAL2
 
