This is  S.Shestov's version of ASPIICS Level-2 processor

Running:

  cd Level-2-processor
  python3 l2_master.py /PATH/TO/INPUT/aspiics_wb_l1_bar_foo.fits  --outdir /PATH/TO/OUTPUT


Calibration data should be copied/linked to this folder, such that calibr_data.json exists in the current directory and paths specified within this file are correct.

There are various command-line parameters, get help with python3 l2_master.py --help
  --filter "Wideband"           -- force Particular filter ("Wideband", "He I", "Fe XIV", "Polarizer 0", "Polarizer 60", "Polarizer 120")
  -C or --cal calibr_file.json  -- use calibration file
  --mark_IO  True/False         -- mark IO in the image
  --mark_suncenter  True/False  -- mark solar center in the image

