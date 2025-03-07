"""
test_exptime.py

author: Ken Hirata (@kenhira)

This script obtains statistics on the exposure time of the ARCSIX field camera data.

Usage:
	python3 test_exptime.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from georadii.util import read_fits

if __name__ == "__main__":
	overwrite = True
	if overwrite:
		# all_fits_files = glob.glob('/Volumes/ARCSIX2KSS/ARCSIX_RF*/**/Capture_*.fits')
		# all_fits_files = glob.glob('/Volumes/ARCSIX4KSS/ARCSIX_RF16_2024_08_07/Camera/Capture 2024-08-07T10_45_08Z/Capture_*.fits')
		# all_fits_files = glob.glob('/Volumes/ARCSIX4KSS/ARCSIX_RF17_2024_08_08/Camera/Capture 2024-08-08T10_47_43Z/Capture_*.fits')
		# all_fits_files = glob.glob('/Volumes/ARCSIX4KSS/ARCSIX_RF18_2024_08_09/Camera/Capture 2024-08-09T10_55_13Z/Capture_*.fits')
		# all_fits_files = glob.glob('/Volumes/ARCSIX4KSS/ARCSIX_RF19_2024_08_15/Camera/Capture 2024-08-15T09_59_02Z/Capture_*.fits')
		all_fits_files = glob.glob('/Volumes/ARCSIX4KSS/LeipzigCalibrations/Navy_radiometric_0.*ms/*/*.fits')
		exptime_arr = np.zeros(len(all_fits_files))
		for ifits, fits_file in enumerate(all_fits_files):
			_, header = read_fits(fits_file, header_only=True)
			exptime = header['EXPTIME']
			exptime_arr[ifits] = exptime*1000. #ms
			print('%06d/%d : %f' % (ifits + 1, len(all_fits_files), exptime))
		dirname = 'npy'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		with open('npy/exptime2', 'wb') as f:
			np.save(f, exptime_arr)
	else:
		with open('npy/exptime2', 'rb') as f:
			exptime_arr = np.load(f)
	
	plt.hist(exptime_arr)
	plt.show()
