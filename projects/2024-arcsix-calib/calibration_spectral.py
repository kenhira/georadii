"""
calibration_spectral.py

author: Ken Hirata (@kenhira)

This script runs the spectral calibration for the ARCSIX camera using the monochromator images.

Usage:
	python3 calibration_spectral.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from georadii.util import read_fits

from skimage.morphology import flood


if __name__ == "__main__":
	# Toggle bewteen ARCSIX vs Navy data
	id_key = 'ARCSIX'
	# id_key = 'Navy'

	# Fits file location
	fits_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/%s_Camera_Spectral_Cal_350_750/*.fits' % (id_key)

	# Wavelength list (must match the number of fits files)
	wavelength = np.arange(350., 750.01, 5.)
	
	# Toggle and npy file name for reloading the spectral calibration image files
	# use_npy = True
	use_npy = False
	npy_overwrite = True
	# npy_overwrite = False
	spec_npy_filename = 'npy/img_spec_%s.npy' % (id_key)

	# Pixel indices for cropping the region illuminated by the monochromator
	# ix0, ix1 = 0, 3080
	# iy0, iy1 = 0, 2096
	ix0, ix1 = 1604, 1613
	iy0, iy1 = 1084, 1124
	# ix0, ix1 = 1593, 1606
	# iy0, iy1 =  956,  972
	# area_test = True  ; itest = 24 # for testing the cropping region
	area_test = False

	# Output file save toggle
	savefile = True
	
	# Plot toggle
	plot = True
	# plot = False
	
	# Save/reuse the image data with npy file format
	if (not use_npy) or (use_npy and npy_overwrite):
		# Find and load all the relevant fits files for spectral calibration
		print('Reading .fits files in %s' % (fits_location))
		all_fits_files = sorted(glob.glob(fits_location))
		fimg, header = read_fits(all_fits_files[0])
		img_all = np.zeros((len(all_fits_files), fimg.shape[0], fimg.shape[1], fimg.shape[2])) # array to store all images
		for ifits, fits_file in enumerate(all_fits_files):
			fimg, header = read_fits(fits_file)
			img_all[ifits, :, :, :] = fimg
		
		if use_npy:
			print('Writing to %s ...' % (spec_npy_filename))
			dirname = 'npy'
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			with open(spec_npy_filename, 'wb') as f:
				np.save(f, img_all)
	else:
		print('Reading %s ...' % (spec_npy_filename))
		with open(spec_npy_filename, 'rb') as f:
			img_all = np.load(f)

	
	# Test the cropping region if necessary
	if area_test:
		test_img  = img_all[itest, :, :, :]
		test_img2 = img_all[itest, iy0:iy1, ix0:ix1, :]
		figtest = plt.figure()
		axt1 = figtest.add_subplot(121)
		axt1.imshow(test_img/np.max(test_img)*5.)
		axt2 = figtest.add_subplot(122)
		axt2.imshow(test_img2/np.max(test_img)*5.)
		plt.show()
		quit()

	# Average the region cropped for each of the wavelength
	spec_count_arr = np.mean(img_all[:, iy0:iy1, ix0:ix1, :], axis=(1, 2))


	# Reference files for light source and grating (spectral intensity and efficiency)
	input_filename = './data/1642448-k.txt'
	grating_filename = './data/efficiency_grating1.txt'
	with open(input_filename, 'r') as file:
		wvs, vals = [], []
		for line in file:
			parts = line.split()
			wvs.append(float(parts[0]))		# First column
			vals.append(float(parts[1]))	# Second column
	wv_inp = np.array(wvs)
	vl_inp = np.array(vals)
	with open(grating_filename, 'r') as file:
		wvs, vals = [], []
		for iline, line in enumerate(file):
			if line.startswith("#"):  # Skip header
				continue
			parts = line.split()
			wvs.append(float(parts[0]))		# First column
			vals.append(float(parts[1]))	# Second column
	wv_grt = np.array(wvs)
	vl_grt = np.array(vals)

	# Convert to the target wavelength range array
	# wv_arr = np.arange(350., 750., 0.1)
	wv_arr = np.arange(350., 1050., 0.1)
	vl_arr = np.interp(wv_arr, wv_inp, vl_inp, left=np.nan, right=np.nan)
	ef_arr = np.interp(wv_arr, wv_grt, vl_grt, left=np.nan, right=np.nan)*0.01

	# Relative intensity - seen by the camera
	relative_intensity = vl_arr*ef_arr

	if plot:
		fig = plt.figure(figsize=(10, 6))
		ax1 = fig.add_subplot(221)
		ax1.plot(wv_inp, vl_inp, color='black', marker='o')
		ax1.set_xlim(200., 1050.)
		ax1.set_ylim(0.0, None)
		ax1.set_xlabel(r'Wavelength $\rm (nm)$')
		ax1.set_ylabel(r'Input')
		ax2 = fig.add_subplot(223)
		ax2.plot(wv_grt, vl_grt, color='black', marker='o')
		ax2.set_xlim(200., 1050.)
		ax2.set_ylim(0.0, 100.)
		ax2.set_xlabel(r'Wavelength $\rm (nm)$')
		ax2.set_ylabel(r'Grating efficiency $\rm (\%)$')

		ax3 = fig.add_subplot(122)
		ax3.plot(wv_arr, relative_intensity, color='black')
		ax3.set_xlim(np.min(wv_arr), np.max(wv_arr))
		ax3.set_ylim(0.0, None)
		ax3.set_xlabel(r'Wavelength $\rm (nm)$')
		ax3.set_title('Input * effciency')

		plt.tight_layout()
		plt.show()

	# Convert to the target wavelength range array
	rel_intensity = np.interp(wavelength, wv_arr, relative_intensity)

	# Spectal response function
	response = spec_count_arr/rel_intensity[:, np.newaxis]

	# Discard wavelength range with no reference data
	out_wavelength = wavelength[~np.isnan(response[:, 0])]
	out_response   = response[~np.isnan(response[:, 0])]

	# Normalize the results (*5 because the measurments were done every 5 nm)
	out_response[:, 0] /= np.sum(out_response[:, 0])*5.
	out_response[:, 1] /= np.sum(out_response[:, 1])*5.
	out_response[:, 2] /= np.sum(out_response[:, 2])*5.

	wvl_avg = np.mean(out_response*out_wavelength[:, np.newaxis], axis=0)/np.mean(out_response, axis=0)
	print('Average wavelengths:', wvl_avg)


	# Save the response function file
	if savefile:
		dirname = 'out_spec_resp'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		output_file = '%s/response_%s.txt' % (dirname, id_key)
		with open(output_file, 'w') as file:
			file.write("# Wavelength (nm)\tResponse(R)\tResponse(G)\tResponse(B)\n")  # Header
			for iwl, wl in enumerate(out_wavelength):
				file.write("%5.1f\t%9.3e\t%9.3e\t%9.3e\n" % (wl, out_response[iwl, 0], out_response[iwl, 1], out_response[iwl, 2]))

	if plot:
		dirname = 'out_spec_resp'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		
		fig = plt.figure(figsize=(6, 5))
		ax1 = fig.add_subplot(111)
		ax1.plot(out_wavelength, out_response[:, 0], color='red'   , label='R: mean=%6.1f nm' % wvl_avg[0])
		ax1.plot(out_wavelength, out_response[:, 1], color='green' , label='G: mean=%6.1f nm' % wvl_avg[1])
		ax1.plot(out_wavelength, out_response[:, 2], color='blue'  , label='B: mean=%6.1f nm' % wvl_avg[2])
		ax1.set_ylim(0.0, 0.015)
		ax1.set_xlabel(r'Wavelength $\rm (nm)$')
		ax1.set_ylabel(r'Response $\rm (nm^{-1})$')
		ax1.set_title('%s' % id_key)
		ax1.legend()
		plt.tight_layout()
		plt.savefig('%s/out_%s.png' % (dirname, id_key), dpi=300)
		plt.show()
