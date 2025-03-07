"""
calibration_radiometric.py

author: Ken Hirata (@kenhira)

This script runs the radiometric calibration of the ARCSIX camera from the integrating sphere image set.

Usage:
	python3 calibration_radiometric.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from skimage.morphology import flood

from georadii.util import read_fits

### Global definition
# Center pixel of the camera
# alpha0, beta0 = 1594.75, 933.27 # ARCSIX astrometry based
alpha0, beta0 = 1596.11, 934.36  # ARCSIX checkerboard based
# alpha0, beta0 = 1561, 1048 # Navy ?

# Radius of the valid region around the center (in pixel)
rad  = 850.

######

def read_img_and_dark(fits_location, dark_fits_location, use_npy=False, npyoverwrite=False):
	if (not use_npy) or (use_npy and npy_overwrite):
		all_dark_fits_files = sorted(glob.glob(dark_fits_location))[:]
		fimg, header = read_fits(all_dark_fits_files[0], flipud=True, fliplr=True)
		xx, yy = np.meshgrid(np.arange(fimg.shape[1]), np.arange(fimg.shape[0]))
		imgsum_arr = np.zeros_like(fimg[:, :, :], dtype=float)
		for ifits, fits_file in enumerate(all_dark_fits_files):
			fimg, header = read_fits(fits_file, flipud=True, fliplr=True)
			# fimg /= float(2**16)
			fimg_main = fimg[:, :, :]
			fimg_main[yy > np.max(yy) - 50] = np.nan # Ignore the timestamp portion of the image
			imgsum_arr[:, :, :] += fimg_main
		imgdark = imgsum_arr/len(all_dark_fits_files)
		# print('dark:', np.nanmin(imgdark), np.nanmax(imgdark), np.nanmean(imgdark), np.nanstd(imgdark))

		if ist is None or ien is None:
			all_fits_files = sorted(glob.glob(fits_location))[:]
		else:
			all_fits_files = sorted(glob.glob(fits_location))[ist:ien:intv]
		fimg, header = read_fits(all_fits_files[0], flipud=True, fliplr=True)
		img_all = np.zeros((len(all_fits_files), fimg.shape[0], fimg.shape[1], fimg.shape[2])) # array to store all images
		
		for ifits, fits_file in enumerate(all_fits_files):
			fimg, header = read_fits(fits_file, flipud=True, fliplr=True)
			img_all[ifits, :, :, :] = fimg
		
		if use_npy:
			print('Writing to %s ...' % (sum_npy_fname))
			dirname = 'npy'
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			with open(sum_npy_fname, 'wb') as f:
				np.save(f, img_all)
				np.save(f, imgdark)
	else:
		print('Reading %s ...' % (sum_npy_fname))
		with open(sum_npy_fname, 'rb') as f:
			img_all = np.load(f)
			imgdark = np.load(f)
	return img_all, imgdark

# Function to calculate the integrating sphere radiance that the camera should observe
# by multiplying the sphere datasheet radiance and the spectral response function 
def get_ref_rad(id_key, scale):
	response_filename = '../../data/spectral/cam/response_%s.txt' % (id_key)
	with open(response_filename, 'r') as file:
		wvs, vals = [], [[], [], []]
		for iline, line in enumerate(file):
			if line.startswith("#"):  # Skip header
				continue
			parts = line.split()
			wvs.append(float(parts[0]))		# First column
			vals[0].append(float(parts[1]))	# Second column
			vals[1].append(float(parts[2]))	# Second column
			vals[2].append(float(parts[3]))	# Second column
	wv_res = np.array(wvs)
	vl_res = np.array(vals)

	spec_resp = np.zeros((3, len(wvs)))
	spec_resp[0, :] = vl_res[0, :]/np.sum(vl_res[0, :])/(wv_res[1] - wv_res[0])
	spec_resp[1, :] = vl_res[1, :]/np.sum(vl_res[1, :])/(wv_res[1] - wv_res[0])
	spec_resp[2, :] = vl_res[2, :]/np.sum(vl_res[2, :])/(wv_res[1] - wv_res[0])

	input_filename = './data/ISS-28-P-15310287_kaldata.txt'
	
	with open(input_filename, 'r') as file:
		wvs, vals = [], []
		for line in file:
			if line.startswith("#"):  # Skip header
				continue
			parts = line.split()
			wvs.append(float(parts[0]))		# First column
			vals.append(float(parts[1]))	# Second column
	wv_inp = np.array(wvs)
	vl_inp = np.array(vals)

	vl_inp *=  scale #2.0189/1.7316

	spec_int = np.interp(wv_res, wv_inp, vl_inp)
	weighted_int = spec_int[np.newaxis, :]*spec_resp[:, :]
	target_int = np.trapezoid(weighted_int, x=wv_res, axis=1)

	return target_int

# Function to crop out pixel group with a certain threshold and derive the 
# sum of RGB values and number of images available for a given pixel.
def clipper(img_all, tol=0.5, norm_arr=None, verbose=False):
	imgsum_arr = np.zeros_like(img_all[0, :, :, :], dtype=float)
	count_arr  = np.zeros_like(img_all[0, :, :, 0], dtype=int)
	if norm_arr is None:
		norm_arr = np.ones_like(imgsum_arr)
	for ifits in range(img_all.shape[0]):
		fimg = img_all[ifits, :, :, :]#/float(2**16)

		xx, yy = np.meshgrid(np.arange(img_all.shape[2]), np.arange(img_all.shape[1]))

		fimg_main = fimg[:, :, :]*norm_arr[:, :, :]
		fimg_main[yy > np.max(yy) - 50] = np.nan #  Ignore the timestamp region

		indmax = np.unravel_index(np.nanargmax(fimg_main[:, :, 1]), fimg_main[:, :, 1].shape)
		area = flood(fimg_main[:, :, 1], indmax, tolerance=tol*np.nanmax(fimg_main[:, :, 1]))
		
		aperture = area & (yy < np.max(yy) - 50)

		if verbose:
			fmax = np.nanmax(fimg_main)
			fig = plt.figure(figsize=(12, 5))
			ax1 = fig.add_subplot(131)
			ax1.imshow(fimg_main/fmax, origin='lower')
			ax2 = fig.add_subplot(132)
			ax2.imshow(fimg_main/fmax, origin='lower')
			ax2.contourf(aperture, 1, hatches=['', '///'], origin='lower', color='white', alpha=0.)
			fimg_0 = fimg_main[:, :, :]
			fimg_0[~aperture] = np.nan
			ax3 = fig.add_subplot(133)
			ax3.imshow(fimg_0/fmax, origin='lower')
			plt.savefig('out_clipper_%03d.png' % (ifits), dpi=300)

		count_arr[aperture] += 1
		imgsum_arr[aperture, :] += fimg[aperture, :]
	return imgsum_arr, count_arr

def fit_polynomial(dist, coefficient):
	def fit_func(x, c0, c2, c4, c6):
		return c0 + c2*x**2. + c4*x**4. + c6*x**6.
	popt, pcov = optimize.curve_fit(fit_func, dist.flatten(), coefficient[:, :, 0].flatten(), nan_policy='omit')#, bounds=(0, [3., 1., 0.5]))
	print('0:', popt)
	predicted_c[:, :, 0] = fit_func(dist, *popt)
	popt, pcov = optimize.curve_fit(fit_func, dist.flatten(), coefficient[:, :, 1].flatten(), nan_policy='omit')#, bounds=(0, [3., 1., 0.5]))
	print('1:', popt)
	predicted_c[:, :, 1] = fit_func(dist, *popt)
	popt, pcov = optimize.curve_fit(fit_func, dist.flatten(), coefficient[:, :, 2].flatten(), nan_policy='omit')#, bounds=(0, [3., 1., 0.5]))
	print('2:', popt)
	predicted_c[:, :, 2] = fit_func(dist, *popt)
	return predicted_c

def func_quad(param, inp):
	c0 = param[0]
	c2 = param[1]
	x = inp[0, :]
	y = inp[1, :]
	d = np.sqrt((x - alpha0)**2. + (y - beta0)**2.)
	return c0 + c2*d**2.

def fit_func_quad(param, inp):
	v = inp[2, :]
	vpred = func_quad(param, inp)
	return (vpred - v)**2.

def func_6th(param, inp):
	c0 = param[0]
	c2 = param[1]
	c4 = param[2]
	c6 = param[3]
	x = inp[0, :]
	y = inp[1, :]
	d = np.sqrt((x - alpha0)**2. + (y - beta0)**2.)
	return c0 + c2*d**2. + c4*d**4. + c6*d**6.

def fit_func_6th(param, inp):
	v = inp[2, :]
	vpred = func_6th(param, inp)
	return (vpred - v)**2.

def func_oblique(param, inp):
	alpha = param[0]
	beta = param[1]
	c0 = param[2]
	c2 = param[3]
	c4 = param[4]
	c6 = param[5]
	x = inp[0, :]
	y = inp[1, :]

	## (x - a)**2 + (y - b)**2 = d**2
	## a = - (alpha - alpha0)*d/rad + alpha
	## b = - ( beta -  beta0)*d/rad + beta
	## Thus, a = alpha0 & b = beta0 at d = rad (outer circle)
	##  and, a = alpha  & b = beta  at d = 0   (at the (shifted) center)
	
	# Solve for d from the above equation
	a_ = ((alpha - alpha0)**2. + (beta - beta0)**2. - rad*rad)/(rad*rad)
	b_ = ((alpha - alpha0)*(x - alpha) + (beta - beta0)*(y - beta))/rad
	c_ = (x - alpha)**2. + (y - beta)**2.
	d = (-b_ - np.sqrt(b_*b_ - a_*c_))/a_
	vpred = c0 + c2*d**2. + c4*d**4. + c6*d**6.
	return vpred

def fit_func_oblique(param, inp):
	v = inp[2, :]
	vpred = func_oblique(param, inp)
	# return np.sum((vpred - v)**2.)
	return (vpred - v)**2.

if __name__ == "__main__":
	# Toggle for execution
	execute_step1 = True
	# execute_step1 = False
	execute_step2 = True

	### Step 1: Load the images and fill the hemisphere with them

	# Toggle bewteen ARCSIX vs Navy data
	id_key = 'ARCSIX'
	# id_key = 'Navy'

	# Measurement settings (exposure time set and scaling factor)
	exptime_rec = 0.25
	intensity_scale = 1.9761/1.7316 # 0.25 ms data was taken on Dec 16, 2024
	# # exptime_rec = 0.20
	# # exptime_rec = 0.30
	# exptime_rec = 0.50
	# intensity_scale = 2.0189/1.7316 # 0.2, 0.3, 0.5 ms data was taken on Dec 17, 2024
	
	# Fits file location
	suffix = ''
	fits_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/%s_radiometric_%4.2fms/*/*.fits' % (id_key, exptime_rec)
	dark_fits_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/%s_dark*_%4.2fms/*/*.fits' % (id_key, exptime_rec)
	# # suffix = '_1'
	# # suffix = '_2'
	# suffix = '_3'
	# fits_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/other_20241216/%s_radiometric_%4.2fms%s/*/*.fits' % (id_key, exptime_rec, suffix)
	# dark_fits_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/other_20241216/%s_dark*_%4.2fms%s/*/*.fits' % (id_key, exptime_rec, suffix)

	# Test the cropping for identifying the integrating sphere aperture
	# crop_test = True
	crop_test = False
	# ist, ien, intv = 0, 1, 1
	ist, ien = None, None

	# Name of the coefficient npy file
	coef_npy_name = 'coefficient.npy'

	### Step 2: Apply a fitting function to the coefficient (inverse sensitivity)

	# Whether to read the Camp2Ex data or not
	read_camp2ex_cal = False
	# read_camp2ex_cal = True
	# radiometric_response = '%s/Downloads/cam_sebastian/calibration_factors_20190522.nc' % (os.getenv('HOME'))

	# Type of the fitting function
	# fit_mode = 'quad'
	# fit_mode = '6th'
	fit_mode = 'oblique_6th' # oblique code model

	# Whether to plot
	plot = True

	# (It is found that exposure times of 0.20 ms and 0.25 ms both cause the same exposure time)
	# if exptime_rec == 0.20:
	# 	exptime = 0.25
	# else:
	# 	exptime = exptime_rec
	exptime = exptime_rec

	if execute_step1:
		# Save/reuse the image data with npy file format
		img_all, imgdark = read_img_and_dark(fits_location, dark_fits_location, use_npy=False, npyoverwrite=True) # does not use npy
		# sum_npy_fname = 'npy/img_rad_%s_%4.2f.npy' % (id_key, exptime_rec)
		# img_all, imgdark = read_img_and_dark(fits_location, dark_fits_location, use_npy=True, npyoverwrite=True, sum_npy_fname=sum_npy_fname)  # save into npy
		# img_all, imgdark = read_img_and_dark(fits_location, dark_fits_location, use_npy=True, npyoverwrite=False, sum_npy_fname=sum_npy_fname) # use the saved npy

		# Define the distance to the center pixel, which will be use later
		centerpix = np.int_(np.round(np.array([alpha0, beta0])))
		xx, yy = np.meshgrid(np.arange(img_all.shape[2]), np.arange(img_all.shape[1]))
		dist = np.sqrt((xx - centerpix[0])**2. + (yy - centerpix[1])**2.)
		
		# Obtain the radiance that the camera should see through the integarting sphere aperture
		target_int = get_ref_rad(id_key, intensity_scale)

		## Iteration No. 1 (use the original images to select the )
		
		# Extract the illuminated pixels using a thresholind method and get the RGB sum and number of those
		imgsum_arr, count_arr = clipper(img_all, tol=0.5) # tol is a tuning parameter to properly pick the aperture
		rgb_arr = imgsum_arr[:, :, :]/count_arr[:, :, np.newaxis]
		rgb_arr -= imgdark # Subtract the dark values (though it is negligibly small)

		# Derive the coefficient (inverse sensitivity)
		# coefficient = target_int[np.newaxis, np.newaxis, :]/(rgb_arr[:, :, :]*float(2**16))/(exptime*0.001)
		coefficient = target_int[np.newaxis, np.newaxis, :]*exptime*0.001/rgb_arr[:, :, :]

		# Fit the coefficient with polynomial
		predicted_c = np.zeros((imgsum_arr.shape[0], imgsum_arr.shape[1], 3))
		predicted_c = fit_polynomial(dist, coefficient)

		# Predcit the RGB value
		# predicted_rgb = target_int[np.newaxis, np.newaxis, :]/predicted_c[:, :, :]/(exptime*0.001)/float(2**16)
		predicted_rgb = target_int[np.newaxis, np.newaxis, :]*exptime*0.001/predicted_c[:, :, :]
		rgb_diff = (predicted_rgb - rgb_arr)/rgb_arr*100.

		if plot:
			fig02 = plt.figure(figsize=(5, 3.5))
			ax02 = fig02.add_subplot(111)
			cd1 = ax02.imshow(count_arr, origin='lower')
			ax02.scatter(centerpix[0], centerpix[1], marker='+', color='red')
			cb1 = fig02.colorbar(cd1, ax=ax02)
			cb1.set_label('# of images')
		
		if plot:
			fig1 = plt.figure(figsize=(10, 7))
			ax1 = fig1.add_subplot(221)
			# cd2 = ax1.imshow(rgb_arr[:, :, 1], origin='lower')
			cd2 = ax1.contourf(xx, yy, rgb_arr[:, :, 1], vmin=np.nanmin(rgb_arr[:, :, 1]), vmax=np.nanmax(rgb_arr[:, :, 1]), levels=17)
			ax1.scatter(centerpix[0], centerpix[1], marker='+', color='red')
			ax1.set_aspect('equal')
			cb2 = fig1.colorbar(cd2, ax=ax1)
			cb2.set_label('RGB value')

			ax7 = fig1.add_subplot(222)
			# cd7 = ax7.imshow(rgb_diff[:, :, 1], vmin=-np.nanmax(np.abs(rgb_diff[:, :, 1])), vmax=np.nanmax(np.abs(rgb_diff[:, :, 1])), cmap='seismic', origin='lower')
			cd7 = ax7.contourf(xx, yy, rgb_diff[:, :, 1], vmin=-np.nanmax(np.abs(rgb_diff[:, :, 1])), vmax=np.nanmax(np.abs(rgb_diff[:, :, 1])), levels=17, cmap='seismic')
			ax7.scatter(centerpix[0], centerpix[1], marker='+', color='black')
			ax7.set_aspect('equal')
			cb7 = fig1.colorbar(cd7, ax=ax7)
			cb7.set_label('RGB value prediction error (%)')

		## Iteration No. 2 (use the quadratically corrected image to pick up the )

		# Extract the illuminated pixels using a thresholind method and get the RGB sum and number of those
		imgsum_arr, count_arr = clipper(img_all, tol=0.25, norm_arr=predicted_c) # tol is a tuning parameter to properly pick the aperture
		imgsum_arr -= imgdark
		rgb_arr = imgsum_arr[:, :, :]/count_arr[:, :, np.newaxis]

		# Derive the coefficient (inverse sensitivity)
		# coefficient = target_int[np.newaxis, np.newaxis, :]/(rgb_arr[:, :, :]*float(2**16))/(exptime*0.001)
		coefficient = target_int[np.newaxis, np.newaxis, :]*exptime*0.001/rgb_arr[:, :, :]

		# Write out the coefficient file
		print('Writing to %s ...' % (coef_npy_name))
		dirname = 'npy'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		with open('npy/%s' % (coef_npy_name), 'wb') as f:
			np.save(f, xx)
			np.save(f, yy)
			np.save(f, coefficient)
		
		# Fit the coefficient with polynomial
		predicted_c = np.zeros((imgsum_arr.shape[0], imgsum_arr.shape[1], 3))
		predicted_c = fit_polynomial(dist, coefficient)

		# Predcit the RGB value
		# predicted_rgb = target_int[np.newaxis, np.newaxis, :]/predicted_c[:, :, :]/(exptime*0.001)/float(2**16)
		predicted_rgb = target_int[np.newaxis, np.newaxis, :]*exptime*0.001/predicted_c[:, :, :]
		rgb_diff = (predicted_rgb - rgb_arr)/rgb_arr*100.

		if plot:
			ax5 = fig1.add_subplot(223)
			# cd5 = ax5.imshow(rgb_arr[:, :, 1], origin='lower')
			cd5 = ax5.contourf(xx, yy, rgb_arr[:, :, 1], vmin=np.nanmin(rgb_arr[:, :, 1]), vmax=np.nanmax(rgb_arr[:, :, 1]), levels=17)
			ax5.scatter(centerpix[0], centerpix[1], marker='+', color='red')
			ax5.set_aspect('equal')
			cb5 = fig1.colorbar(cd5, ax=ax5)
			cb5.set_label('RGB value')

			ax9 = fig1.add_subplot(224)
			# cd9 = ax9.imshow(rgb_diff[:, :, 1], vmin=-np.nanmax(np.abs(rgb_diff[:, :, 1])), vmax=np.nanmax(np.abs(rgb_diff[:, :, 1])), cmap='seismic', origin='lower')
			cd9 = ax9.contourf(xx, yy, rgb_diff[:, :, 1], vmin=-np.nanmax(np.abs(rgb_diff[:, :, 1])), vmax=np.nanmax(np.abs(rgb_diff[:, :, 1])), levels=17, cmap='seismic')
			ax9.scatter(centerpix[0], centerpix[1], marker='+', color='black')
			ax9.set_aspect('equal')
			cb9 = fig1.colorbar(cd9, ax=ax9)
			cb9.set_label('RGB value prediction error (%)')

			fig1.suptitle('Int. sphere measurements from %s camera (EXP:%4.2f ms)' % (id_key, exptime))

			fig1.set_tight_layout(True)

			dirname = 'out_rad_calib'
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			fig1.savefig('out_rad_calib/out_int_%s%s_%4.2fms.png' % (id_key, suffix, exptime_rec), dpi=300)
			
			if execute_step2 is False:
				plt.show()
	
	if execute_step2:

		if not read_camp2ex_cal:
			with open('npy/%s' % (coef_npy_name), 'rb') as f:
				xx = np.load(f)
				yy = np.load(f)
				coefficient = np.load(f)
		else:
			import netCDF4 as nc
			f = nc.Dataset(radiometric_response, 'r')
			scale = 5.
			y = np.array(f['x'])
			x = np.array(f['y'])
			calib_factor_r_raw = np.flip(np.array(f['calib_factor_r']), axis=1)*scale
			calib_factor_g_raw = np.flip(np.array(f['calib_factor_g']), axis=1)*scale
			calib_factor_b_raw = np.flip(np.array(f['calib_factor_b']), axis=1)*scale
			f.close()

			xx, yy = np.meshgrid(x, y)
			coefficient = np.zeros((xx.shape[0], xx.shape[1], 3), dtype=float)
			coefficient[:, :, 0] = calib_factor_r_raw*(0.00025*0.00025)
			coefficient[:, :, 1] = calib_factor_g_raw*(0.00025*0.00025)
			coefficient[:, :, 2] = calib_factor_b_raw*(0.00025*0.00025)


		for ich in range(3):
			coeff_1ch = coefficient[:, :, ich]
			
			dist = np.sqrt((xx - alpha0)**2. + (yy - beta0)**2.)*0.09
			valid = ~np.isnan(coeff_1ch)
			ang_thres = rad*0.09
			valid[dist > ang_thres] = False
			inputs = np.stack((xx[valid].flatten(), yy[valid].flatten(), coeff_1ch[valid].flatten()))

			if fit_mode == 'oblique_6th':
				guesses = np.array([
					1640.,
					935.,
					2.6e-2,
					2.5e-8,
					-3.6e-14,
					3.5e-20,
				])
				sol = optimize.root(fit_func_oblique, guesses, args=(inputs), method='lm', tol=1e-12)
			elif fit_mode == '6th':
				guesses = np.array([
					2.6e-2,
					2.5e-8,
					-3.6e-14,
					3.5e-20,
				])
				sol = optimize.root(fit_func_6th, guesses, args=(inputs), method='lm', tol=1e-12)
			elif fit_mode == 'quad':
				guesses = np.array([
					2.6e-2,
					2.5e-8,
					# -3.6e-14,
					# 3.5e-20,
				])
				sol = optimize.root(fit_func_quad, guesses, args=(inputs), method='lm', tol=1e-12)


			print('Success?', sol.success)
			# print(sol.x)
			print('Results')
			if fit_mode == 'oblique_6th':
				print('radcx, radcy = %8.2f, %8.2f # Center of the radiometric sensitivity' % (sol.x[0], sol.x[1]))
				print('c0, c2, c4, c6 = %8.2e, %8.2e, %8.2e, %8.2e' % (sol.x[2], sol.x[3], sol.x[4], sol.x[5]))
				xy_all = np.stack((xx.flatten(), yy.flatten()))
				vpredicted = func_oblique(sol.x, xy_all).reshape(xx.shape)

			elif fit_mode == '6th':
				print('c0, c2, c4, c6 = %8.2e, %8.2e, %8.2e, %8.2e' % (sol.x[0], sol.x[1], sol.x[2], sol.x[3]))
				xy_all = np.stack((xx.flatten(), yy.flatten()))
				vpredicted = func_6th(sol.x, xy_all).reshape(xx.shape)

			elif fit_mode == 'quad':
				print('c0, c2 = %8.2e, %8.2e' % (sol.x[0], sol.x[1]))
				xy_all = np.stack((xx.flatten(), yy.flatten()))
				vpredicted = func_quad(sol.x, xy_all).reshape(xx.shape)
			
			vpredicted[~valid] = np.nan

			residual = vpredicted - coeff_1ch
			offset = np.nanmean(residual[(residual >= np.nanpercentile(residual.flatten(), 40)) & (residual <= np.nanpercentile(residual.flatten(), 60))])

			rdiff = residual/vpredicted*100.
			rdiff_mean, rdiff_std = np.nanmean(rdiff), np.nanstd(rdiff)
			
			if plot:
				fig = plt.figure(figsize=(12, 8))

				ax1 = fig.add_subplot(221)
				cb = ax1.contourf(xx, yy, coeff_1ch, vmin=np.nanmin(coeff_1ch), vmax=np.nanmax(coeff_1ch), levels=27, cmap='cividis')
				# cb = ax1.pcolormesh(xx, yy, coeff_1ch, vmin=np.nanmin(coeff_1ch), vmax=np.nanmax(coeff_1ch))
				ax1.scatter(alpha0, beta0, marker='+', color='red')
				ax1.set_aspect('equal')
				ax1.set_title('%s coefficient (inverse sensitivity)' % (['Red', 'Green', 'Blue'][ich]))
				cb1 = fig.colorbar(cb, ax=ax1)
				# cb1.set_label(r'$\rm (W/m^2/nm/sr)\cdot(ms)/ count$')
				cb1.set_label(r'$\rm (W \ m^{-2} \ nm^{-1} \ sr^{-1}\cdot s \cdot count^{-1})$')

				ax2 = fig.add_subplot(222)
				cb = ax2.contourf(xx, yy, vpredicted[:, :], vmin=np.nanmin(coeff_1ch), vmax=np.nanmax(coeff_1ch), levels=27, cmap='cividis')
				# cb = ax2.pcolormesh(xx, yy, vpredicted[:, :], vmin=np.nanmin(coeff_1ch), vmax=np.nanmax(coeff_1ch))
				ax2.scatter(alpha0, beta0, marker='+', color='red')
				ax2.set_aspect('equal')
				ax2.set_title('Predicted coefficient')
				cb2 = fig.colorbar(cb, ax=ax2)
				cb2.set_label(r'$\rm (W \ m^{-2} \ nm^{-1} \ sr^{-1}\cdot s \cdot count^{-1})$')

				ax3 = fig.add_subplot(223)
				cb = ax3.pcolormesh(xx, yy, rdiff[:, :], vmin=-np.nanmax(np.abs(rdiff[:, :])), vmax=np.nanmax(np.abs(rdiff[:, :])), cmap='seismic')
				ax3.set_aspect('equal')
				ax3.set_title('Coefficient prediction error (%)')
				ax3.scatter(alpha0, beta0, marker='+', color='red')
				cb3 = fig.colorbar(cb, ax=ax3)
				cb3.set_label(r'$\rm \% $')

				ax4 = fig.add_subplot(224)
				ax4.hist(rdiff.flatten(), 60)
				ax4.set_xlim(-12.5, 12.5)
				ax4.set_xlabel('Coefficient prediction error (%)')
				ax4.set_title('Prediction error frequency')
				txt = 'mean={0:6.2f}%, std={1:6.2f}%'.format(rdiff_mean, rdiff_std)
				ax4.text(0.02, 0.87, txt, transform=ax4.transAxes)

				plt.tight_layout()
				plt.savefig('out_rad_calib/rad_residual_%s_%s_ch%d.png' % ('camp2ex' if read_camp2ex_cal else '%s%s_%4.2fms' % (id_key, suffix, exptime_rec), fit_mode, ich), dpi=500)
		
		if plot:
			plt.show()