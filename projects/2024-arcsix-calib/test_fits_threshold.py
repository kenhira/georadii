"""
test_fits_threshold.py

author: Ken Hirata (@kenhira)

This script tests thresholding methods on the radioemtric calibration image set.

Usage:
	python3 test_fits_threshold.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import cv2

from georadii.util import read_fits

from skimage.filters import sobel
from skimage.morphology import flood
from skimage import segmentation

if __name__ == "__main__":
	id_key = 'ARCSIX'
	# id_key = 'Navy'
	# exptime = 0.20
	# exptime = 0.25
	# exptime = 0.30
	exptime = 0.50
	fits_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/%s_radiometric_%4.2fms/*/*.fits' % (id_key, exptime)

	all_fits_files = sorted(glob.glob(fits_location))[:]

	# ist, ien = 0, 1
	ist, ien = 50, 51
	# ist, ien = 0, len(all_fits_files)
	for ind in range(ist, ien):
		fits_file = all_fits_files[ind]

		fimg, header = read_fits(fits_file)
		fimg /= float(2**16)
		xx, yy = np.meshgrid(np.arange(fimg.shape[1]), np.arange(fimg.shape[0]))

		fimg_main = fimg[:, :, :]
		fimg_main[yy < 50] = np.nan
		# threshold_aperture = 0.10
		# area = fimg_main[:, :, 1] > threshold_aperture
		indmax = np.unravel_index(np.nanargmax(fimg_main[:, :, 1]), fimg_main[:, :, 1].shape)
		area = flood(fimg_main[:, :, 1], indmax, tolerance=float(2**13)/float(2**16))
		# indmax = np.unravel_index(np.nanargmax(fimg_main[:, :, 1]), fimg_main[:, :, 1].shape)
		# elevation_map = sobel(fimg[:, :, 1], mask=yy > 50)
		# markers = np.zeros_like(fimg[:, :, 1])
		# markers[indmax] = 1
		# area = segmentation.watershed(elevation_map, markers)
		aperture = area & (yy > 50)
		
		
		gray_image = (fimg[:, :, 1]*255).astype('uint8')

		adaptive_thresh_gaussian = cv2.adaptiveThreshold(
			gray_image, 
			maxValue=255, 
			adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
			thresholdType=cv2.THRESH_BINARY, 
			blockSize=11,  # Size of the local region (must be odd)
			C=2  # Constant subtracted from the mean
		)

		# Apply mean adaptive thresholding
		adaptive_thresh_mean = cv2.adaptiveThreshold(
			gray_image, 
			maxValue=255, 
			adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
			thresholdType=cv2.THRESH_BINARY, 
			blockSize=11, 
			C=2
		)

		plt.figure(figsize=(10, 6))

		plt.subplot(1, 3, 1)
		plt.title("Original Image (Grayscale)")
		plt.imshow(gray_image, cmap="gray")
		plt.axis("off")

		plt.subplot(1, 3, 2)
		plt.title("Adaptive Threshold (Gaussian)")
		plt.imshow(adaptive_thresh_gaussian, cmap="gray")
		plt.axis("off")

		plt.subplot(1, 3, 3)
		plt.title("Adaptive Threshold (Mean)")
		plt.imshow(adaptive_thresh_mean, cmap="gray")
		plt.axis("off")

		plt.tight_layout()
		plt.show()
		

		# fig = plt.figure(figsize=(12, 5))
		# ax1 = fig.add_subplot(131)
		# ax1.imshow(fimg, origin='lower')
		# ax2 = fig.add_subplot(132)
		# ax2.imshow(fimg, origin='lower')
		# ax2.contourf(aperture, 1, hatches=['', '///'], origin='lower', color='white', alpha=0.)
		# ax3 = fig.add_subplot(133)
		# ax3.imshow(fimg, origin='lower')
		# ax3.contourf(aperture, 1, hatches=['', '///'], origin='lower', color='white', alpha=0.)
		# plt.show()

