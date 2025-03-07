"""
misc_find_aperture.py

author: Ken Hirata (@kenhira)

This code is used to test the geometric center of an aperture in the all-sky camera image. 
It reads a FITS file, displays the image, and calculates the center of the aperture using two methods: 
1. The midpoint between two specified pixels.
2. A circle fitting method using a set of circular points.

Usage:
	python3 misc_find_aperture.py <path_to_fits_file>

Arguments:
	<path_to_fits_file> : str
		The path to the FITS file to be processed.

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	from georadii.util import read_fits, find_center

	factor = 0.8

	#ARCSIX
	f_radius = 1.017
	leftpix  = np.array([1004,  162])
	rightpix = np.array([2190, 1703])

	#CAMPEX
	# f_radius = 1.03
	# leftpix  = np.array([1004,  162])
	# rightpix = np.array([2190, 1703])

	cir_points = np.array([	[1230.,    3.],
							[ 947.,  172.],
							[ 620.,  712.],
							[ 760., 1484.],
							[1643., 1933.],
							[2151., 1776.],
							[2592.,  890.],
							[2427.,  380.],
							[1954.,    2.],])
	# head_point = np.array()

	if len(sys.argv) > 1:
		print(f"Arguments received: {sys.argv[1:]}")
		filename = sys.argv[1]
		img, header  = read_fits(filename, flipud=True, fliplr=True)
		print('header', header)
		#print('header')
		#for key, val in header.items():
		#	print(key, val)
		plt.imshow(factor*img/img.max())

		centerpix = 0.5*(leftpix + rightpix)
		print('Method 1 (left + right): centerpix', centerpix)
		r = np.sqrt((centerpix - leftpix)[0]**2. + (centerpix - leftpix)[1]**2.)*f_radius
		cent_pred, r_pred = find_center(cir_points[:, 0], cir_points[:, 1])
		cent_pred = np.round(cent_pred, 1)
		print('Method 2 (circle fit):   centerpix', cent_pred)

		ang = np.linspace(0., 2*np.pi, 1000)

		plt.scatter(centerpix[0], centerpix[1], marker='o', color='red')
		plt.scatter(cent_pred[0], cent_pred[1], marker='o', color='violet')
		plt.plot(centerpix[0] + r*np.cos(ang), centerpix[1] + r*np.sin(ang), color='red', linestyle='dashed')
		plt.plot(cent_pred[0] + r_pred*np.cos(ang), cent_pred[1] + r_pred*np.sin(ang), color='violet', linestyle='dashed')
		plt.scatter(cir_points[:, 0], cir_points[:, 1], marker='o', s=10, color='blue', zorder=10)

		plt.show()
	else:
		print("No arguments were passed.")	
