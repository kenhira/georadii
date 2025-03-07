"""
calibration_geometric.py

author: Ken Hirata (@kenhira)

This script runs the geometric calibration of the ARCSIX camera from the checkerboard pattern images.

Usage:
	python3 calibration_geometric.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import optimize

# function for fitting
def function(x, c1, c2):
	return c1*x + c2*x**2.

def function_6(x, c1, c2, c3, c4, c5, c6):
	return c1*x + c2*x**2. + c3*x**3. + c4*x**4. + c5*x**5. + c6*x**6.

# function to solve for the root of the 9th order polynomial equation of distortion
def solve_disortion(theta_distorted, c1, c2, c3, c4):
	roots = np.roots([c4, 0., c3, 0., c2, 0., c1, 0., 1., -theta_distorted])
	# print('theta_distorted is', theta_distorted)
	# print('root is', roots)
	condition = (np.abs(roots.real - theta_distorted) < np.deg2rad(5.)) & (np.abs(roots.imag) < 1e-3)
	possible_ans = roots[condition]
	if len(possible_ans) == 1:
		return possible_ans[0].real
	else:
		# print('Possible answer is', possible_ans)
		return None

if __name__ == "__main__":
	# File location
	id_key = 'ARCSIX'
	img_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/geometric/%s_geometric/*.jpg' % (id_key)
	# id_key = 'Navy'
	# img_location = '/Volumes/ARCSIX4KSS/LeipzigCalibrations/geometric/%s_geometric/*.jpeg' % (id_key)

	plot = True
	# plot = False

	# plot_verbose = True
	plot_verbose = False

	# Define the dimensions of checkerboard 
	CHECKERBOARD = (8, 6)
	
	# stop the iteration when specified 
	# accuracy, epsilon, is reached or 
	# specified number of iterations are completed. 
	criteria = (cv2.TERM_CRITERIA_EPS + 
				cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
	
	# Vector for 3D points 
	object_points = [] 
	# Vector for 2D points 
	image_points = []
	# List of images
	image_list = []
	
	#  3D points real world coordinates 
	objectp3d = np.zeros((1, CHECKERBOARD[0]  
						* CHECKERBOARD[1],  
						3), np.float32) 
	objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
								0:CHECKERBOARD[1]].T.reshape(-1, 2) 

	images = glob.glob(img_location)
	for filename in images: 
		image = cv2.imread(filename)
		grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	
		# Find the chess board corners 
		# If desired number of corners are 
		# found in the image then ret = true 
		ret, corners = cv2.findChessboardCorners( 
						grayColor, CHECKERBOARD,  
						cv2.CALIB_CB_ADAPTIVE_THRESH  
						+ cv2.CALIB_CB_FAST_CHECK + 
						cv2.CALIB_CB_NORMALIZE_IMAGE) 
	
		# If desired number of corners can be detected then, 
		# refine the pixel coordinates and display 
		# them on the images of checker board 
		if ret == True:
			image_list.append(image)
			object_points.append(objectp3d) 
	
			# Refining pixel coordinates 
			# for given 2d points. 
			corners2 = cv2.cornerSubPix( 
				grayColor, corners, (11, 11), (-1, -1), criteria) 
	
			image_points.append(corners2) 
	
			# Draw and display the corners 
			image_to_plot = cv2.drawChessboardCorners(image,  
											CHECKERBOARD,  
											corners2, ret) 
	
		if plot_verbose:
			cv2.imshow('img', image_to_plot) 
			cv2.waitKey(0) 
	cv2.destroyAllWindows() 
	
  
	# Perform camera calibration by 
	# passing the value of above found out 3D points (object_points) 
	# and its corresponding pixel coordinates of the 
	# detected corners (image_points) 
	ret, matrix, distortion, r_vecs, t_vecs = \
		cv2.fisheye.calibrate( 
		object_points, image_points, grayColor.shape, None, None,
		None, None, 
		cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW)  # these flags are needed!

	# Displaying required output 
	print(" Camera matrix:") 
	print(matrix) 
	
	print("\n Distortion coefficient:") 
	print(distortion) 
	
	# print("\n Rotation Vectors:") 
	# print(r_vecs) 
	
	# print("\n Translation Vectors:") 
	# print(t_vecs)


	averaged_focal_length = 0.5*(matrix[0, 0] + matrix[1, 1])

	matrix_uniform = np.copy(matrix[:, :])
	matrix_uniform[0, 0] = averaged_focal_length
	matrix_uniform[1, 1] = averaged_focal_length

	xcenter_pred = matrix[0, 2]
	ycenter_pred = matrix[1, 2]

	# Comparison of predicted vs actual location of the checkerboard pattern
	npt = sum([len(image_points[iimg]) for iimg in range(len(image_list))])
	imgp_act, imgp_pred = np.zeros((npt, 2)), np.zeros((npt, 2))
	idx = 0
	for iimg, image in enumerate(image_list):
		image_points_predicted = \
			cv2.fisheye.projectPoints(object_points[iimg], r_vecs[iimg], t_vecs[iimg], matrix, distortion)
		# print('# %d' % (iimg + 1))
		for ip in range(len(image_points[iimg])):
			imgp_act[idx, :] = image_points[iimg][ip, 0, :]
			imgp_pred[idx, :] = image_points_predicted[0][0, ip, :]
			idx += 1
			# print(image_points[iimg][ip, 0, :], image_points_predicted[0][0, ip, :])
		
	imgp_dist = np.sqrt((imgp_act[:, 0] - imgp_pred[:, 0])**2. + (imgp_act[:, 1] - imgp_pred[:, 1])**2.) # error distance in pixel
	imgp_r = np.sqrt((imgp_act[:, 0] - xcenter_pred)**2. + (imgp_act[:, 1] - ycenter_pred)**2.) # distance from the center in pixel

	zenmax = solve_disortion(np.nanmax(imgp_r)/averaged_focal_length, 
					distortion[0, 0], distortion[1, 0], distortion[2, 0], distortion[3, 0])
	print('zenmax', np.rad2deg(zenmax))

	dirname = 'out_geom_calib'
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	if plot:
		bins = np.linspace(0., 1000., 21)
		bin_indices = np.digitize(imgp_r, bins)
		binned_means = [np.mean(imgp_dist[bin_indices == i]) if np.any(bin_indices == i) else np.nan for i in range(1, len(bins))]
		binned_p1 = [np.percentile(imgp_dist[bin_indices == i], 25) if np.any(bin_indices == i) else np.nan for i in range(1, len(bins))]
		binned_p3 = [np.percentile(imgp_dist[bin_indices == i], 75) if np.any(bin_indices == i) else np.nan for i in range(1, len(bins))]
		bin_centers = (bins[:-1] + bins[1:]) / 2.

		ang1 = np.arange(0., 2.*np.pi, 0.001)

		fig = plt.figure(figsize=(6, 7))
		ax1 = fig.add_subplot(211)
		ct1 = ax1.scatter(imgp_act[:, 0], imgp_pred[:, 1], c=imgp_dist, s=5)
		ax1.scatter(xcenter_pred, ycenter_pred, marker='+', color='red')
		ax1.plot(xcenter_pred + 1000.*np.cos(ang1), ycenter_pred + 1000.*np.sin(ang1), linestyle='dashed', color='grey')
		ax1.set_xlim(0, 3080)
		ax1.set_ylim(0, 2096)
		ax1.set_aspect('equal')
		cb1 = fig.colorbar(ct1, ax=ax1)
		cb1.set_label('Error (pixel)')
		ax2 = fig.add_subplot(212)
		ax2.scatter(imgp_r[:], imgp_dist[:], s=4, color='orange')
		ax2.plot(bin_centers, binned_means, color='black')
		ax2.errorbar(bin_centers, binned_means, yerr=[binned_p1, binned_p3], capsize=2.5, markersize=5, fmt='o', color='black')
		ax2.set_xlabel('Pixel from center')
		ax2.set_ylabel('Error (pixel)')
		ax2.set_xlim(0, 1000)
		ax2.set_ylim(0, None)
		ax2.text(0.7, -0.18, 'Errorbar denotes quartiles', transform=ax2.transAxes, fontsize=8)
		fig.savefig('out_geom_calib/error_%s.png' % id_key, dpi=300)
		# plt.show()
	
	the = np.linspace(0.0, zenmax, 10000) # undistorted (acutal) zenith angle
	thed = the*(1. + distortion[0]*the**2. + distortion[1]*the**4. + distortion[2]*the**6. + distortion[3]*the**8.)
	pix = thed*averaged_focal_length

	print('Center pixel (in ARCSIX configuration): %9.2f, %9.2f' % ((image.shape[1] - matrix[0, 2]), (image.shape[0] - matrix[1, 2])))
	popt, pcov = optimize.curve_fit(function, pix, the*180./np.pi)
	print('Quadratic fit coeff (linear, quadtratic)', popt)
	popt6, pcov6 = optimize.curve_fit(function_6, pix, the*180./np.pi)
	print('6th order poly fit coeff', popt6)

	if plot:
		## Test plot ##
		fig2 = plt.figure(figsize=(4.5, 6))
		ax21 = fig2.add_subplot(211)
		ax21.plot(pix, the*180./np.pi, color='black', label='Derived distortion')
		ax21.plot(pix, 0.09*pix, color='lightblue', label='0.09 deg/pix linear')
		ax21.plot(pix, function(pix, *popt), color='blue', label='Quadratic fit')
		ax21.plot(pix, function_6(pix, *popt6), color='red', label='6th order fit')
		text1 = r'Quad: $\theta=%6.1e p^2 + %6.1e p$' % (popt[1],  popt[0])
		text1 = text1.replace('+ -', '-').replace('e-0', r' \times 10^{-').replace('e-', r' \times 10^{-').replace('p', '} p')
		text2 = r'6th:  $\theta=%6.1e p^6 + %6.1e p^5 + %6.1e p^4 $' % (popt6[5], popt6[4], popt6[3]) \
					+ '\n' + r'       $+ %6.1e p^3 + %6.1e p^2 + %6.1e p$' % (popt6[2], popt6[1], popt6[0])
		text2 = text2.replace('+  -', '-').replace('+ -', '-').replace('e-0', r' \times 10^{-').replace('e-', r' \times 10^{-').replace('p', '} p')
		ax21.text(0.3, 0.15, text1, color='blue', transform=ax21.transAxes, fontsize=7)
		ax21.text(0.15, 0.015, text2, color='red',  transform=ax21.transAxes, fontsize=7)
		# ax21.set_xlabel('Pixel from center')
		ax21.set_ylabel('Angle (deg)')
		ax21.set_xlim(0., 1000)
		ax21.set_ylim(0., 90.)
		ax21.legend()
		ax22 = fig2.add_subplot(212)
		ax22.plot(np.arange(0., 1000.), np.arange(0., 1000.)*0., color='lightgrey')
		ax22.plot(pix, 0.09*pix - the*180./np.pi, color='lightblue', label='0.09 deg/pix linear')
		ax22.plot(pix, function(pix, *popt) - the*180./np.pi, color='blue', label='Quadratic fit')
		ax22.plot(pix, function_6(pix, *popt6) - the*180./np.pi, color='red', label='6th order fit')
		ax22.set_xlabel('Pixel from center')
		ax22.set_ylabel('Angle error (deg)')
		ax22.set_xlim(0., 1000)
		ax22.legend()
		plt.tight_layout()
		fig2.savefig('out_geom_calib/angle_fit_%s.png' % id_key, dpi=300)
		
		plt.show()