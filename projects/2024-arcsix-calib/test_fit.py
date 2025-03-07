"""
test_fit.py

author: Ken Hirata (@kenhira)

This script tests various 2D fitting methods on the radiometric calibration data.

Usage:
	python3 test_fit.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# alpha0, beta0 = 1594.75, 2080. - 933.27
# alpha0, beta0 = 1594.75, 1146.73
alpha0, beta0 = 1594.75, 933.27
rad  = 850.

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

# def polyn(x, c0, c2, c4, c6):
# 	return c0 + c2*x**2. + c4*x**4. + c6*x**6.

if __name__ == "__main__":
	# fit_mode = 'quad'
	# fit_mode = '6th'
	fit_mode = 'oblique_6th'

	# ich = 0
	ich = 1
	# ich = 2

	# Plot
	plot = True
	# plot = False

	with open('npy/coefficient.npy', 'rb') as f:
		xx = np.load(f)
		yy = np.load(f)
		coefficient = np.load(f)

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
		ax1.set_title('Coefficient (inverse sensitivity)')
		cb1 = fig.colorbar(cb, ax=ax1)
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
		plt.savefig('out_rad_calib/test_residual_%s_ch%d.png' % (fit_mode, ich), dpi=500)
		plt.show()