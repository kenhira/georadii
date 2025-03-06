"""
plot_spectral.py

author: Ken Hirata (@kenhira)

This script plots the spectral response functions of various instruments.

Usage:
	python3 plot_spectral.py

"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from georadii.util import get_spec_resp


if __name__ == "__main__":

	wavelength_cam, spec_resp_cam, nch_cam = get_spec_resp('../../data/spectral/cam/response_ARCSIX.txt', instrument='cam')
	wavelength_rsp, spec_resp_rsp, nch_rsp = get_spec_resp('../../data/spectral/rsp/aggregated_output.txt', instrument='rsp')
	print('spec_resp_rsp', spec_resp_rsp)
	print('wavelength_rsp', wavelength_rsp)

	cols = ['red', 'green', 'blue']
	cols2 = ['purple', 'cyan', 'gold', 'magenta', 'pink', 'brown', 'yellow', 'gray', 'black', 'orange']
	
	dirname = 'out_spec_resp'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fig = plt.figure(figsize=(6, 4))
	ax1 = fig.add_subplot(111)
	ax11 = ax1.twinx()
	for ich in range(nch_cam):
		spec_resp_cam[ich, :] /= np.trapezoid(spec_resp_cam[ich, :], x=wavelength_cam)
		ax1.plot(wavelength_cam, spec_resp_cam[ich, :], color=cols[ich], label='Cam %s' % (['R', 'G', 'B'][ich]))
		ax1.fill_between(wavelength_cam, spec_resp_cam[ich, :], color=cols[ich], alpha=0.15)
	for ich in range(4):
	# for ich in range(nch_rsp):
		spec_resp_rsp[ich, :] /= np.trapezoid(spec_resp_rsp[ich, :], x=wavelength_rsp)
		wvl_avg = np.trapezoid(spec_resp_rsp[ich, :]*wavelength_rsp, x=wavelength_rsp) / np.trapezoid(spec_resp_rsp[ich, :], x=wavelength_rsp)
		ax11.plot(wavelength_rsp, spec_resp_rsp[ich, :], color=cols2[ich], linestyle='dashed', label='RSP ch%d (%6.1f nm)' % (ich + 1, wvl_avg))
		ax11.fill_between(wavelength_rsp, spec_resp_rsp[ich, :], color=cols2[ich], alpha=0.35)
	ax1.set_xlim(350., 750.)
	ax1.set_ylim(0.0, None)
	ax1.set_xlabel(r'Wavelength $\rm (nm)$')
	ax1.set_ylabel(r'Camera response $\rm (nm^{-1})$')
	ax11.set_ylim(0.0, None)
	ax11.set_ylabel(r'RSP response $\rm (nm^{-1})$')
	ax1.set_title('Spectral response')
	
	ax1.legend(loc='upper left')
	ax11.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
	plt.tight_layout()
	plt.savefig('%s/plot_spec.png' % (dirname), dpi=300)
	plt.show()
