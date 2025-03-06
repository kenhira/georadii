"""
save_fits.py

author: Ken Hirata (@kenhira)

This script reads a .fits file, processes the image data, and saves the resulting image as a .png file.

Usage:
	python3 save_fits.py <input_fits_file> <output_png_file> [scaling_factor]

Arguments:
	<input_fits_file> : str : Path to the input .fits file.
	<output_png_file> : str : Name of the output .png file.
	[scaling_factor] : float, optional : Factor to scale the image data. Default is 1.0.

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from georadii.util import read_fits

if __name__ == "__main__":
	if len(sys.argv) in [3, 4]: # Check if the correct number of arguments were passed
		print(f"Arguments received: {sys.argv[1:]}")
		filename = sys.argv[1]
		savename = sys.argv[2]
		if len(sys.argv) == 4:
			factor = float(sys.argv[3])
		else:
			factor = 1.0

		# Extract the image data from .fits file
		img, header  = read_fits(filename)
		print('header:')
		print(header)

		# Plot the image
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot(111)
		ax.imshow(factor*img/img.max())
		ax.set_title(os.path.basename(filename).split('/')[-1])
		dirname = 'out_save_fits'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		fn_out = '%s/%s' % (dirname, savename)
		plt.savefig(fn_out, dpi=300)
		plt.show()
	else:
		print("Two or three arguments need to be present.")	
