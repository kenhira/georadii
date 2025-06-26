"""
open_fits.py

author: Ken Hirata (@kenhira)

This script opens the fits file and provide a quicklook. If two fits files are provided, it will show the difference between the two images.

Usage:
	python3 open_fits.py

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from georadii.util import read_fits

if __name__ == "__main__":
	if len(sys.argv) == 3:
		try:
			factor = float(sys.argv[2])
		except ValueError:
			print("Second argument must be a number (factor).")
			sys.exit(1)
		print(f"Arguments received: {sys.argv[1]}, factor: {factor}")
		filename = sys.argv[1]

		# Extract the image data from .fits file
		img, header  = read_fits(filename)
		print('img shape:', img.shape)
		print('header:')
		print(header)

		# Plot the image
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot(111)
		ax.imshow(factor*img*2**(-16)) # ARCSIX/CAMP2Ex
		ax.set_title(os.path.basename(filename).split('/')[-1])

		# Make the directory to store the output pngs
		dirname = 'out_openfits'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		fnum = 1
		dir2name = '%s/%04d' %(dirname, fnum)
		while os.path.exists(dir2name):
			fnum += 1
			dir2name = '%s/%04d' %(dirname, fnum)
		os.makedirs(dir2name)

		plt.savefig('%s/out.png' % (dir2name), dpi=300)
		plt.show()
	elif len(sys.argv) == 4:
		try:
			factor = float(sys.argv[3])
		except ValueError:
			print("Third argument must be a number (factor).")
			sys.exit(1)
		print(f"Arguments received: {sys.argv[1]} {sys.argv[2]}, factor: {factor}")
		filename  = sys.argv[1]
		filename2 = sys.argv[2]

		# Extract the image data from .fits file
		img, header  = read_fits(filename)
		print('header:')
		print(header)
		img2, header2  = read_fits(filename2)
		print('header:')
		print(header2)

		# Plot the image
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot(111)
		ax.imshow(factor*np.abs(img - img2)/np.abs(img - img2).max())
		ax.set_title(os.path.basename(filename).split('/')[-1] + " - " + os.path.basename(filename2).split('/')[-1])

		# Make the directory to store the output pngs
		dirname = 'out_openfits'
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		fnum = 1
		dir2name = '%s/%04d' %(dirname, fnum)
		while os.path.exists(dir2name):
			fnum += 1
			dir2name = '%s/%04d' %(dirname, fnum)
		os.makedirs(dir2name)

		plt.savefig('%s/out.png' % (dir2name), dpi=300)
		plt.show()
	else:
		print("Usage:")
		print("  python3 open_fits.py <fits_file> <factor>")
		print("  python3 open_fits.py <fits_file1> <fits_file2> <factor>")
