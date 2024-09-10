import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from georadii.util import read_fits

if __name__ == "__main__":
	factor = 3.
	if len(sys.argv) == 2:
		print(f"Arguments received: {sys.argv[1]}")
		filename = sys.argv[1]

		# Extract the image data from .fits file
		img, header  = read_fits(filename)
		print('header:')
		print(header)

		# Plot the image
		fig = plt.figure(figsize=(12, 9))
		ax = fig.add_subplot(111)
		ax.imshow(factor*img/img.max())
		ax.set_title(os.path.basename(filename).split('/')[-1])
		plt.show()
	else:
		print("Not enough arguments were passed.")	
