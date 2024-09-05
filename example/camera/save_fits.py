import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from georadii.util import read_fits

if __name__ == "__main__":
	factor = 25.
	if len(sys.argv) == 3:
		print(f"Arguments received: {sys.argv[1:]}")
		filename = sys.argv[1]
		savename = sys.argv[2]
		img, header  = read_fits(filename)
		print('header:')
		print(header)

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
		print("Not enough arguments were passed.")	
