"""
misc_aggregate_rsp_spectral.py

author: Ken Hirata (@kenhira)

This script reads multiple text files containing spectral respons functions, interpolates the data to a common wavelength range, and aggregates the results into a single DataFrame.
It then saves the aggregated data to a new text file.

Usage:
	python3 misc_aggregate_rsp_spectral.py

"""

import pandas as pd
import os
import numpy as np

if __name__ == "__main__":

	directory = '../../data/spectral/rsp/'
	txt_files = ['410.txt', '470.txt', '555.txt', '670.txt', '865.txt', '960.txt', '1590.txt', '1880.txt', '2250.txt']

	wvl = np.arange(300., 2500.01, 0.5)

	df = pd.DataFrame()
	df['Wavelength'] = wvl
	for i, txt_file in enumerate(txt_files):
		file_path = os.path.join(directory, txt_file)
		temp_df = pd.read_csv(file_path, sep='\s+', header=None, skipfooter=1, engine='python')
		wavelengths = temp_df.iloc[:, 0].to_numpy()
		values = temp_df.iloc[:, 1].to_numpy()
		interp_values = np.interp(wvl, wavelengths, values, left=0, right=0)
		df['band_%snm' % txt_file.split(".")[0]] = interp_values

	# Save the aggregated DataFrame to a new file with header
	output_file = os.path.join(directory, 'aggregated_output.txt')
	df.to_csv(output_file, sep='\t', index=False, header=True)

