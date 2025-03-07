# Georadii (Georeferencing and Radiometric Toolkit for Airborne Imagery)

## Overview
This package provides Python-based tools to handle airborne radiometric measurements. The development was aimed primarily for analyzing nadir-view all-sky camera data from NASA's [Arctic Radiation-Cloud-Aerosol-Surface Interaction Experiment](https://espo.nasa.gov/arcsix) (ARCSIX). The tool also supports some data processing for Research Scanning Polarimeter (RSP) and Airborne Visible / Infrared Imaging Spectrometer (AVIRIS).

The tools included in this package is intended for the following workflow:

> 1. Raw data in image space (e.g., camera) -> Georeferenced data (lat/lon, viewing angle)
> 
> 2. Georeferenced data (e.g., camera, RSP, AVIRIS, etc.)-> Gridded data

## Installation

First, clone this repository:
```bash
git clone https://github.com/kenhira/georadii.git
```

Get into the directory:
```bash
cd georadii
```

It is recommended to create a virtual environment for this project.
```bash
conda create -n georadii python=3.11
conda activate georadii
```

This project requires the following packages:
 - NumPy
 - SciPy
 - Matplotlib
 - Cartopy
 - NetCDF
 - h5py
 - astropy
 - Pandas
 - Scikit-image
 - Rasterio
 - pysolar
 - OpenCV

You can run the bash file to install all the required packages:
```bash
bash install_packages.sh
```

Now you can install the package:
```bash
python3 -m pip install -e .
```

## Usage
Please refer to the example code.


