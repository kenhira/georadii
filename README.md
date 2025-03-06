# Georadii (Georeferencing and Radiometric Toolkit for Airborne Imagery)

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


