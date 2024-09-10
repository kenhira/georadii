# georadii/example/aviris

This directory contains example codes for using the Georadii package for AVIRIS data.

## Contents
 - open_aviris.py
 - gridding_aviris.py

## Usage
To run `gridding_aviris.py`, you need the H5 file containing 3 channel AVIRIS data, which can be generated using `open_aviris.py`.

### Running `open_aviris.py`
Running this program requires an additional module called [Spectral Python](https://www.spectralpython.net/index.html) `spectral`. You can install this package using `pip`.
```bash
python3 -m pip install spectral
```
Before starting, make sure that you have sufficient storage in your computer. The raw AVIRIS can be as big as a few TBs. 
 1. Go to [AVIRIS-3 data portal](https://popo.jpl.nasa.gov/mmgis-aviris/?s=ujooa) and locate the flight segment that you are interested in.
 2. Find the "Radiance Download Link" of the segment of interest.
 3. In the terminal (after making sure you have enough storage), execute `curl -O <link>`. Replace <link> with the actual link.
 4. Extract the downloaded tar file with `tar zxvf <file>. Replace <file> with the tar file that you just downloaded.
 5. Modify <directory_name> and <fi_head> of this code to match the directory name and common heading keyword of the files.
 6. Place this program in the directory level above <directory_name> and run it.

Details of the AVIRIS raw data format can be found in [their distribution document](https://aviris.jpl.nasa.gov/dataportal/20170911_AV_Download.readme).

### Running `gridding_aviris.py`
 1. Place the H5 file of the AVIRIS image segment of your interest generated with `open_aviris.py`.
 2. Modify the <fn_h5> to match the H5 file path, gridding bounds of the gridding procedure.
 3. Change the gridding settings in <gridding_meta> as you desire.
 4. Run the program.
