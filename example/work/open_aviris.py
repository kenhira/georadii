"""
open_aviris.py

author: Ken Hirata (@kenhira)

Instruction:
1. Go to `https://popo.jpl.nasa.gov/mmgis-aviris/?s=ujooa` and locate the flight segment that you are interested in.
2. Find the "Radiance Download Link" of the segment of interest.
3. In the terminal (after making sure you have enough storage), execute `curl -O <link>`. Replace <link> with the actual link.
4. Extract the downloaded tar file with `tar zxvf <file>. Replace <file> with the tar file that you just downloaded.
5. Modify <directory_name> and <fi_head> of this code to match the directory name and common heading keyword of the files.
6. Place this program in the directory level above <directory_name> and run it.

"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import spectral.io.envi as envi
import glob
from pyproj import Transformer
import h5py

def read_aviris_geo(dir_name, file_head, out_format='latlon'):
    # find the files needed for the analysis
    fb_glt = glob.glob("%s/%s_*_GLT"         % (dir_name, file_head))[0]
    fh_glt = glob.glob("%s/%s_*_GLT.hdr"     % (dir_name, file_head))[0]
    fb_igm = glob.glob("%s/%s_*_IGM"         % (dir_name, file_head))[0]
    fh_igm = glob.glob("%s/%s_*_IGM.hdr"     % (dir_name, file_head))[0]
    fb_obs = glob.glob("%s/%s_*_OBS"         % (dir_name, file_head))[0]
    fh_obs = glob.glob("%s/%s_*_OBS.hdr"     % (dir_name, file_head))[0]
    fb_img = glob.glob("%s/%s_*_RDN_ORT"     % (dir_name, file_head))[0]
    fh_img = glob.glob("%s/%s_*_RDN_ORT.hdr" % (dir_name, file_head))[0]

    # open each file
    glt = envi.open(fh_glt, fb_glt)
    igm = envi.open(fh_igm, fb_igm)
    obs = envi.open(fh_obs, fb_obs)
    img = envi.open(fh_img, fb_img)

    # Read the file content
    ftxt = open(fh_img, 'r')
    lines = ftxt.readlines()
    ftxt.close()
    nzone = lines[11].strip().split(',')[7].strip()

    # Mask out regions with no corresponding location (index == 0)
    mask = ((glt[:, :, 0] != 0) | (glt[:, :, 1] != 0)).reshape(glt.shape[0], glt.shape[1])
    
    # Pick one channel of the multi-spectral image
    img1 = img[:, :, [19, 32, 52]] # channels that are somewhat close to RGB channels
    img1[~mask] = np.nan

    # Shifting the index so it starts with zero as it should in Python
    cols = np.clip(abs(glt[:, :, 0]) - 1, 0, None).reshape(glt.shape[0], glt.shape[1])
    rows = np.clip(abs(glt[:, :, 1]) - 1, 0, None).reshape(glt.shape[0], glt.shape[1])

    # Find the coordinate for eacg image pixel by using the lookup table
    easting  = igm[:, :, :][rows, cols, 0]
    northing = igm[:, :, :][rows, cols, 1]

    if out_format.lower() == 'latlon':
        # Convert the Mercator representation to lat/lon
        transformer = Transformer.from_crs("epsg:326%s" % nzone, "epsg:4326", always_xy=True)
        east_x_flat  = easting.flatten()
        north_y_flat = northing.flatten()
        lon_flat, lat_flat = transformer.transform(east_x_flat, north_y_flat)
        latitudes = lat_flat.reshape(northing.shape)
        longitudes = lon_flat.reshape(easting.shape)
        return longitudes, latitudes, img1
    elif out_format.lower() == 'epsg':
        return easting, northing, nzone, img1

if __name__ == "__main__":

    # Specify the directory name and file header
    directory_name = "ang20240611t134629"
    fi_head        = "ang20240611t134629"

    # plot = True
    plot = False

    if plot:
        cartopy_proj = ccrs.Orthographic(central_longitude=-70., central_latitude=84.,)
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': cartopy_proj})
    
    out_coordinate = 'latlon'
    lons, lats, img1 = read_aviris_geo(directory_name, fi_head, out_format=out_coordinate)
    if plot:
        ax.pcolormesh(lons, lats, img1/np.nanmax(img1), transform=ccrs.PlateCarree())

    # out_coordinate = 'epsg' # for testing the alternative plotting method
    # easting, northing, nzone, img1 = read_aviris_geo(directory_name, fi_head, out_format=out_coordinate)
    # utm_zone_n = ccrs.UTM(zone=int(nzone), southern_hemisphere=False)
    # if plot:
    #     ax.pcolormesh(easting, northing, img1/np.nanmax(img1), transform=utm_zone_n)
    
    if plot:
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        plt.savefig('out_01.png', dpi=300)

    # Save the 3 channel output as H5 file
    with h5py.File('%s_3ch.h5' % fi_head, 'w') as f:
        # Create a dataset within the group
        dset1 = f.create_dataset('radiance',  data=img1)
        dset1.attrs['unit'] = 'microwatts per centimeter_squared per nanometer per steradian'
        dset1.attrs['dimensions'] = img1.shape
        dset2 = f.create_dataset('longitude', data=lons)
        dset2.attrs['unit'] = 'degree'
        dset2.attrs['dimensions'] = lons.shape
        dset3 = f.create_dataset('latitude',  data=lats)
        dset3.attrs['unit'] = 'degree'
        dset3.attrs['dimensions'] = lats.shape
    
