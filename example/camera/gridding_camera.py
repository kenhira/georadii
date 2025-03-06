"""
gridding_camera.py

author: Ken Hirata (@kenhira)

This script processes camera images, performs georeferencing, and grids the data to create a geographical map. The resulting image is saved as a .png file.

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator

from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import read_fits 

if __name__ == "__main__":

    # Select the time range for which the image files are going to be loaded
    date       = '2024-06-11'
    start_time = '13:08:48'
    end_time   = '13:08:55'

    # Housekeeping file location
    hsk_location = '../testdata/'

    # Fits file location
    fits_location = '../testdata/**/*.fits'

    # Make the directpry to save the output pngs, if non-existent
    dirname = 'out_gridding'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    camtool.load_hsk(location=hsk_location)

    # Retrieve all the image files for the specified time period
    fits_list = camtool.load_fits(start_time, end_time, location=fits_location)

    # Extract the image file and image metadata
    rad_geom, t_act = camtool.rad_and_geom_from_fits(fits_list[0], mask_aircraft_shadow=False)
    print('Camera timestamp:', t_act)

    # Create the Georadii object (automatic georeferencing)
    cam1 = Georadii(rad_geom, input_type='camera', input_coordinate='geometry', input_meta={})

    # Gridding
    lon_xx, lat_yy, imgout_camera, ncount = cam1.gridded()

    # Plot the gridded image
    xmin, xmax = np.nanmin(lon_xx), np.nanmax(lon_xx)
    ymin, ymax = np.nanmin(lat_yy), np.nanmax(lat_yy)

    alpha = 0.65
    amp = 0.4
    img_trans = np.zeros((imgout_camera.shape[0], imgout_camera.shape[1], 4))
    out_array1 = amp*imgout_camera[:, :, :]/np.nanmean(imgout_camera[:, :, :])
    img_trans[:, :, 0:3] = np.where(out_array1 > 1., 1., out_array1)
    img_trans[:, :, 3]   = alpha

    cartopy_proj = ccrs.Orthographic(central_longitude=-70., central_latitude=83.,)

    fig  = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=cartopy_proj)

    ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
    ax.set_title('Gridded camera image')
    ax.set_extent([xmin, xmax, ymin, ymax])
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))

    # Save the output png
    fnum = 1
    fn_out = '%s/%04d.png' % (dirname, fnum)
    while os.path.exists(fn_out):
        fnum += 1
        fn_out = '%s/%04d.png' % (dirname, fnum)
    plt.savefig(fn_out, dpi=300)
    plt.show()