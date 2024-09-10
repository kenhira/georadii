"""
gridding_aviris.py

author: Ken Hirata (@kenhira)

Instruction:
1. Run open_aviris.py and generate an H5 file of the AVIRIS image segment of your interest.
2. Modify the <fn_h5> to match the H5 file path, gridding bounds of the gridding procedure.
3. Change the gridding settings in <gridding_meta> as you desire.
4. Run this code.

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator

from georadii.georadii import Georadii
from georadii.meta_flights import Camera_arcsix
from georadii.util import read_fits 

if __name__ == "__main__":

    # Open an H5 file that contains (un-gridded) AVIRIS image
    fn_h5 = '/Users/kehi6101/Downloads/ang20240611t122853_3ch.h5'
    with h5py.File(fn_h5, 'r') as f:
        lons = f['longitude'][...]
        lats = f['latitude'][...]
        rads = f['radiance'][...]

    # Define bounds for gridding (AVIRIS image can cover large spatial domain!)
    xmin, xmax, ymin, ymax = -64.5, -63.5, 85.46, 85.54

    # Subselect the region within the bounds (if None, all the region gets gridded)
    valid_domain = (xmin < lons) & (lons < xmax) & (ymin < lats) & (lats < ymax)
    # valid_domain = None

    # Define metadata and image input and create the Georadii object
    latlon_meta = {'longeo': lons, 'latgeo': lats, 'valid_domain': valid_domain}
    img = {'data': rads, 'type': 'radiance', 'unit': 'radiance'}
    aviris1 = Georadii(img, input_type='aviris', input_coordinate='latlon', input_meta=latlon_meta, mode='manual')

    # Define a grid system for gridding
    gridding_meta = {   'transform' : { 'active' :  True,
                                        'center' :  (0.5*(ymin + ymax), 0.5*(xmin + xmax)),
                                        'inclination'   : 30.},
                        'x'         : { 'min'    :  -0.2,
                                        'max'    :   0.2,
                                        'incr'   :   0.0002},
                        'y'         : { 'min'    :  -0.2,
                                        'max'    :   0.2,
                                        'incr'   :   0.0002}}
    # Gridding
    lon_xx, lat_yy, imgout, ncount = aviris1.gridded(gridding_meta)

    # Plot the gridded image
    xcenter = 0.5*(xmin + xmax)
    ycenter = 0.5*(ymin + ymax)
    alpha = 1.0
    amp = 0.4
    img_trans = np.zeros((imgout.shape[0], imgout.shape[1], 4))
    imgout[imgout < 0.] = np.nan
    out_array = amp*imgout/np.nanmean(imgout)
    img_trans[:, :, 0:3] = np.where(out_array > 1., 1., out_array)
    img_trans[:, :, 3] = alpha

    cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)

    fig  = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=cartopy_proj)

    ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
    ax.set_extent([xmin, xmax, ymin, ymax])
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    ax.annotate('Gridded AVIRIS image', (0.01, 1.05), xycoords='axes fraction')

    # Make the directpry to save the output pngs, if non-existent
    dirname = 'out_gridding_aviris'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save the output png
    fnum = 1
    fn_out = '%s/%04d.png' % (dirname, fnum)
    while os.path.exists(fn_out):
        fnum += 1
        fn_out = '%s/%04d.png' % (dirname, fnum)
    plt.savefig(fn_out, dpi=300)
    plt.show()