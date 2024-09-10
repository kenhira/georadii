import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator

from georadii.georadii import Georadii
from georadii.meta_flights import Camera_arcsix
from georadii.util import read_fits 

if __name__ == "__main__":

    # Edit the time and attitude offset

    ts_off  =  0.  # s
    pit_off =  0.0 #1.5  # deg
    rol_off =  0.0 #1.5  # deg
    hed_off =  0.0  # deg rotate clockwise

    # Select the time range for which the image files are going to be loaded
    date       = '2024-08-15'
    start_time = '15:39:00'
    end_time   = '15:39:30'

    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    camtool.load_hsk(location='/Users/kehi6101/Downloads/ARCSIX_HSK/')
    camtool.flight_meta['toff_sec'] = ts_off

    # Retrieve all the image files for the specified time period
    fits_list = camtool.load_fits(start_time, end_time, location='ARCSIX2KSS')

    # Make the directory to store the output pngs
    dirname = 'out_gridding'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fnum = 1
    dir2name = '%s/%04d' %(dirname, fnum)
    while os.path.exists(dir2name):
        fnum += 1
        dir2name = '%s/%04d' %(dirname, fnum)
    os.makedirs(dir2name)

    for ifits, fits_file in enumerate(fits_list):
        # Extract the image file
        img, header = read_fits(fits_file, flipud=camtool.flipud, fliplr=camtool.fliplr)

        # Get the aircraft status at the time of image acquisition
        t_act, aircraft_status = camtool.interpolate_hsk_for_fits(header['DATE-OBS'])
        print('Camera timestamp:', t_act)

        # Define the camera image metadata needed for georeferencing
        img_meta = {'type'		:	'count',
                    'unit'		:	'number',
                    'exptime'	:   header['EXPTIME'],
                    'radcal'	:	True,
                    'radcaldic'	: 	camtool.radcal_dict_simple}
        img_meta.update(aircraft_status)

        cam_setting = camtool.camera_setting
        cam_setting['pit_off'] = pit_off
        cam_setting['hed_off'] = hed_off
        cam_setting['rol_off'] = rol_off
        img_meta.update(camtool.camera_setting)

        # Create the Georadii object (automatic georeferencing)
        cam1 = Georadii(img, input_type='camera', input_coordinate='camera', input_meta=img_meta)

        # Gridding
        xcenter, ycenter = -22.0, 84.76
        # xcenter, ycenter = -25.0, 84.6
        gridding_meta = {   'transform' : { 'active' :  True,
                            'center' :  (xcenter, ycenter),
                            'inclination'   : 0.},
            'x'         : { 'min'    :  -0.2,
                            'max'    :   0.2,
                            'incr'   :   0.0001},
            'y'         : { 'min'    :  -0.2,
                            'max'    :   0.2,
                            'incr'   :   0.0001}}
        lon_xx, lat_yy, imgout_camera, ncount = cam1.gridded()

        # Plot the gridded image
        xmin, xmax = xcenter - 0.7, xcenter + 0.7
        ymin, ymax = ycenter - 0.05, ycenter + 0.05

        alpha = 1.0
        amp = 0.4
        img_trans = np.zeros((imgout_camera.shape[0], imgout_camera.shape[1], 4))
        out_array1 = amp*imgout_camera[:, :, :]/np.nanmean(imgout_camera[:, :, :])
        img_trans[:, :, 0:3] = np.where(out_array1 > 1., 1., out_array1)
        img_trans[:, :, 3]   = alpha

        cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)

        fig  = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=cartopy_proj)

        ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
        ax.set_extent([xmin, xmax, ymin, ymax])
        g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
        ax.annotate('Gridded camera image: ' + datetime.datetime.strftime(t_act, '%Y-%m-%d %H:%M:%S.%f'), (0.01, 1.05), xycoords='axes fraction')

        # Save the output png
        fn_out = '%s/%04d.png' % (dir2name, ifits)
        plt.savefig(fn_out, dpi=300)
    # plt.show()