import os
import h5py
import datetime
import numpy as np
from PIL import Image

import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.ticker import FixedLocator
import matplotlib.colors as colors

from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import read_fits
from georadii.util import find_matched_keypoints, matched_points_to_latlon

def dms_to_decimal(degrees, minutes, seconds, direction):
    """
    Convert degrees, minutes, and seconds to decimal degrees.
    """
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal

def camera_vs_gogle_single(ts_off, pit_off, rol_off, hed_off):
    ### CAMERA
    # date       = '2024-05-31'
    # # start_time = '12:31:49'
    # # end_time   = '12:31:57'
    # start_time = '18:28:03'
    # end_time   = '18:28:11'
    # date       = '2024-06-05'
    # start_time = '11:04:00'
    # end_time   = '11:04:08'
    # # start_time = '18:46:58'
    # end_time   = '18:47:06'
    # date       = '2024-06-06'
    # start_time = '11:02:54'
    # end_time   = '11:03:04'
    # start_time = '18:38:05'
    # end_time   = '18:38:13'
    # date       = '2024-07-29'
    # # start_time = '11:21:08'
    # # end_time   = '11:21:16'
    # start_time = '18:44:40'
    # end_time   = '18:44:48'
    # date       = '2024-07-30'
    # # start_time = '11:07:40'
    # # end_time   = '11:07:48'
    # start_time = '18:14:40'
    # end_time   = '18:14:48'
    # date       = '2024-08-01'
    # # start_time = '11:14:50'
    # # end_time   = '11:14:58'
    # start_time = '18:38:12'
    # end_time   = '18:38:20'
    date       = '2024-08-02'
    start_time = '11:08:50'
    end_time   = '11:08:58'
    # start_time = '19:09:14'
    # end_time   = '19:09:22'
    # date       = '2024-08-15'
    # # start_time = '11:01:55'
    # # end_time   = '11:02:03'
    # start_time = '18:30:19'
    # end_time   = '18:30:27'

    # channel 
    ich = 0
    # ich = 1
    # ich = 2

    ### Google
    image_path = "%s/Desktop/pituffik_google_earth.jpg" % (os.getenv('HOME'))
    # image_path = "%s/Desktop/pituffik2_google_earth.jpg" % (os.getenv('HOME'))


    # Example DMS coordinates and pixel locations (corrected third point longitude)
    coordinates_dms = [
        {"lat": (76, 31, 45.00, 'N'), "lon": (68, 51, 15.00, 'W'), "pixel": (1604, 675)},
        {"lat": (76, 31, 45.00, 'N'), "lon": (68, 49, 45.00, 'W'), "pixel": (1604, 1300)},
        {"lat": (76, 31, 30.00, 'N'), "lon": (68, 51, 15.00, 'W'), "pixel": (2480, 675)},
        {"lat": (76, 31, 30.00, 'N'), "lon": (68, 49, 45.00, 'W'), "pixel": (2480, 1300)}
        # {"lat": (76, 31, 30.00, 'N'), "lon": (68, 56, 00.00, 'W'), "pixel": (703, 306)},
        # {"lat": (76, 31, 30.00, 'N'), "lon": (68, 51, 00.00, 'W'), "pixel": (703, 821)},
        # {"lat": (76, 30, 30.00, 'N'), "lon": (68, 56, 00.00, 'W'), "pixel": (1303, 306)},
        # {"lat": (76, 30, 30.00, 'N'), "lon": (68, 51, 00.00, 'W'), "pixel": (1303, 821)}
    ]

    # Convert DMS to decimal degrees and store them with their pixel values
    coordinates = [
        {
            "lat": dms_to_decimal(*coord["lat"]),
            "lon": dms_to_decimal(*coord["lon"]),
            "pixel": coord["pixel"]
        }
        for coord in coordinates_dms
    ]

    # Load the satellite image (corrected file path)
    image = Image.open(image_path)
    image_width, image_height = image.size
    image = np.array(image)

    # Calculate the edge coordinates (extent)
    x_pixels = [coord["pixel"][0] for coord in coordinates]
    y_pixels = [coord["pixel"][1] for coord in coordinates]
    lats = [coord["lat"] for coord in coordinates]
    lons = [coord["lon"] for coord in coordinates]

    # Compute latitude and longitude ranges
    lon_min = min(lons) + (0 - min(x_pixels)) * (max(lons) - min(lons)) / (max(x_pixels) - min(x_pixels))
    lon_max = min(lons) + (image_width - min(x_pixels)) * (max(lons) - min(lons)) / (max(x_pixels) - min(x_pixels))
    lat_min = max(lats) - (image_height - min(y_pixels)) * (max(lats) - min(lats)) / (max(y_pixels) - min(y_pixels))
    lat_max = max(lats) - (0 - min(y_pixels)) * (max(lats) - min(lats)) / (max(y_pixels) - min(y_pixels))

    extent = [lon_min, lon_max, lat_min, lat_max]

    # Calculate the average lat/lon for the orthographic projection center
    average_lat = sum(lats) / len(lats)
    average_lon = sum(lons) / len(lons)

    ### Read and extract CAMERA ###
    camtool = Camera_arcsix(date)
    camtool.load_aircraft(location='../../data/platform/hsk_from_lvis/')
    camtool.flight_meta['time offset']['time1'] = start_time
    camtool.flight_meta['time offset']['time2'] = end_time
    camtool.flight_meta['time offset']['second1'] = ts_off
    camtool.flight_meta['time offset']['second2'] = ts_off
    # fits_list = camtool.load_fits(start_time, end_time, location='ARCSIX2KSS')
    fits_list = camtool.load_fits(start_time, end_time, location='argus')

    img, header = read_fits(fits_list[0], flipud=camtool.flipud, fliplr=camtool.fliplr)
    imgobj = {'data': img,}
    t_act, aircraft_status = camtool.interpolate_hsk_for_fits(header['DATE-OBS'])
    # print('Camera timestamp:', t_act)

    img_meta = {'type'		:	'count',
                'unit'		:	'number',
                'exptime'	:   header['EXPTIME'],
                'fov'	    :	60.,
                # 'degperpix' :	 0.09,
                # 'degperpix2':	 0.0,
                'degperpix' :	 0.0868,
                'degperpix2':	 5.46e-06,}

    img_meta.update(aircraft_status)

    cam_setting = camtool.camera_misc
    cam_setting['pit_off'] = pit_off
    cam_setting['hed_off'] = hed_off
    cam_setting['rol_off'] = rol_off
    # cam_setting['centerpix']  = np.array([1594.75, 933.27])

    img_meta.update(camtool.camera_misc)

    # print('img_meta:', img_meta)

    cam1 = Georadii(imgobj, input_type='camera', input_coordinate='camera', input_meta=img_meta)

    # GRIDDING
    xmin = extent[0]
    xmax = extent[1]
    ymin = extent[2]
    ymax = extent[3]
    xcenter, ycenter = 0.5*(xmin + xmax), 0.5*(ymin + ymax)
    gridding_meta = {   'transform' : { 'type'   :  'rotate',
                                        'center' :  (xcenter, ycenter),
                                        'inclination'   : 0.},
                        'x'         : { 'min'    :  -0.02,
                                        'max'    :   0.02,
                                        'incr'   :   0.00002},
                        'y'         : { 'min'    :  -0.02,
                                        'max'    :   0.02,
                                        'incr'   :   0.00002}}
    lon_xx, lat_yy, imgout_camera, ncount = cam1.gridded(gridding_meta)

    factor = 4.0
    # factor = 2.0
    # factor = 1.5
    # factor = 1.0
    # factor = 0.5

    out_array1 = imgout_camera[:, :, 1]/np.nanmax(imgout_camera[:, :, 1])*factor
    out_array1[np.isnan(out_array1)] = 0.
    out_array3 = imgout_camera[:, :, :]/np.nanmax(imgout_camera[:, :, :])*factor
    out_array3[np.isnan(out_array3)] = 0.


    res_id = 't%06.2f_p%06.2f_r%06.2f_h%06.2f' % (ts_off, pit_off, rol_off, hed_off)

    ### Plotting

    
    img_trans = np.zeros((imgout_camera.shape[0], imgout_camera.shape[1], 4))
    img_trans[:, :, 0] = np.where(out_array1 > 1., 1., out_array1)
    img_trans[:, :, 1] = 0.
    img_trans[:, :, 2] = 0.
    img_trans[:, :, 3] = 0.55
    img_col = np.zeros((image.shape[0], image.shape[1], 4))
    img_col[:, :, 0] = 0.
    img_col[:, :, 1] = image[:, :, 1]/255.
    img_col[:, :, 2] = 0.
    img_col[:, :, 3] = 0.85
    img_trans2 = np.zeros((imgout_camera.shape[0], imgout_camera.shape[1], 4))
    img_trans2[:, :, 0] = np.where(out_array3[:, :, 0] > 1., 1., out_array3[:, :, 0])
    img_trans2[:, :, 1] = np.where(out_array3[:, :, 1] > 1., 1., out_array3[:, :, 1])
    img_trans2[:, :, 2] = np.where(out_array3[:, :, 2] > 1., 1., out_array3[:, :, 2])
    img_trans2[:, :, 3] = 1.0

    cartopy_proj = ccrs.Orthographic(central_longitude=average_lon, central_latitude=average_lat)

    fig  = plt.figure(figsize=(7, 6))
    ax01 = fig.add_subplot(111, projection=cartopy_proj)
    ax01.set_extent(extent, crs=ccrs.PlateCarree())
    ax01.imshow(img_col, origin='upper', extent=extent, transform=ccrs.PlateCarree())
    for coord in coordinates:
        ax01.plot(coord["lon"], coord["lat"], marker='+', color='red', markersize=8, transform=ccrs.PlateCarree())

    ax01.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
    # g1 = ax01.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    # g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    # g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))

    dirname = 'out_cam_google/%s_%s_%s' % (date, start_time.replace(':', ''), end_time.replace(':', ''))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig('%s/%s.png' % (dirname, res_id), dpi=300)

    fig2  = plt.figure(figsize=(7, 6))
    ax02 = fig2.add_subplot(211, projection=cartopy_proj)
    ax02.set_extent(extent, crs=ccrs.PlateCarree())
    ax02.pcolormesh(lon_xx, lat_yy, img_trans2, transform=ccrs.PlateCarree(), zorder=10)
    g1 = ax02.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))

    ax03 = fig2.add_subplot(212, projection=cartopy_proj)
    ax03.set_extent(extent, crs=ccrs.PlateCarree())
    ax03.imshow(image, origin='upper', extent=extent, transform=ccrs.PlateCarree())

    dirname = 'out_cam_google/%s_%s_%s' % (date, start_time.replace(':', ''), end_time.replace(':', ''))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig2.savefig('%s/raw_%s.png' % (dirname, res_id), dpi=300)
    
    plt.clf()

if __name__ == "__main__":

    # ts_off_list  = np.array([13.0, 14.0])
    # ts_off_list  = np.array([13.0, 13.5, 14.0])
    # ts_off_list  = np.array([12.5, 13.0, 13.5, 14.0, 14.5])
    # ts_off_list  = np.arange(10.0, 15.01, 0.5)
    # ts_off_list  = np.arange(12.5, 14.51, 0.1)
    # ts_off_list  = np.array([0.0])
    # ts_off_list  = np.arange(-0.5, 1.5, 0.1)
    # ts_off_list  = np.arange(0.0, 2.0, 0.1)
    ts_off_list  = np.arange(0.0, 2.01, 0.05)
    #ts_off_list  = np.arange(-1.0, 4.0, 0.1)
    # ts_off_list  = np.arange(-1.0, 3.01, 0.2)
    pit_off_list = np.array([ 0.0])
    rol_off_list = np.array([-0.1])
    hed_off_list = np.array([ 0.5])
    # ts_off_list  = np.arange( 0.0, 3.01, 0.5)
    # pit_off_list = np.arange(-3.0, 3.01, 0.5)
    # rol_off_list = np.arange(-3.0, 3.01, 0.5)
    # hed_off_list = np.arange(-3.0, 3.01, 0.5)
    # ts_off_list  = np.arange( 1.5, 2.51, 0.5)
    # pit_off_list = np.arange( 0.0, 2.01, 0.5)
    # rol_off_list = np.arange( 0.0, 2.01, 0.5)
    # hed_off_list = np.array([0.0])

    for ts_off in ts_off_list:
        for pit_off in pit_off_list:
            for rol_off in rol_off_list:
                for hed_off in hed_off_list:
                    camera_vs_gogle_single(ts_off, pit_off, rol_off, hed_off)

    
