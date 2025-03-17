import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
import time
import netCDF4 as nc
import bz2

from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import read_fits, calc_viewing_angles, write_surface_grid_to_nc, makedir_numbered, calc_bearing

if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded
    # date       = '2024-08-15'
    # start_time = '15:39:00'
    # end_time   = '15:39:30'
    # date       = '2024-06-11'
    # start_time = '13:08:48'
    # end_time   = '13:08:55'
    date       = '2024-06-05'
    # start_time = '12:43:00'
    # end_time   = '12:44:59'
    # start_time = '13:16:00'
    # end_time   = '13:16:32'
    # start_time = '13:15:23'
    # end_time   = '13:17:22'
    start_time = '14:16:01'
    end_time   = '14:16:30'
    # start_time = '16:16:01'
    # end_time   = '14:16:30'
    # date       = '2024-06-06'
    # start_time = '16:16:00' ; end_time = '16:17:00'
    # # start_time = '16:41:46' ; end_time = '16:47:26'

    # Make the directory to store the output pngs
    dirname = 'out_gridding'
    dir2name = makedir_numbered(dirname)


    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    camtool.load_hsk(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))

    # camtool.alts -= 20. # Correct the altitudes

    # Obtain the meta data of the leg
    leg_bearing, dist_2x = calc_bearing( camtool.interpolate_hsk_for_timestr(start_time)['lon'], 
                                        camtool.interpolate_hsk_for_timestr(start_time)['lat'],
                                        camtool.interpolate_hsk_for_timestr(end_time)['lon'],
                                        camtool.interpolate_hsk_for_timestr(end_time)['lat'])
    print('leg_bearing', leg_bearing)
    leg_alt = camtool.interpolate_hsk_for_timestr_avg(start_time, end_time)['alt']
    print('leg_alt', leg_alt)
    # leg_hed = camtool.interpolate_hsk_for_timestr_avg(start_time, end_time)['hed']
    # print('leg_hed', leg_hed)
    xcenter = camtool.interpolate_hsk_for_timestr_mid(start_time, end_time)['lon']
    ycenter = camtool.interpolate_hsk_for_timestr_mid(start_time, end_time)['lat']

    inclination = leg_bearing - 90.
    dist_y = leg_alt*np.tan(np.deg2rad(65. - 5.))
    dist_ydeg = np.rad2deg(dist_y/6371000.0)
    incr_deg = dist_ydeg/250.
    dist_xdeg = np.rad2deg(0.5*dist_2x/6371000.0) + 0.5*dist_ydeg
    gridding_meta = {   'transform' : { 'type' :  'rotate',
                        'center' :  (xcenter, ycenter),
                        'inclination'   : inclination},
        'x'         : { 'min'    :  -dist_xdeg,
                        'max'    :   dist_xdeg,
                        'incr'   :   incr_deg},
        'y'         : { 'min'    :  -dist_ydeg,
                        'max'    :   dist_ydeg,
                        'incr'   :   incr_deg}}
    # gridding_meta = {   'transform' : { 'type' :  'ease2',},
    #     'x'         : { 'incr'   :   100.},
    #     'y'         : { 'incr'   :   100.}}
    lon_xx, lat_yy = Georadii.grid_define(gridding_meta)

    # Retrieve all the image files for the specified time period
    fits_list = camtool.load_fits(start_time, end_time, location='ARCSIX2KSS')[::1]

    nearest_fits = np.zeros_like(lon_xx, dtype=np.int32)
    imgout_agg = np.zeros((lon_xx.shape[0], lon_xx.shape[1], 3))
    flgout_agg = np.zeros_like(lon_xx)
    vza_grid_agg = np.zeros_like(lon_xx)
    vaa_grid_agg = np.zeros_like(lon_xx)
    ncount_agg = np.zeros_like(lon_xx)
    dist = np.full_like(lon_xx, np.inf)
    latlon_aircraft = np.zeros((len(fits_list), 2))
    for ifits, fits in enumerate(fits_list):
        aircraft_status, t_act = camtool.hsk_from_fits(fits)
        print(fits, t_act)
        _, dist1 = calc_bearing(aircraft_status['lon'], aircraft_status['lat'], lon_xx, lat_yy)
        latlon_aircraft[ifits, 0] = aircraft_status['lat']
        latlon_aircraft[ifits, 1] = aircraft_status['lon']
        nearest_fits[dist1 < dist] = ifits
        dist[dist1 < dist] = dist1[dist1 < dist]
    
    fig_nearest_fits = plt.figure(figsize=(7, 7))
    cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)
    ax_nearest_fits = fig_nearest_fits.add_subplot(111, projection=cartopy_proj)
    mesh = ax_nearest_fits.pcolormesh(lon_xx, lat_yy, nearest_fits, transform=ccrs.PlateCarree(), cmap='viridis', zorder=10)
    ax_nearest_fits.scatter(latlon_aircraft[:, 1], latlon_aircraft[:, 0], marker='+', color='red', transform=ccrs.PlateCarree(), zorder=10)
    ax_nearest_fits.set_title('Nearest Fits')
    cbar = plt.colorbar(mesh, ax=ax_nearest_fits, orientation='vertical', pad=0.05, aspect=50)
    cbar.set_label('Fits Index')
    plt.savefig('%s/nearest_fits.png' % dir2name, dpi=300)
    plt.close(fig_nearest_fits)

    for ifits, fits_file in enumerate(fits_list):
        # Extract the image file and the image metadata
        rad_geom, t_act = camtool.rad_and_geom_from_fits(fits_file, mask_aircraft_shadow=True)

        # Create the Georadii object (automatic georeferencing)
        img_meta = {'fov'	:	66.0,}
        # img_meta = {}
        cam1 = Georadii(rad_geom, input_type='camera', input_coordinate='geometry', input_meta=img_meta)
        
        # Gridding
        lon_xx, lat_yy, imgout_camera, ncount, flgout_camera = cam1.gridded(gridding_meta, use_c=True)

        vza_grid, vaa_grid = calc_viewing_angles(lon_xx, lat_yy, 
                                rad_geom['aircraft_status']['lon'], rad_geom['aircraft_status']['lat'],
                                rad_geom['aircraft_status']['alt'])
        
        imgout_agg[nearest_fits == ifits]   = imgout_camera[nearest_fits == ifits]
        ich = ifits % 3
        # imgout_agg[nearest_fits == ifits, ich] = imgout_camera[nearest_fits == ifits, 1]
        # imgout_agg[~np.isnan(imgout_camera[:, :, 1]), ich] = imgout_camera[~np.isnan(imgout_camera[:, :, 1]), 1]
        flgout_agg[nearest_fits == ifits]   = flgout_camera[nearest_fits == ifits]
        vza_grid_agg[nearest_fits == ifits] = vza_grid[nearest_fits == ifits]
        vaa_grid_agg[nearest_fits == ifits] = vaa_grid[nearest_fits == ifits]
        ncount_agg[nearest_fits == ifits]   = ncount[nearest_fits == ifits]


    # Write the gridded image to a netCDF file
    output_ncfile = '%s/gridded_img_%s_%s_%s.nc' % (dir2name, date.replace("-", ""), start_time.replace(":", ""), end_time.replace(":", ""))
    write_surface_grid_to_nc(output_ncfile, imgout_agg, lon_xx, lat_yy, vza_grid_agg, vaa_grid_agg, ncount_agg, date, t_act)

    # Compress the netCDF file using bzip2
    with open(output_ncfile, 'rb') as f_in:
        with bz2.open(output_ncfile + '.bz2', 'wb') as f_out:
            f_out.writelines(f_in)

    # Plot the gridded image
    # xmin, xmax = xcenter - 0.7, xcenter + 0.7
    # ymin, ymax = ycenter - 0.05, ycenter + 0.05
    xmin, xmax = np.nanmin(lon_xx[~np.isnan(imgout_agg[:, :, 0])]), np.nanmax(lon_xx[~np.isnan(imgout_agg[:, :, 0])])
    ymin, ymax = np.nanmin(lat_yy[~np.isnan(imgout_agg[:, :, 0])]), np.nanmax(lat_yy[~np.isnan(imgout_agg[:, :, 0])])

    alpha = 1.0
    amp = 0.4
    # amp = 0.2
    img_trans = np.zeros((imgout_agg.shape[0], imgout_agg.shape[1], 4))
    out_array1 = amp*imgout_agg[:, :, :]/np.nanmean(imgout_agg[:, :, :])
    img_trans[:, :, 0:3] = np.where(out_array1 > 1., 1., out_array1)
    img_trans[:, :, 3]   = alpha

    cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)

    fig  = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=cartopy_proj)

    ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
    # ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    g1.top_labels = False
    g1.right_labels = False
    ax.annotate('Gridded camera image: ' + date + ' ' + start_time + '-' + end_time, (0.01, 1.05), xycoords='axes fraction')

    # Save the output png
    # fn_out = '%s/%04d.png' % (dir2name, ifits)
    fn_out = '%s/%s_%s_%s.png' % (dir2name, date.replace("-", ""), start_time.replace(":", ""), end_time.replace(":", ""))
    fig.savefig(fn_out, dpi=300)


    fig2  = plt.figure(figsize=(18, 15))
    ax1 = fig2.add_subplot(211, projection=cartopy_proj)
    mesh1 = ax1.pcolormesh(lon_xx, lat_yy, flgout_agg, transform=ccrs.PlateCarree(), zorder=10)
    g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    g1.top_labels = False
    g1.right_labels = False
    cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='vertical', pad=0.05, aspect=50)
    cbar1.set_label('Flag Output')

    ax2 = fig2.add_subplot(212, projection=cartopy_proj)
    mesh2 = ax2.pcolormesh(lon_xx, lat_yy, vza_grid_agg, transform=ccrs.PlateCarree(), zorder=10)
    g2 = ax2.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g2.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g2.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    g2.top_labels = False
    g2.right_labels = False
    cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='vertical', pad=0.05, aspect=50)
    cbar2.set_label('Viewing Zenith Angle')

    fig2.suptitle('Gridded camera image: ' + date + ' ' + start_time + '-' + end_time)
    fn_out2 = '%s/%s_%s_%s_misc.png' % (dir2name, date.replace("-", ""), start_time.replace(":", ""), end_time.replace(":", ""))
    fig2.savefig(fn_out2, dpi=1200)

