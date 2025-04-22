import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
import time

import netCDF4 as nc

from georadii.georadii import Georadii
from georadii.lvis import LVIS_arcsix

from cartopy.crs import Projection, Mercator
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom

if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded
    # date       = '2024-05-31'
    # # start_time = '15:54:00' ; end_time = '15:57:00'
    # # start_time = '16:34:00' ; end_time = '16:36:00'
    # start_time = '16:34:00' ; end_time = '16:34:10'
    # # start_time = '16:16:00' ; end_time = '16:17:00'
    # # start_time = '16:41:46' ; end_time = '16:47:26'

    date       = '2024-06-05'
    # start_time = '12:43:00'
    # end_time   = '12:44:59'
    # start_time = '13:16:00'
    # end_time   = '13:16:32'
    # start_time = '13:16:00'
    # end_time   = '13:16:40'
    start_time = '13:15:23'
    end_time   = '13:15:33'
    # start_time = '14:07:01'
    # end_time   = '14:07:30'
    # start_time = '15:40:00'
    # end_time   = '15:40:30'

    # Instantiate the camera toolkit for the specified date
    lvis_tool = LVIS_arcsix(date)

    lvis_geom, lvis_meta = lvis_tool.load_lvis_l2(start_time, end_time, location='/Users/kehi6101/Downloads/lvis/l2/')

    # lat_arr = lvis_meta['lat']
    # lon_arr = lvis_meta['lon']
    # out_arr = lvis_geom['data'][:, 0, 1]
    # cbarlabl = 'LVIS max-amp altitude (m)'
    lat_arr = lvis_meta['lat']
    lon_arr = lvis_meta['lon']
    out_arr = lvis_geom['data'][:, 0, 2] - lvis_geom['data'][:, 0, 0]
    cbarlabl = 'LVIS alt max - min (m)'
    # lat_arr = lvis_meta['lat']
    # lon_arr = lvis_meta['lon']
    # # out_arr = lvis_meta['incident']
    # # out_arr = lvis_meta['range']
    # out_arr = lvis_meta['range']*np.cos(np.deg2rad(lvis_meta['incident']))
    # cbarlabl = 'LVIS range (m)'

    xcenter, ycenter = np.mean(lvis_meta['lon']), np.mean(lvis_meta['lat'])
    xmin, xmax = np.nanmin(lvis_meta['lon']), np.nanmax(lvis_meta['lon'])
    ymin, ymax = np.nanmin(lvis_meta['lat']), np.nanmax(lvis_meta['lat'])

    # Scatter plot of LVIS data
    cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)
    fig  = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=cartopy_proj)

    sc = ax.scatter(
        # lvis_meta['lon'], lvis_meta['lat'], 
        # c=lvis_geom['data'][:, 0, 1], 
        lon_arr, lat_arr,
        c=out_arr,
        cmap='viridis', 
        s=4, 
        transform=ccrs.PlateCarree()
    )
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    g1.right_labels = False
    g1.top_labels = False
    
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    # cbar.set_label('LVIS max-amp altitude (m)')
    cbar.set_label(cbarlabl)
    
    plt.show()



    #     # Save the output png
    #     # fn_out = '%s/%04d.png' % (dir2name, ifits)
    #     fn_out = '%s/%04d_%s_%s.png' % (dir2name, ifits, date.replace("-", ""), t_act.strftime("%H%M%S"))
    #     plt.savefig(fn_out, dpi=300)


    # # Instantiate the camera toolkit for the specified date
    # camtool = Camera_arcsix(date)

    # # Load housekeeping file needed for identifying the aircraft status
    # camtool.load_aircraft(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))

    # # Retrieve all the image files for the specified time period
    # # fits_list = camtool.load_fits(start_time, end_time, location='ARCSIX2KSS')[::1]
    # fits_list = camtool.load_fits(start_time, end_time, location='argus')[::1]

    # # Make the directory to store the output pngs
    # dirname = 'out_gridding'
    # dir2name = makedir_numbered(dirname)

    # for ifits, fits_file in enumerate(fits_list):
    #     # Extract the image file and the image metadata
    #     rad_geom, t_act = camtool.rad_and_geom_from_fits(fits_file, mask_aircraft_shadow=False)

    #     # Create the Georadii object (automatic georeferencing)
    #     img_meta = {'fov'	:	65.0,}
    #     # img_meta = {}
    #     cam1 = Georadii(rad_geom, input_type='camera', input_coordinate='geometry', input_meta=img_meta)

    #     # Gridding
    #     xcenter, ycenter = rad_geom['aircraft_status']['lon'], rad_geom['aircraft_status']['lat']
    #     alt = rad_geom['aircraft_status']['alt']
    #     dist_m = alt*np.tan(np.deg2rad(img_meta.get('fov', 70.0) + 5.))
    #     dist_deg = np.rad2deg(dist_m/6371000.0)
    #     incr_deg = dist_deg/250.
    #     gridding_meta = {   'transform' : { 'type' :  'rotate',
    #                         'center' :  (xcenter, ycenter),
    #                         'inclination'   : 0.},
    #         'x'         : { 'min'    :  -dist_deg,
    #                         'max'    :   dist_deg,
    #                         'incr'   :   incr_deg},
    #         'y'         : { 'min'    :  -dist_deg,
    #                         'max'    :   dist_deg,
    #                         'incr'   :   incr_deg}}
    #     lon_xx, lat_yy, imgout_camera, ncount, flgout_camera = cam1.gridded(gridding_meta, use_c=True)

    #     vza_grid, vaa_grid = calc_viewing_angles(lon_xx, lat_yy, 
    #                             rad_geom['aircraft_status']['lon'], rad_geom['aircraft_status']['lat'],
    #                             rad_geom['aircraft_status']['alt'])

    #     # Write the gridded image to a netCDF file
    #     output_ncfile = '%s/gridded_img_%s_%s.nc' % (dir2name, date.replace("-", ""), t_act.strftime("%H%M%S"))
    #     write_surface_grid_to_nc(output_ncfile, imgout_camera, lon_xx, lat_yy, vza_grid, vaa_grid, ncount, date, t_act)

    #     # Plot the gridded image
    #     # xmin, xmax = xcenter - 0.7, xcenter + 0.7
    #     # ymin, ymax = ycenter - 0.05, ycenter + 0.05
    #     xmin, xmax = np.nanmin(lon_xx[~np.isnan(imgout_camera[:, :, 0])]), np.nanmax(lon_xx[~np.isnan(imgout_camera[:, :, 0])])
    #     ymin, ymax = np.nanmin(lat_yy[~np.isnan(imgout_camera[:, :, 0])]), np.nanmax(lat_yy[~np.isnan(imgout_camera[:, :, 0])])

    #     alpha = 1.0
    #     amp = 0.4
    #     img_trans = np.zeros((imgout_camera.shape[0], imgout_camera.shape[1], 4))
    #     out_array1 = amp*imgout_camera[:, :, :]/np.nanmean(imgout_camera[:, :, :])
    #     img_trans[:, :, 0:3] = np.where(out_array1 > 1., 1., out_array1)
    #     img_trans[:, :, 3]   = alpha

    #     # cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)
    #     # angup = 0.0
    #     angup = rad_geom['aircraft_status']['saa']
    #     print('angup:', angup)
    #     cartopy_proj = Orthographic_rotated(central_longitude=xcenter, central_latitude=ycenter, azimuth=angup)

    #     fig  = plt.figure(figsize=(7, 7))
    #     ax = fig.add_subplot(111, projection=cartopy_proj)

    #     ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
    #     # ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
    #     g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    #     g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    #     g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    #     ax.annotate('Gridded camera image: ' + datetime.datetime.strftime(t_act, '%Y-%m-%d %H:%M:%S.%f'), (0.01, 1.05), xycoords='axes fraction')
    #     ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9 - np.sin(np.radians(-angup)) * 0.09, 0.9 - np.cos(np.radians(-angup)) * 0.09), 
    #         xycoords='axes fraction', textcoords='axes fraction',
    #         fontsize=12, ha='center', va='top',
    #         arrowprops=dict(facecolor='black', arrowstyle='simple', lw=1.5))

    #     # Add a horizontal distance scale bar of 0.1 km at the bottom of the plot
    #     scale_bar_length_m = min([b for b in [5, 10, 20, 50, 100, 200, 500, 1000, 2000] if b >= 0.05*(ymax - ymin)*111120])
    #     print('scale_bar_length_m:', scale_bar_length_m)
    #     x0, y0 = xmin + 0.06 * (xmax - xmin), ymin + 0.06 * (ymax - ymin)
    #     x1, y1 = Geodesic().direct(points=(x0, y0), azimuths=angup + 90., distances=scale_bar_length_m)[0, 0:2]
    #     display_coords = ax.projection.transform_points(ccrs.PlateCarree(), np.array([x0, x1]), np.array([y0, y1]))
    #     print('display_coords:', display_coords)
    #     ax.plot(display_coords[:, 0], display_coords[:, 1], transform=ax.transData, color='black', linewidth=2)
    #     # ax.text(0.5, 0.02, 'Scale bar: 0.1 km', transform=ax.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    #     ax.text(np.mean(display_coords[:, 0]), np.mean(display_coords[:, 1]) - 50, f'{scale_bar_length_m} m', 
    #         transform=ax.transData, horizontalalignment='center', verticalalignment='bottom', fontsize=12)



    #     # Save the output png
    #     # fn_out = '%s/%04d.png' % (dir2name, ifits)
    #     fn_out = '%s/%04d_%s_%s.png' % (dir2name, ifits, date.replace("-", ""), t_act.strftime("%H%M%S"))
    #     plt.savefig(fn_out, dpi=300)
    # # plt.show()