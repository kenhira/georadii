import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
import netCDF4 as nc
import bz2

from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import calc_viewing_angles, write_surface_grid_to_nc_archive, makedir_numbered, calc_bearing, nearest_grid_width_in_deg

if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded
    # date       = '2024-08-15'
    # start_time = '15:39:00'
    # end_time   = '15:39:30'
    # date       = '2024-06-11'
    # start_time = '13:08:48'
    # end_time   = '13:08:55'
    # date       = '2024-06-05'
    # start_time = '12:43:00'
    # end_time   = '12:44:59'
    # start_time = '13:16:00'
    # end_time   = '13:16:32'
    # start_time = '13:16:00'
    # end_time   = '13:16:40'
    # start_time = '13:15:23'
    # end_time   = '13:17:22'
    # start_time = '14:16:00'
    # end_time   = '14:16:05'
    # end_time   = '14:16:30'
    # end_time   = '14:17:05'
    # start_time = '15:40:00'
    # end_time   = '15:40:30'
    # start_time = '15:16:00'
    # end_time   = '15:16:30'
    # start_time = '16:16:01'
    # end_time   = '16:16:30'
    # date       = '2024-06-06'
    # start_time = '16:16:00' ; end_time = '16:17:00'
    # # start_time = '16:41:46' ; end_time = '16:47:26'
    
    date       = '2024-06-05'

    st_en_times = [
        ['14:20:00', '14:20:09'], # test
        # ['12:24:59', '12:27:41'],
        # ['12:28:19', '12:34:56'],
        # ['12:36:18', '12:36:37'],
        # ['12:36:58', '12:49:39'],
        # ['12:50:34', '12:59:15'],
        # ['13:03:28', '13:10:30'],
        # ['13:11:42', '13:21:56'],
        # ['13:23:12', '13:23:26'],
        # ['13:23:45', '13:28:10'],
        # ['13:28:27', '13:28:56'],
        # ['13:32:56', '13:41:42'],
        # ['13:42:51', '13:48:38'],
        # ['13:50:00', '13:50:27'],
        # ['13:50:30', '13:51:03'],
        # ['13:52:30', '13:53:26'],
        # ['13:55:20', '13:55:35'],
        # ['13:57:07', '13:58:23'],
        # ['13:59:43', '14:01:00'],
        # ['14:02:22', '14:03:36'],
        # ['14:04:50', '14:06:03'],
        # ['14:07:25', '14:09:15'],
        # ['14:10:24', '14:12:03'],
        # ['14:13:04', '14:14:47'],
        # ['14:15:29', '14:29:19'],
        # ['14:30:18', '14:30:33'],
        # ['14:34:18', '14:35:00'],
        # ['14:35:19', '15:15:10'],
        # ['15:19:43', '15:55:47'],
        # ['16:00:55', '16:37:54'],
        # ['16:40:03', '17:11:33'],
        # ['17:11:43', '17:28:48'],
        # ['17:29:00', '17:31:25'],
        # ['17:31:34', '17:33:20'],
        # ['17:33:46', '17:35:22'],
        # ['17:35:26', '18:00:00'],
    ]

    # Make the directory to store the output pngs
    dirname = 'out_gridding'
    dir2name = makedir_numbered(dirname)


    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    # camtool.load_aircraft(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))
    # camtool.load_aircraft(location='%s/Work/camera/src/georadii/projects/2024-arcsix-archive/hsk_lvis/' % (os.getenv('HOME')))
    camtool.load_aircraft(location='../../data/platform/hsk_from_lvis/')

    for st_en_time in st_en_times:
        start_time = st_en_time[0]
        end_time   = st_en_time[1]

        # Loop over 60-second segments within the time period
        start_dt = datetime.datetime.strptime(start_time, "%H:%M:%S")
        end_dt = datetime.datetime.strptime(end_time, "%H:%M:%S")
        current_dt = start_dt

        while current_dt < end_dt:
            duration = np.minimum(60, (end_dt - current_dt).total_seconds())
            seg_start_time = current_dt.strftime("%H:%M:%S")
            seg_end_time = (current_dt + datetime.timedelta(seconds=duration)).strftime("%H:%M:%S")
            print("Processing segment: %s to %s" % (seg_start_time, seg_end_time))
                    

            # Obtain the meta data of the leg
            leg_bearing, dist_2x = calc_bearing( camtool.interpolate_hsk_for_timestr(seg_start_time)['lon'], 
                                                camtool.interpolate_hsk_for_timestr(seg_start_time)['lat'],
                                                camtool.interpolate_hsk_for_timestr(seg_end_time)['lon'],
                                                camtool.interpolate_hsk_for_timestr(seg_end_time)['lat'])
            print('leg_bearing', leg_bearing)
            leg_alt = camtool.interpolate_hsk_for_timestr_avg(seg_start_time, seg_end_time)['alt']
            print('leg_alt', leg_alt)
            # leg_hed = camtool.interpolate_hsk_for_timestr_avg(seg_start_time, seg_end_time)['hed']
            # print('leg_hed', leg_hed)
            xcenter = camtool.interpolate_hsk_for_timestr_mid(seg_start_time, seg_end_time)['lon']
            ycenter = camtool.interpolate_hsk_for_timestr_mid(seg_start_time, seg_end_time)['lat']

            inclination = leg_bearing - 90.
            dist_y = leg_alt*np.tan(np.deg2rad(65. - 5.))
            dist_ydeg = np.rad2deg(dist_y/6371000.0)
            incr_deg_approx = dist_ydeg/250.
            incr_deg, res_m = nearest_grid_width_in_deg(incr_deg_approx,
                [ 0.04, 0.05, 0.08, 0.1, 0.125, 0.16, 0.2, 0.4, 0.5, 0.8, 
                    1., 1.25, 1.6, 2., 4., 5., 8., 10., 12.5, 16., 20., 40., 50., 80., 
                    100., 125., 160., 200., 400., 500., 800., 1000.,])
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
            # fits_list = camtool.load_fits(seg_start_time, seg_end_time, location='ARCSIX2KSS')[::1]
            fits_list = camtool.load_fits(seg_start_time, seg_end_time, location='argus')[::1]

            nearest_fits = np.zeros_like(lon_xx, dtype=np.int32)
            imgout_agg = np.zeros((lon_xx.shape[0], lon_xx.shape[1], 3))
            imgout_agg2 = np.zeros((lon_xx.shape[0], lon_xx.shape[1], 3))
            flgout_agg = np.zeros_like(lon_xx)
            vza_grid_agg = np.zeros_like(lon_xx)
            vaa_grid_agg = np.zeros_like(lon_xx)
            sza_grid_agg = np.zeros_like(lon_xx)
            saa_grid_agg = np.zeros_like(lon_xx)
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
                imgout_agg2[~np.isnan(imgout_camera[:, :, 1]), ich] = imgout_camera[~np.isnan(imgout_camera[:, :, 1]), 1]
                flgout_agg[nearest_fits == ifits]   = flgout_camera[nearest_fits == ifits]
                vza_grid_agg[nearest_fits == ifits] = vza_grid[nearest_fits == ifits]
                vaa_grid_agg[nearest_fits == ifits] = vaa_grid[nearest_fits == ifits]
                sza_grid_agg[nearest_fits == ifits] = rad_geom['aircraft_status']['sza']
                saa_grid_agg[nearest_fits == ifits] = rad_geom['aircraft_status']['saa']
                ncount_agg[nearest_fits == ifits]   = ncount[nearest_fits == ifits]


            # Write the gridded image to a netCDF file
            output_ncfile = '%s/gridded_img_%s_%s_%s.nc' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            write_surface_grid_to_nc_archive(output_ncfile, imgout_agg[:, :, 1:2], lon_xx, lat_yy, vza_grid_agg, vaa_grid_agg, sza_grid_agg, saa_grid_agg, \
                                                ncount_agg, date, t_act)

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
            amp2 = 0.2
            img_trans = np.zeros((imgout_agg.shape[0], imgout_agg.shape[1], 4))
            out_array1 = amp*imgout_agg[:, :, :]/np.nanmean(imgout_agg[:, :, :])
            img_trans[:, :, 0:3] = np.where(out_array1 > 1., 1., out_array1)
            img_trans[:, :, 3]   = alpha
            img_trans2 = np.zeros((imgout_agg2.shape[0], imgout_agg2.shape[1], 4))
            out_array2 = amp2*imgout_agg2[:, :, :]/np.nanmean(imgout_agg2[:, :, :])
            img_trans2[:, :, 0:3] = np.where(out_array2 > 1., 1., out_array2)
            img_trans2[:, :, 3]   = alpha

            cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)

            fig  = plt.figure(figsize=(18, 15))
            ax = fig.add_subplot(111, projection=cartopy_proj)

            ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=10)
            ax.text(0.95, 0.95, f'res_m: {res_m} m', transform=ax.transAxes, fontsize=10, zorder=100,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
            # ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
            g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
            g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
            g1.top_labels = False
            g1.right_labels = False
            ax.annotate('Gridded camera image: ' + date + ' ' + seg_start_time + '-' + seg_end_time, (0.01, 1.05), xycoords='axes fraction')

            # Save the output png
            # fn_out = '%s/%04d.png' % (dir2name, ifits)
            fn_out = '%s/%s_%s_%s.png' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            fig.savefig(fn_out, dpi=1200)

            fig1  = plt.figure(figsize=(18, 15))
            ax1 = fig1.add_subplot(111, projection=cartopy_proj)
            mesh1 = ax1.pcolormesh(lon_xx, lat_yy, img_trans2, transform=ccrs.PlateCarree(), zorder=10)
            g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
            g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
            g1.top_labels = False
            g1.right_labels = False

            fig1.suptitle('Gridded camera image: ' + date + ' ' + seg_start_time + '-' + seg_end_time)
            fn_out1 = '%s/%s_%s_%s_overlap.png' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            fig1.savefig(fn_out1, dpi=300)


            fig2  = plt.figure(figsize=(12, 10))
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

            fig2.suptitle('Gridded camera image: ' + date + ' ' + seg_start_time + '-' + seg_end_time)
            fn_out2 = '%s/%s_%s_%s_misc.png' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            fig2.savefig(fn_out2, dpi=300)

            current_dt += datetime.timedelta(seconds=60)

