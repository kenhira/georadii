import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import netCDF4 as nc
import bz2

from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import calc_viewing_angles, write_surface_grid_to_nc_archive, makedir_numbered, calc_bearing, nearest_grid_width_in_deg

from cartopy.crs import Projection, Mercator
from cartopy.geodesic import Geodesic
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon, Point
import shapely.geometry as sgeom
class Orthographic_rotated(Projection):
    def __init__(self, central_longitude=0.0, central_latitude=0.0, azimuth=0.0,
                 globe=None):
        proj4_params = [('proj', 'ortho'), ('lon_0', central_longitude),
                        ('lat_0', central_latitude), ('alpha', azimuth)]
        super(Orthographic_rotated, self).__init__(proj4_params, globe=globe)

        WGS84_SEMIMAJOR_AXIS = 6378137.0

        # TODO: Let the globe return the semimajor axis always.
        a = np.float64(self.globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS)
        b = np.float64(self.globe.semiminor_axis or a)

        if b != a:
            warnings.warn('The proj4 "ortho" projection does not appear to '
                          'handle elliptical globes.')

        # To stabilise the projection of geometries, we reduce the boundary by
        # a tiny fraction at the cost of the extreme edges.

        def _ellipse_boundary(semimajor=2, semiminor=1, easting=0, northing=0, n=201):
            """
            Defines a projection boundary using an ellipse.

            This type of boundary is used by several projections.

            """

            t = np.linspace(0, 2 * np.pi, n)
            coords = np.vstack([semimajor * np.cos(t), semiminor * np.sin(t)])
            coords += ([easting], [northing])
            return coords[:, ::-1]
        coords = _ellipse_boundary(a * 0.99999, b * 0.99999, n=61)
        self._boundary = sgeom.polygon.LinearRing(coords.T)
        self._xlim = self._boundary.bounds[::2]
        self._ylim = self._boundary.bounds[1::2]
        self._threshold = np.diff(self._xlim)[0] * 0.02

    @property
    def boundary(self):
        return self._boundary

    @property
    def threshold(self):
        return self._threshold

    @property
    def x_limits(self):
        return self._xlim

    @property
    def y_limits(self):
        return self._ylim

def _nice_step(rng, nticks=6):
    # Choose 1,2,5 * 10**exp step so ticks are "nice"
    raw = float(rng) / float(nticks)
    exp = np.floor(np.log10(raw))
    frac = raw / 10**exp
    if frac <= 1.0:
        mult = 1.0
    elif frac <= 2.0:
        mult = 2.0
    elif frac <= 5.0:
        mult = 5.0
    else:
        mult = 10.0
    return mult * 10**exp

if __name__ == "__main__":
    # date       = '2024-06-05'

    # st_en_times = [
    #     ['12:22:48', '12:23:45'],
    #     ['12:24:52', '12:27:41'],
    #     ['12:28:19', '12:34:56'],
    #     ['12:36:18', '12:36:37'],
    #     ['12:36:58', '12:49:39'],
    #     ['12:50:34', '12:59:15'],
    #     ['13:03:28', '13:10:30'],
    #     ['13:11:42', '13:21:56'],
    #     ['13:23:12', '13:23:26'],
    #     ['13:23:45', '13:28:10'],
    #     ['13:28:27', '13:28:56'],
    #     ['13:32:56', '13:41:42'],
    #     ['13:42:51', '13:48:38'],
    #     ['13:50:00', '13:50:27'],
    #     ['13:50:30', '13:51:03'],
    #     ['13:52:30', '13:53:26'],
    #     ['13:55:20', '13:55:35'],
    #     ['13:57:07', '13:58:23'],
    #     ['13:59:43', '14:01:00'],
    #     ['14:02:22', '14:03:36'],
    #     ['14:04:50', '14:06:03'],
    #     ['14:07:25', '14:09:15'],
    #     ['14:10:24', '14:12:03'],
    #     ['14:13:04', '14:14:47'],
    #     ['14:15:29', '14:29:19'],
    #     ['14:30:18', '14:30:33'],
    #     ['14:34:18', '14:35:00'],
    #     ['14:35:19', '15:15:10'],
    #     ['15:19:43', '15:55:47'],
    #     ['16:00:55', '16:37:54'],
    #     ['16:40:03', '17:11:33'],
    #     ['17:11:43', '17:28:48'],
    #     ['17:29:00', '17:31:25'],
    #     ['17:31:34', '17:33:20'],
    #     ['17:33:46', '17:35:22'],
    #     ['17:35:26', '18:12:14'],
    #     ['18:12:47', '18:19:39'],
    #     ['18:20:18', '18:21:14'],
    #     ['18:21:42', '18:23:44'],
    #     ['18:24:33', '18:26:00'],
    #     ['18:27:24', '18:28:54'],
    #     ['18:29:01', '18:36:29'],
    #     ['18:36:39', '18:39:19'],
    #     ['18:39:42', '18:40:42'],
    #     ['18:40:57', '18:41:24'],
    #     ['18:41:35', '18:46:14'],
    #     ['18:46:17', '18:47:04'],
    #     ['18:47:07', '18:51:11'],
    # ]

    date       = '2024-07-29'
    st_en_times = [
        ['12:24:59', '12:33:38'],
        ['12:34:14', '12:49:01'],
        ['12:49:33', '12:50:49'],
        ['12:50:51', '12:51:07'],
        ['12:53:01', '12:53:18'],
        ['12:53:33', '13:21:20'],
        ['13:21:22', '13:23:45'],
        ['13:24:10', '13:24:55'],
        ['13:26:38', '13:27:45'],
        ['13:29:26', '13:30:37'],
        ['13:32:08', '13:33:07'],
        ['13:34:52', '13:35:51'],
        ['13:35:55', '13:36:37'],
        ['13:38:22', '13:39:10'],
        ['13:40:35', '13:41:08'],
        ['13:41:14', '13:41:50'],
        ['13:41:54', '13:42:22'],
        ['13:43:48', '13:46:17'],
        ['13:47:35', '13:50:06'],
        ['13:55:16', '14:13:21'],
        ['14:15:25', '14:15:44'],
        ['14:17:35', '14:18:26'],
        ['14:19:43', '14:20:21'],
        ['14:20:34', '14:20:49'],
        ['14:22:13', '14:22:55'],
        ['14:24:21', '14:25:13'],
        ['14:25:47', '14:46:34'],
        ['14:48:42', '14:48:57'],
        ['14:49:33', '14:52:04'],
        ['14:54:59', '14:55:47'],
        ['14:58:02', '14:58:50'],
        ['14:59:28', '15:02:22'],
        ['15:02:37', '15:06:16'],
        ['15:06:56', '15:08:16'],
        ['15:08:52', '15:10:16'],
        ['15:10:37', '15:19:22'],
        ['15:20:32', '15:21:31'],
        ['15:22:46', '15:24:48'],
        ['15:25:43', '15:26:55'],
        ['15:29:24', '15:29:39'],
        ['15:29:55', '15:38:14'],
        ['15:38:39', '15:42:57'],
        ['15:43:19', '15:45:31'],
        ['15:46:07', '15:46:30'],
        ['15:47:37', '15:48:04'],
        ['15:48:19', '15:52:55'],
        ['15:53:39', '15:55:09'],
        ['15:56:27', '15:56:50'],
        ['15:58:02', '15:58:54'],
        ['16:01:06', '16:01:19'],
        ['16:01:37', '16:17:34'],
        ['16:20:55', '16:21:34'],
        ['16:21:38', '16:25:00'],
        ['16:25:04', '16:25:20'],
        ['16:25:23', '16:34:35'],
        ['16:37:01', '16:37:35'],
        ['16:37:39', '16:41:42'],
        ['16:41:46', '16:44:11'],
        ['16:45:17', '16:47:57'],
        ['16:48:31', '16:49:24'],
        ['16:52:11', '16:52:50'],
        ['16:53:05', '17:25:12'],
        ['17:25:27', '17:28:29'],
        ['17:28:53', '17:30:13'],
        ['17:30:46', '17:32:23'],
        ['17:32:35', '17:35:02'],
        ['17:35:06', '17:37:53'],
        ['17:38:05', '17:44:35'],
        ['17:44:39', '17:47:45'],
        ['17:48:27', '17:50:21'],
        ['17:50:22', '17:50:55'],
        ['17:51:26', '17:52:42'],
        ['17:53:37', '17:54:19'],
        ['17:54:23', '17:54:41'],
        ['17:55:00', '17:58:21'],
        ['17:58:22', '17:58:40'],
        ['17:58:49', '18:00:00'],
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

    # channel control (for the spring only green channel is valid)
    if datetime.datetime.strptime(date, "%Y-%m-%d") <= datetime.datetime(2024, 6, 30): # spring
        chst, nch = 1, 1
    else: # summer
        chst, nch = 0, 3

    center_wavelengths = [ 498.5,  558.9,  626.8]

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
            # fits_list = camtool.load_fits(seg_start_time, seg_end_time, location='workstation')[::1]
            fits_list = camtool.load_fits(seg_start_time, seg_end_time, location='ARCSIX2KSS')[::1]
            # fits_list = camtool.load_fits(seg_start_time, seg_end_time, location='argus')[::1]

            nearest_fits = np.zeros_like(lon_xx, dtype=np.int32)
            imgout_agg = np.zeros((lon_xx.shape[0], lon_xx.shape[1], 3))
            imgout_agg2 = np.zeros((lon_xx.shape[0], lon_xx.shape[1], 3))
            time_agg = np.zeros_like(lon_xx)
            flgout_agg = np.zeros_like(lon_xx)
            vza_grid_agg = np.zeros_like(lon_xx)
            vaa_grid_agg = np.zeros_like(lon_xx)
            sza_grid_agg = np.zeros_like(lon_xx)
            saa_grid_agg = np.zeros_like(lon_xx)
            ncount_agg = np.zeros_like(lon_xx)
            dist = np.full_like(lon_xx, np.inf)
            latlon_aircraft = np.zeros((len(fits_list), 3))
            for ifits, fits in enumerate(fits_list):
                aircraft_status, t_act = camtool.hsk_from_fits(fits)
                if ifits == 0:
                    t_act0 = t_act
                if ifits == len(fits_list) - 1:
                    t_act1 = t_act
                print(fits, t_act)
                _, dist1 = calc_bearing(aircraft_status['lon'], aircraft_status['lat'], lon_xx, lat_yy)
                latlon_aircraft[ifits, 0] = aircraft_status['lat']
                latlon_aircraft[ifits, 1] = aircraft_status['lon']
                latlon_aircraft[ifits, 2] = aircraft_status['alt']
                nearest_fits[dist1 < dist] = ifits
                dist[dist1 < dist] = dist1[dist1 < dist]
            
            # fig_nearest_fits = plt.figure(figsize=(7, 7))
            # cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)
            # ax_nearest_fits = fig_nearest_fits.add_subplot(111, projection=cartopy_proj)
            # mesh = ax_nearest_fits.pcolormesh(lon_xx, lat_yy, nearest_fits, transform=ccrs.PlateCarree(), cmap='viridis', zorder=10)
            # ax_nearest_fits.scatter(latlon_aircraft[:, 1], latlon_aircraft[:, 0], marker='+', color='red', transform=ccrs.PlateCarree(), zorder=10)
            # ax_nearest_fits.set_title('Nearest Fits')
            # cbar = plt.colorbar(mesh, ax=ax_nearest_fits, orientation='vertical', pad=0.05, aspect=50)
            # cbar.set_label('Fits Index')
            # plt.savefig('%s/nearest_fits.png' % dir2name, dpi=300)
            # plt.close(fig_nearest_fits)

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
                time_agg[(nearest_fits == ifits) & (~np.isnan(imgout_camera[:, :, 1]))] = t_act.hour + t_act.minute/60. + t_act.second/3600. + t_act.microsecond/(3600*1e6)
                flgout_agg[nearest_fits == ifits]   = flgout_camera[nearest_fits == ifits]
                vza_grid_agg[nearest_fits == ifits] = vza_grid[nearest_fits == ifits]
                vaa_grid_agg[nearest_fits == ifits] = vaa_grid[nearest_fits == ifits]
                sza_grid_agg[nearest_fits == ifits] = rad_geom['aircraft_status']['sza']
                saa_grid_agg[nearest_fits == ifits] = rad_geom['aircraft_status']['saa']
                ncount_agg[nearest_fits == ifits]   = ncount[nearest_fits == ifits]


            # Write the gridded image to a netCDF file
            output_ncfile = '%s/gridded_img_%s_%s_%s.nc' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            write_surface_grid_to_nc_archive(output_ncfile, imgout_agg[:, :, chst:chst + nch], lon_xx, lat_yy, vza_grid_agg, vaa_grid_agg, sza_grid_agg, saa_grid_agg, \
                                                ncount_agg, date, time_agg, t_act0, t_act1, wvlc=center_wavelengths[chst:chst + nch])

            # # Compress the netCDF file using bzip2
            # with open(output_ncfile, 'rb') as f_in:
            #     with bz2.open(output_ncfile + '.bz2', 'wb') as f_out:
            #         f_out.writelines(f_in)
            # # Optionally, remove the original uncompressed netCDF file
            # import os
            # os.remove(output_ncfile)

            # Plot the gridded image
            # xmin, xmax = xcenter - 0.7, xcenter + 0.7
            # ymin, ymax = ycenter - 0.05, ycenter + 0.05
            xmin, xmax = np.nanmin(lon_xx[~np.isnan(imgout_agg[:, :, 0])]), np.nanmax(lon_xx[~np.isnan(imgout_agg[:, :, 0])])
            ymin, ymax = np.nanmin(lat_yy[~np.isnan(imgout_agg[:, :, 0])]), np.nanmax(lat_yy[~np.isnan(imgout_agg[:, :, 0])])

            alpha = 1.0
            amp = 0.4
            amp2 = 0.2
            img_trans = np.zeros((imgout_agg.shape[0], imgout_agg.shape[1], 4))
            out_array1 = np.zeros((imgout_agg.shape[0], imgout_agg.shape[1], 3))
            # if nch == 1:
            #     out_array1[:, :, 0] = amp*imgout_agg[:, :, chst]/np.nanmean(imgout_agg[:, :, chst])
            #     out_array1[:, :, 1] = amp*imgout_agg[:, :, chst]/np.nanmean(imgout_agg[:, :, chst])
            #     out_array1[:, :, 2] = amp*imgout_agg[:, :, chst]/np.nanmean(imgout_agg[:, :, chst])
            # elif nch == 3:
            #     out_array1 = amp*imgout_agg[:, :, :]/np.nanmean(imgout_agg[:, :, :])
            out_array1 = amp*imgout_agg[:, :, :]/np.nanmean(imgout_agg[:, :, :])
            img_trans[:, :, 0:3] = np.where(out_array1 > 1., 1., out_array1)
            img_trans[:, :, 3]   = alpha
            img_trans2 = np.zeros((imgout_agg2.shape[0], imgout_agg2.shape[1], 4))
            out_array2 = amp2*imgout_agg2[:, :, :]/np.nanmean(imgout_agg2[:, :, :])
            img_trans2[:, :, 0:3] = np.where(out_array2 > 1., 1., out_array2)
            img_trans2[:, :, 3]   = alpha

            # cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)
            cartopy_proj = Orthographic_rotated(central_longitude=xcenter, central_latitude=ycenter, azimuth=leg_bearing - 90.)

            # fig  = plt.figure(figsize=(8, 18))
            figsy = 8
            figsx = np.minimum(lon_xx.shape[1]/lon_xx.shape[0]*figsy, 20)
            fig  = plt.figure(figsize=(figsx, figsy))
            ax = fig.add_subplot(111, projection=cartopy_proj)

            ax.pcolormesh(lon_xx, lat_yy, img_trans, transform=ccrs.PlateCarree(), zorder=-10)
            # ax.text(0.95, 0.95, f'res_m: {res_m} m', transform=ax.transAxes, fontsize=10, zorder=100,
            #         verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
            # ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
            
            # Compute the four corners of the region in lon/lat
            # Use the edge points of lon_xx and lat_yy to define the corners
            corners_lonlat = np.array([
                [lon_xx[0, 0],     lat_yy[0, 0]],
                [lon_xx[0, -1],    lat_yy[0, -1]],
                [lon_xx[-1, -1],   lat_yy[-1, -1]],
                [lon_xx[-1, 0],    lat_yy[-1, 0]]
            ])
            # Project the corners to the map projection
            corners_proj = cartopy_proj.transform_points(ccrs.PlateCarree(),
                                                        corners_lonlat[:, 0], corners_lonlat[:, 1])
            # Get min/max in projected coordinates
            x_proj_min = np.min(corners_proj[:, 0])
            x_proj_max = np.max(corners_proj[:, 0])
            y_proj_min = np.min(corners_proj[:, 1])
            y_proj_max = np.max(corners_proj[:, 1])

            # Set the extent in projection coordinates
            ax.set_extent([x_proj_min, x_proj_max, y_proj_min, y_proj_max], crs=cartopy_proj)
            
            # Add gridlines with a fancy design: alternating colors and custom label backgrounds
            xstep = _nice_step(np.abs(xmax - xmin), nticks=10)
            ystep = _nice_step(np.abs(ymax - ymin), nticks=10)
            xgrid = np.arange(np.floor(xmin/xstep)*xstep, np.ceil(xmax/xstep)*xstep + 1e-12, xstep)
            ygrid = np.arange(np.floor(ymin/ystep)*ystep, np.ceil(ymax/ystep)*ystep + 1e-12, ystep)
            # g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            # g1.xlocator = FixedLocator(xgrid)
            # g1.ylocator = FixedLocator(ygrid)

            # Define the projected panel polygon using the four corners
            panel_poly = Polygon([
                (corners_proj[0, 0], corners_proj[0, 1]),
                (corners_proj[1, 0], corners_proj[1, 1]),
                (corners_proj[2, 0], corners_proj[2, 1]),
                (corners_proj[3, 0], corners_proj[3, 1])
            ])

            # Draw longitude gridlines
            for xval in xgrid:
                # Create a line in lon-lat space at constant longitude
                lons = np.full_like(ygrid, xval)
                lats = ygrid
                # Project to panel coordinates
                pts_proj = cartopy_proj.transform_points(ccrs.PlateCarree(), lons, lats)
                line_proj = LineString([(pt[0], pt[1]) for pt in pts_proj])
                # Intersect with panel polygon
                inter = panel_poly.intersection(line_proj)
                # inter can be MultiLineString, LineString, or empty
                if inter.is_empty:
                    continue
                if inter.geom_type == 'MultiLineString':
                    lines = list(inter.geoms)
                elif inter.geom_type == 'LineString':
                    lines = [inter]
                else:
                    continue
                for seg in lines:
                    xys = np.array(seg.coords)
                    ax.plot(xys[:, 0], xys[:, 1], color='gray', lw=0.7, zorder=20)
                    # Label at both ends
                    for pt in [xys[0], xys[-1]]:
                        # Inverse project to lon/lat for label
                        lonlat = ccrs.PlateCarree().transform_point(pt[0], pt[1], cartopy_proj)
                        # Determine which edge the point is on
                        x, y = pt[0], pt[1]
                        x0, x1 = corners_proj[:, 0].min(), corners_proj[:, 0].max()
                        y0, y1 = corners_proj[:, 1].min(), corners_proj[:, 1].max()
                        pad = 8  # points
                        # Default alignment and offset
                        ha, va = 'center', 'center'
                        dx, dy = 0, 0
                        # Check which edge: left, right, top, bottom
                        x_range = x1 - x0
                        y_range = y1 - y0
                        edge_tol_x = 0.02 * x_range
                        edge_tol_y = 0.02 * y_range
                        if np.abs(x - x0) < edge_tol_x:
                            ha = 'right'
                            dx = -pad
                        elif np.abs(x - x1) < edge_tol_x:
                            ha = 'left'
                            dx = pad
                        if np.abs(y - y1) < edge_tol_y:
                            va = 'bottom'
                            dy = pad
                        elif np.abs(y - y0) < edge_tol_y:
                            va = 'top'
                            dy = -pad
                        # Transform offset from points to display, then to data
                        trans = ax.transData + plt.matplotlib.transforms.ScaledTranslation(dx/72, dy/72, ax.figure.dpi_scale_trans)
                        ax.text(
                            x, y, LONGITUDE_FORMATTER(lonlat[0]), fontsize=8, color='black',
                            ha=ha, va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                            zorder=30, transform=trans
                        )

            # Draw latitude gridlines
            for yval in ygrid:
                # Create a line in lon-lat space at constant latitude
                lons = xgrid
                lats = np.full_like(xgrid, yval)
                pts_proj = cartopy_proj.transform_points(ccrs.PlateCarree(), lons, lats)
                line_proj = LineString([(pt[0], pt[1]) for pt in pts_proj])
                inter = panel_poly.intersection(line_proj)
                if inter.is_empty:
                    continue
                if inter.geom_type == 'MultiLineString':
                    lines = list(inter.geoms)
                elif inter.geom_type == 'LineString':
                    lines = [inter]
                else:
                    continue
                for seg in lines:
                    xys = np.array(seg.coords)
                    ax.plot(xys[:, 0], xys[:, 1], color='gray', lw=0.7, zorder=20)
                    for pt in [xys[0], xys[-1]]:
                        lonlat = ccrs.PlateCarree().transform_point(pt[0], pt[1], cartopy_proj)
                        x, y = pt[0], pt[1]
                        x0, x1 = corners_proj[:, 0].min(), corners_proj[:, 0].max()
                        y0, y1 = corners_proj[:, 1].min(), corners_proj[:, 1].max()
                        pad = 8  # points
                        ha, va = 'center', 'center'
                        dx, dy = 0, 0
                        # Check which edge: left, right, top, bottom
                        x_range = x1 - x0
                        y_range = y1 - y0
                        edge_tol_x = 0.02 * x_range
                        edge_tol_y = 0.02 * y_range
                        if np.abs(x - x0) < edge_tol_x:
                            ha = 'right'
                            dx = -pad
                        elif np.abs(x - x1) < edge_tol_x:
                            ha = 'left'
                            dx = pad
                        if np.abs(y - y1) < edge_tol_y:
                            va = 'bottom'
                            dy = pad
                        elif np.abs(y - y0) < edge_tol_y:
                            va = 'top'
                            dy = -pad
                        trans = ax.transData + plt.matplotlib.transforms.ScaledTranslation(dx/72, dy/72, ax.figure.dpi_scale_trans)
                        ax.text(
                            x, y, LATITUDE_FORMATTER(lonlat[1]), fontsize=8, color='black',
                            ha=ha, va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                            zorder=30, transform=trans
                        )

            # Add a north arrow at the top-left corner, just outside the panel
            arrow_length_px = 20  # arrow length in pixels

            # Place arrow just outside the axes (negative offset in axes fraction)
            arrow_x_frac = 0.03  # left of the panel
            arrow_y_frac = 1.06   # above the panel

            # Calculate angle in radians for the arrow direction
            theta = np.deg2rad(90. - leg_bearing)
            dx_px = arrow_length_px * np.sin(theta)
            dy_px = arrow_length_px * np.cos(theta)

            # Start point in display coordinates
            start_disp = ax.transAxes.transform((arrow_x_frac, arrow_y_frac))
            end_disp = (start_disp[0] + dx_px, start_disp[1] + dy_px)

            # Move the 'N' label farther from the arrow tip
            n_offset_px = 10  # increase this value for more distance
            n_disp = (end_disp[0] + n_offset_px * np.sin(theta), end_disp[1] + n_offset_px * np.cos(theta))

            # Convert back to axes fraction for annotation
            start_axes = ax.transAxes.inverted().transform(start_disp)
            end_axes = ax.transAxes.inverted().transform(end_disp)
            n_axes = ax.transAxes.inverted().transform(n_disp)

            # Annotate 'N' a bit farther than the tip of the arrow
            ax.annotate(
                'N',
                xy=n_axes,
                xycoords='axes fraction',
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='black',
                zorder=200
            )

            # Draw the arrow
            ax.annotate(
                '', 
                xy=end_axes, 
                xytext=start_axes, 
                xycoords='axes fraction',
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=6, headlength=8),
                zorder=100
            )
            
            fig.suptitle('Gridded camera image: ' + date + ' ' + seg_start_time + '-' + seg_end_time, 
                         fontsize=12, verticalalignment='top', horizontalalignment='center')
            # Add annotation to the panel
            ax.annotate('Resolution: %.2f m' % res_m, (0.01, -0.05), xycoords='axes fraction', fontsize=10, zorder=100,
                        verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            ax.annotate('Avg aircraft altitude: %.1f m' % np.nanmean(latlon_aircraft[:, 2]), (0.99, -0.05), xycoords='axes fraction', fontsize=10, zorder=100,
                        verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            # Save the output png
            # fn_out = '%s/%04d.png' % (dir2name, ifits)
            fn_out = '%s/%s_%s_%s.png' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            fig.savefig(fn_out, dpi=600)
            # fig.savefig(fn_out, dpi=1200)

            # fig1  = plt.figure(figsize=(18, 15))
            # ax1 = fig1.add_subplot(111, projection=cartopy_proj)
            # mesh1 = ax1.pcolormesh(lon_xx, lat_yy, img_trans2, transform=ccrs.PlateCarree(), zorder=10)
            # g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            # g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
            # g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
            # g1.top_labels = False
            # g1.right_labels = False

            # fig1.suptitle('Gridded camera image: ' + date + ' ' + seg_start_time + '-' + seg_end_time)
            # fig1.tight_layout()
            # fn_out1 = '%s/%s_%s_%s_overlap.png' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            # fig1.savefig(fn_out1, dpi=300)


            # fig2  = plt.figure(figsize=(12, 10))
            # ax1 = fig2.add_subplot(211, projection=cartopy_proj)
            # mesh1 = ax1.pcolormesh(lon_xx, lat_yy, flgout_agg, transform=ccrs.PlateCarree(), zorder=10)
            # g1 = ax1.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            # g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
            # g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
            # g1.top_labels = False
            # g1.right_labels = False
            # cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='vertical', pad=0.05, aspect=50)
            # cbar1.set_label('Flag Output')

            # ax2 = fig2.add_subplot(212, projection=cartopy_proj)
            # mesh2 = ax2.pcolormesh(lon_xx, lat_yy, vza_grid_agg, transform=ccrs.PlateCarree(), zorder=10)
            # g2 = ax2.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
            # g2.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
            # g2.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
            # g2.top_labels = False
            # g2.right_labels = False
            # cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='vertical', pad=0.05, aspect=50)
            # cbar2.set_label('Viewing Zenith Angle')

            # fig2.suptitle('Gridded camera image: ' + date + ' ' + seg_start_time + '-' + seg_end_time)
            # fn_out2 = '%s/%s_%s_%s_misc.png' % (dir2name, date.replace("-", ""), seg_start_time.replace(":", ""), seg_end_time.replace(":", ""))
            # fig2.savefig(fn_out2, dpi=300)

            current_dt += datetime.timedelta(seconds=60)
        
        # Move all the generated netCDF files into a directory and compress that directory
        import os
        import glob
        import shutil
        import tarfile

        # Create a directory to store all netCDF files for this segment
        nc_dir = os.path.join(dir2name, 'gridded_nc_%s_%s_%s' % (date.replace("-", ""), start_time.replace(":", ""), end_time.replace(":", "")))
        os.makedirs(nc_dir, exist_ok=True)

        # Move all matching netCDF files into the directory
        nc_files = glob.glob('%s/gridded_img_%s_*.nc' % (dir2name, date.replace("-", "")))
        for nc_file in nc_files:
            shutil.move(nc_file, nc_dir)

        # Compress the directory into a tar.gz archive (universal)
        archive_path = '%s/gridded_img_%s_%s_%s_all.tar.gz' % (dir2name, date.replace("-", ""), start_time.replace(":", ""), end_time.replace(":", ""))
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(nc_dir, arcname=os.path.basename(nc_dir))

        # Delete the directory after compression
        shutil.rmtree(nc_dir)

        print('All netCDF files compressed into %s' % (archive_path))
