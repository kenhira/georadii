import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator

import netCDF4 as nc


from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import read_fits, get_ssfr_flux, write_angular_grid_to_nc, plot_angular_grid_rad_and_ref
from georadii.util import calc_bhr_from_hdrf, calc_flx_from_rad

if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded

    # date       = '2024-05-31'
    # f_ssfr_v1 = '%s/Downloads/processed/ARCSIX-SSFR_P3B_20240531_RA.h5' % (os.getenv('HOME'))
    # f_hsk = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240531_v0.h5' % (os.getenv('HOME'))
    # start_time = '15:30:00' ; end_time = '15:38:59'
    # start_time = '15:45:00' ; end_time = '15:55:00'
    # start_time = '15:54:00' ; end_time = '15:57:00'
    # # start_time = '15:57:30' ; end_time = '16:04:30'
    # # start_time = '15:59:00' ; end_time = '15:59:59'
    # # start_time = '16:30:00' ; end_time = '16:40:00'
    # # start_time = '15:40:00'
    # # end_time   = '15:43:00'
    # # start_time = '15:39:00'
    # # end_time   = '15:45:00'
    # # start_time = '15:51:00'
    # # end_time   = '16:04:00'
    # # start_time = '16:30:00'
    # # end_time   = '16:42:00'
    # # end_time   = '16:30:10'
    # # end_time   = '14:16:00'

    date       = '2024-06-05'
    f_ssfr_v1 = '%s/Downloads/processed/ARCSIX-SSFR_P3B_20240605_RA.h5' % (os.getenv('HOME'))
    f_hsk = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240605_v0.h5' % (os.getenv('HOME'))
    # start_time = '12:39:00'
    # end_time   = '12:49:00'
    # start_time = '12:37:57' ; end_time = '12:49:39'
    # start_time = '13:11:42' ; end_time = '13:21:24'
    # start_time = '13:11:42' ; end_time = '13:21:24'
    # start_time = '13:42:51' ; end_time = '13:48:38'
    # start_time = '13:50:00' ; end_time = '13:50:27'
    # start_time = '13:50:30' ; end_time = '13:51:03'
    # start_time = '13:52:30' ; end_time = '13:53:26'
    # start_time = '13:55:20' ; end_time = '13:55:35'
    # start_time = '13:57:07' ; end_time = '13:58:23'
    # start_time = '13:59:43' ; end_time = '14:01:00'
    # start_time = '14:02:22' ; end_time = '14:03:36'
    # start_time = '14:04:50' ; end_time = '14:06:03'
    # start_time = '14:07:25' ; end_time = '14:09:15'
    # start_time = '14:10:24' ; end_time = '14:12:03'
    # start_time = '14:13:04' ; end_time = '14:14:47'
    # start_time = '14:15:29' ; end_time = '14:21:58'
    # start_time = '14:43:19' ; end_time = '14:58:50'
    # start_time = '15:35:13' ; end_time = '15:48:51'
    # start_time = '16:10:08' ; end_time = '16:24:06'
    # start_time = '13:48:00' ; end_time = '13:57:00'
    # start_time = '14:03:30' ; end_time = '14:13:30'
    # start_time = '14:03:30' ; end_time = '14:08:30'
    # start_time = '14:03:30' ; end_time = '14:04:30'
    # start_time = '14:36:00' ; end_time = '14:42:00'
    # start_time = '14:37:00' ; end_time = '14:47:00'
    # start_time = '13:43:00' ; end_time = '13:48:00'
    # start_time = '14:42:00' ; end_time = '14:52:00'
    start_time = '16:08:00' ; end_time = '16:18:00'


    # date       = '2024-06-06'
    # f_ssfr_v1 = '%s/Downloads/processed/ARCSIX-SSFR_P3B_20240606_RA.h5' % (os.getenv('HOME'))
    # f_hsk = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240606_v0.h5' % (os.getenv('HOME'))
    # # start_time = '16:16:00' ; end_time = '16:21:00'
    # # start_time = '16:21:00' ; end_time = '16:26:00'
    # start_time = '16:31:00' ; end_time = '16:37:00'

    # Angular gridding setting
    vzamin =  0.
    vzamax = 90.
    dvza   =  1.
    vaamin = -1.
    vaamax = 359.
    dvaa = 2.
    vza_arr = np.arange(vzamin, vzamax, dvza)
    vaa_arr = np.arange(vaamin, vaamax, dvaa)
    imgcnt = np.zeros((len(vza_arr), len(vaa_arr)))
    imgsum = np.zeros((len(vza_arr), len(vaa_arr)))

    # Get the downward irradiance
    flux_down = get_ssfr_flux(f_ssfr_v1, f_hsk, start_time, end_time, direction='zen', spec_resp_txt="../../data/spectral/cam/response_ARCSIX.txt")
    flxdn = flux_down[1]

    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    # camtool.load_hsk(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))
    # camtool.load_aircraft(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))
    camtool.load_aircraft(location='%s/Downloads/ARCSIX_MetNav/' % (os.getenv('HOME')), kind='MetNav')

    # Retrieve all the image files for the specified time period
    # fits_list = camtool.load_fits(start_time, end_time, location='ARCSIX2KSS')[::1]
    fits_list = camtool.load_fits(start_time, end_time, location='argus')[::1]

    # Make the directory to store the output pngs
    dirname = 'out_cam_hrdf'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fnum = 1
    dir2name = '%s/%04d' %(dirname, fnum)
    while os.path.exists(dir2name):
        fnum += 1
        dir2name = '%s/%04d' %(dirname, fnum)
    os.makedirs(dir2name)

    sza_list, saa_list = [], []
    alt_list = []

    import time
    for ifits, fits_file in enumerate(fits_list):
        t0 = time.time()

        # Extract the image file
        rad_geom, t_act = camtool.rad_and_geom_from_fits(fits_file, mask_fits_filename='../../data/platform/cam/pixel_mask_v20241023.fits')
        print('Camera timestamp:', t_act)
        print("Aircraft pitch:      %6.2f°" % rad_geom['aircraft_status']['pit'])
        print("Aircraft roll:       %6.2f°" % rad_geom['aircraft_status']['rol'])
        print("Aircraft heading:    %6.2f°" % rad_geom['aircraft_status']['hed'])
        print("Aircraft altitude:   %6.2f m" % rad_geom['aircraft_status']['alt'])
        print("Aircraft latitude:   %9.4f°" % rad_geom['aircraft_status']['lat'])
        print("Aircraft longitude:  %9.4f°" % rad_geom['aircraft_status']['lon'])


        cam1 = Georadii(rad_geom, input_type='camera', input_coordinate='geometry', mode='manual')

        sza_one = rad_geom['aircraft_status']['sza']
        saa_one = rad_geom['aircraft_status']['saa']
        wvlc = rad_geom['wavelength'][1]
        sza_list.append(sza_one)
        saa_list.append(saa_one)
        alt_list.append(rad_geom['aircraft_status']['alt'])

        gridding_ang_meta = {'saa': saa_one}
        gridding_ang_meta['vza'] = {'min': vzamin, 'max': vzamax, 'incr': dvza}
        gridding_ang_meta['vaa'] = {'min': vaamin, 'max': vaamax, 'incr': dvaa}
        razi, zen, imgavg_single, imgcnt_single = cam1.gridded_angular(gridmeta=gridding_ang_meta)
        radiance_avg = imgavg_single[:, :, 1] #imgsum_single/np.float64(imgcnt_single[:, :, np.newaxis] if imgcnt_single.ndim == 2 else imgcnt_single)
        imgcnt += imgcnt_single
        imgsum += np.nan_to_num(imgavg_single[:, :, 1]*imgcnt_single)
        
        hdrf = np.pi*radiance_avg/flxdn
        bhr = calc_bhr_from_hdrf(zen, razi, hdrf)
        print('bhr=', bhr)
        flxup = calc_flx_from_rad(zen, razi, radiance_avg)
        print('flxup=', flxup)
        
        fn_out = '%s/%04d.png' % (dir2name, ifits)
        plot_angular_grid_rad_and_ref(fn_out, razi, zen, radiance_avg, np.pi*radiance_avg/flxdn, sza_one, flxdn, t_act.strftime('%H:%M:%S'), alt=rad_geom['aircraft_status']['alt'],
                                        meta={'rad': {'vmin': 0.0, 'vmax': 0.5, 'cmap': 'viridis'},
                                              'ref': {'vmin': 0.6, 'vmax': 1.4, 'cmap': 'seismic'},})
        print(' >== Elapsed time for processing item no. %d: %5.2f sec' % (ifits + 1, time.time() - t0))

    radiance_avg = imgsum/np.float64(imgcnt)
    hdrf = np.pi*radiance_avg/flxdn
    bhr = calc_bhr_from_hdrf(zen, razi, hdrf)
    print('bhr=', bhr)
    flxup = calc_flx_from_rad(zen, razi, radiance_avg)
    print('flxup=', flxup)
    sza_avg = np.mean(np.array(sza_list))
    saa_avg = np.mean(np.array(saa_list))
    alt_avg = np.mean(np.array(alt_list))

    st_dt = datetime.datetime.strptime(start_time, '%H:%M:%S')
    en_dt = datetime.datetime.strptime(end_time,   '%H:%M:%S')
    st_hr = st_dt.hour + st_dt.minute/60. + st_dt.second/3600.
    en_hr = en_dt.hour + en_dt.minute/60. + en_dt.second/3600.
    nimg = len(fits_list)

    output_ncfile = '%s/hdrf_out_%s_%s_%s.nc' % (dir2name, date.replace("-", ""), start_time.replace(":", ""), end_time.replace(":", ""))

    write_angular_grid_to_nc(output_ncfile, radiance_avg, zen, razi, date, st_hr, en_hr, 
                reflectance=hdrf, azi_xx=razi + saa_avg, sza=sza_avg, saa=saa_avg, flxdn=flxdn, wvlc=wvlc, nimg=nimg, alt=alt_avg)

    fn_out = '%s/summary.png' % (dir2name)
    plot_angular_grid_rad_and_ref(fn_out, razi, zen, radiance_avg, hdrf, sza_avg, flxdn, start_time, end_time, nimg, alt=alt_avg,
                                        meta={'ref': {'plottyp': 'contourf', 'vmin': 0.6, 'vmax': 1.4, 'cmap': 'seismic'},})

