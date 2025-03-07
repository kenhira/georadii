import os
import sys
import numpy as np
import h5py
from astropy.io import fits
import cv2
import datetime
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator

# Read housekeeping file (HDF5 binary data format)
def read_hsk_camp2ex(hsk_filename):
    print('Reading {}'.format(hsk_filename))
    h5f = h5py.File(hsk_filename, 'r')
    hsk_data = {'doy': h5f['day_of_year'][...] ,
                'hrs': h5f['tmhr'][...]        ,
                'alt': h5f['gps_altitude'][...],
                'lat': h5f['latitude'][...]    ,
                'lon': h5f['longitude'][...]   ,
                'spd': h5f['ground_speed'][...],
                'hed': h5f['true_heading'][...],
                'rol': h5f['roll_angle'][...]  ,
                'pit': h5f['pitch_angle'][...]  }
    return hsk_data

def jd_to_doy(jd, year):
    jan1_jd = 1721425.5 + 365 * (year - 1) + int((year - 1) / 4)
    return jd - jan1_jd + 1

def read_hsk_arcsix(hsk_filename):
    print('Reading {}'.format(hsk_filename))
    h5f = h5py.File(hsk_filename, 'r')
    hsk_data = {'doy': jd_to_doy(h5f['jday'][...], 2024) ,
                'hrs': h5f['tmhr'][...]        ,
                'alt': h5f['alt'][...],
                'lat': h5f['lat'][...]    ,
                'lon': h5f['lon'][...]   ,
                'hed': h5f['ang_hed'][...],
                'rol': h5f['ang_rol'][...]  ,
                'pit': h5f['ang_pit'][...]  }
    return hsk_data

# ?????
def reshapeFITS(img):
    shp=img.shape
    if len(shp) == 3:
        newshape = (shp[1],shp[2],shp[0])
        newimg = np.zeros(newshape)
        newimg[:,:,0] = img[0,:,:]
        newimg[:,:,1] = img[1,:,:]
        newimg[:,:,2] = img[2,:,:]
    else: # monochromatic image
        newshape = (shp[0],shp[1],3)
        newimg = np.zeros(newshape)
        newimg[:,:,0] = img[:,:]
        newimg[:,:,1] = img[:,:]
        newimg[:,:,2] = img[:,:]
    return newimg

# Read image file (Fits file format)
def read_fits(fits_filename, flipud=False, fliplr=True, mask_fits_filename=None, header_only=False):
	if not os.path.exists(fits_filename):
		print('Error: {} not found.'.format(fits_filename))
		sys.exit()
	print('Reading {}'.format(fits_filename))
	handle = fits.open(fits_filename)
	if not header_only:
		fimg = reshapeFITS(handle[0].data)
		if flipud:
			fimg = np.flipud(fimg)
		if fliplr:
			fimg = np.fliplr(fimg)
		
		if mask_fits_filename is not None:
			print('Reading {}'.format(mask_fits_filename))
			handle_msk = fits.open(mask_fits_filename)
			fmsk = reshapeFITS(handle_msk[0].data)
			if flipud:
				fmsk = np.flipud(fmsk)
			if fliplr:
				fmsk = np.fliplr(fmsk)
			fheader_msk = handle_msk[0].header

			fimg[fmsk > 0.] = np.nan
	else:
		fimg = None
	
	fheader = handle[0].header

	return fimg, fheader

def find_center(xs, ys):
	A = np.vstack((xs, ys, np.ones((len(xs))))).T
	v = -(xs ** 2 + ys ** 2)
	u, residuals, rank, s = np.linalg.lstsq(A, v, rcond=None)
	cx_pred = u[0] / (-2.)
	cy_pred = u[1] / (-2.)
	r_pred = np.sqrt(cx_pred ** 2 + cy_pred ** 2 - u[2])
	return (cx_pred, cy_pred), r_pred

def find_matched_keypoints(image1, image2, dmax=50):
    import cv2
    image1[np.isnan(image1)] = 0.0
    image2[np.isnan(image2)] = 0.0
    # Step 1: Load the images
    # image1 = cv2.imread(image1_path)
    # image2 = cv2.imread(image2_path)
    image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') 
    image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Step 2: Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Step 3: Find keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Step 4: Match descriptors using FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print('good_matches', len(good_matches))

    # Step 5: Extract the locations of the matched points
    points_image1 = []
    points_image2 = []
    
    for match in good_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        # Get the coordinates of the keypoints
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt
        
        points_image1.append((x1, y1))
        points_image2.append((x2, y2))

    # return points_image1, points_image2
    m_1 = np.array([[x, y] for x, y in points_image1])
    m_2 = np.array([[x, y] for x, y in points_image2])

    # Filter out macthes that are too far away
    d_grid = np.sqrt((m_2[:, 0] - m_1[:, 0])**2 + (m_2[:, 1] - m_1[:, 1])**2)
    m1 = m_1[np.where(d_grid < dmax), 0:2][0, :, :]
    m2 = m_2[np.where(d_grid < dmax), 0:2][0, :, :]
    dgrid = d_grid[np.where(d_grid < dmax)]

    # Filter out outliers by comparing to surrounding matches
    from scipy.spatial import cKDTree
    thres_factor = 95
    radius = 200.
    tree = cKDTree(m1)
    neighborhood_idxs = tree.query_ball_point(m1, radius)
    ini_filtered = []
    for ini, nei_idxs in enumerate(neighborhood_idxs):
        if len(nei_idxs) > 1:
            dthres = np.percentile(d_grid[nei_idxs], thres_factor)
            if dgrid[ini] < dthres:
                ini_filtered.append(ini)
    ini_filtered = np.array(ini_filtered)
    print("Filtered:", ini_filtered.size)
    m1 = m1[ini_filtered, :]
    m2 = m2[ini_filtered, :]
    dgrid = dgrid[ini_filtered]

    return m1, m2, dgrid

def matched_points_to_latlon(m1, m2, lon_xx, lat_yy):
    ixm1 = np.int_(m1[:, 0])
    iym1 = np.int_(m1[:, 1])
    rxm1 = m1[:, 0] - ixm1
    rym1 = m1[:, 1] - iym1
    lonm1 = (1. - rxm1)*lon_xx[iym1, ixm1] + rxm1*lon_xx[iym1, ixm1 + 1]
    latm1 = (1. - rym1)*lat_yy[iym1, ixm1] + rym1*lat_yy[iym1 + 1, ixm1]
    ixm2 = np.int_(m2[:, 0])
    iym2 = np.int_(m2[:, 1])
    rxm2 = m2[:, 0] - ixm2
    rym2 = m2[:, 1] - iym2
    lonm2 = (1. - rxm2)*lon_xx[iym2, ixm2] + rxm2*lon_xx[iym2, ixm2 + 1]
    latm2 = (1. - rym2)*lat_yy[iym2, ixm2] + rym2*lat_yy[iym2 + 1, ixm2]
    return lonm1, latm1, lonm2, latm2

def calc_viewing_angles(lon, lat, lon_air, lat_air, alt_air):
    from pyproj import Geod as geod
    g = geod(ellps='WGS84')
    lon_air_tmp = lon_air*np.ones_like(lon)
    lat_air_tmp = lat_air*np.ones_like(lat)
    fwd_az1, fwd_az2, dist = g.inv(lon_air_tmp, lat_air_tmp, lon, lat)
    theta = np.arctan(dist/alt_air)*180./np.pi
    phi   = fwd_az1
    phi[phi < 0.] += 360.
    return theta, phi # vza, vaa (deg)

def get_spec_resp(spec_resp_txt, instrument='cam'):
    if instrument == 'cam':
        nch = 3
    elif instrument == 'rsp':
        nch = 9
    wavelengths = []
    response = [[] for _ in range(nch)]
    with open(spec_resp_txt, "r") as file:
        lines = file.readlines()
    for line in lines[1:]: # Skip the header
        values = line.strip().split()
        if len(values) == nch + 1:
            wavelengths.append(float(values[0]))
            for i in range(nch):
                response[i].append(float(values[i + 1]))
    wavelengths = np.array(wavelengths)
    spec_resp = np.zeros((nch, len(wavelengths)))
    for i in range(nch):
        spec_resp[i, :] = np.array(response[i])
        spec_resp[i, :] /= np.trapz(spec_resp[i, :], x=wavelengths)
        # print('Center wavelength: %9.2f nm' % np.trapz(wavelengths*spec_resp[i, :], x=wavelengths))
    return wavelengths, spec_resp, nch

def get_ssfr_fluxdn(f_RA, fhsk, start_time, end_time, flux_source='ssfr', instrument='cam', spec_resp_txt="./response_ARCSIX.txt"):
    ### Open SSFR file ###
    with h5py.File(f_RA, 'r') as f:
        tmhr_all = f['tmhr'][...]
        if flux_source == 'ssfr':
            rad_all = f['zen/flux'][...]
            wvl = f['zen/wvl'][...] # nm
            tmhr_ssfr = f['tmhr'][...]
        else:
            msg = 'Unknown flux source: %s' % flux_source
            raise ValueError(msg)
    
    hsk_data = read_hsk_arcsix(fhsk)
    tmhr_hsk = hsk_data['hrs']
    pit = np.interp(tmhr_ssfr, tmhr_hsk, hsk_data['pit'])
    rol = np.interp(tmhr_ssfr, tmhr_hsk, hsk_data['rol'])

    level = np.sqrt(pit**2. + rol**2.) < 2.5 # check if the aircraft is not tilted too much
    
    st_dt = datetime.datetime.strptime(start_time, '%H:%M:%S')
    en_dt = datetime.datetime.strptime(end_time,   '%H:%M:%S')

    st_hr = st_dt.hour + st_dt.minute/60. + st_dt.second/3600.
    en_hr = en_dt.hour + en_dt.minute/60. + en_dt.second/3600.

    target = level & (st_hr <= tmhr_all) & (tmhr_all < en_hr) # time and attitude filter
    tmhr = tmhr_all[target]
    rad  = rad_all[target]

    ### Open camera spectral response ###
    wavelengths, spec_resp, nch = get_spec_resp(spec_resp_txt, instrument=instrument)

    # Match the SSFR data to the camera spectral response function wavelength increments
    flux_ssfr = np.zeros((len(tmhr), len(wavelengths)))
    for it in range(len(tmhr)):
        flux_ssfr[it, :] = np.interp(wavelengths, wvl, rad[it, :])
    
    # Calculate the flux down corresponding to each camera channel
    f_resp = np.zeros((nch, len(tmhr)))
    for ich in range(nch):
        f_resp[ich, :] = np.trapezoid(spec_resp[ich, :][np.newaxis, :]*flux_ssfr, x=wavelengths, axis=1)

    # Average temporally
    flux_down = np.nanmean(f_resp[:, :], axis=1)
    for ich in range(nch):
        print('Flux down (ch=%d): %9.4f (W/m^2/nm)' % (ich, flux_down[ich]))

    return flux_down

def write_surface_grid_to_nc(output_ncfile, datout, lonxx, latyy, vzagrid, vaagrid, ncount, date, tact, reflectance=None, flxdn=None, wvlc=None, sza=None, saa=None):
    print('Writing to %s' % (output_ncfile))
    with nc.Dataset(output_ncfile, "w", format="NETCDF4") as ncfile:
        # Define dimensions
        lat_dim = ncfile.createDimension("lat", datout.shape[0])
        lon_dim = ncfile.createDimension("lon", datout.shape[1])
        ch_dim  = ncfile.createDimension("ch",  datout.shape[2] if len(datout.shape) == 3 else 1)

        # Define variables
        lon_var = ncfile.createVariable("longitude", "f4", ("lat", "lon"))
        lat_var = ncfile.createVariable("latitude", "f4", ("lat", "lon"))
        vza_var = ncfile.createVariable("vza", "f4", ("lat", "lon"))
        vaa_var = ncfile.createVariable("vaa", "f4", ("lat", "lon"))
        count_var = ncfile.createVariable("count", "i4", ("lat", "lon"))
        radiance_var = ncfile.createVariable("radiance", "f4", ("lat", "lon", 'ch'), fill_value=np.nan)
        if reflectance is not None:
            reflectance_var = ncfile.createVariable("hdrf", "f4", ("lat", "lon", 'ch'), fill_value=np.nan)
        if flxdn is not None:
            down_flux_var = ncfile.createVariable("fluxdn", "f4", ('ch'))
        if wvlc is not None:
            center_wvl_var = ncfile.createVariable("wvlc", "f4", ('ch'))
        if sza is not None:
            sza_var = ncfile.createVariable("sza", "f4")
        if saa is not None:
            saa_var = ncfile.createVariable("saa", "f4")
        yyyy_var = ncfile.createVariable("year", "i4")
        mm_var = ncfile.createVariable("month", "i4")
        dd_var = ncfile.createVariable("day", "i4")
        time_var = ncfile.createVariable("time", "f4")

        # Add attributes
        lon_var.long_name = "Longitude"
        lon_var.units = "degree"
        lon_var.description = "Longitude at the surface level."

        lat_var.long_name = "Latitude"
        lat_var.units = "degree"
        lat_var.description = "Latitude at the surface level."
        
        vza_var.long_name = "Viewing zenith angle"
        vza_var.units = "degree"
        vza_var.description = "Viewing zenith angle at the surface level."

        vaa_var.long_name = "Viewing azimuth angle"
        vaa_var.units = "degree"
        vaa_var.description = "Viewing azimuth angle at the surface level."

        count_var.long_name = "Count"
        count_var.units = "dimensionless"
        count_var.description = "Number of image pixels averaged over to generate radiance for a given grid."

        radiance_var.long_name = "Radiance"
        radiance_var.units = "W/m^2/nm/sr"
        radiance_var.description = "Radiance as a function of viewing zenith and azimuth angles."

        if reflectance is not None:
            reflectance_var.long_name = "HDRF"
            reflectance_var.units = "dimensionless"
            reflectance_var.description = "HDRF (reflectance) as a function of viewing zenith and azimuth angles."

        if flxdn is not None:
            down_flux_var.long_name = "Downward irradiance"
            down_flux_var.units = "W/m^2/nm"
            down_flux_var.description = "Downward irradiance averaged over the flight leg. The camera response function is applied."
        
        if wvlc is not None:
            center_wvl_var.long_name = "Center wavelength"
            center_wvl_var.units = "nm"
            center_wvl_var.description = "Center wavelength of the camera's spectral response function."
        
        if sza is not None:
            sza_var.long_name = "Solar Zenith Angle"
            sza_var.units = "degrees"
            sza_var.description = "Solar zenith angle averaged over the flight leg."
        
        if saa is not None:
            saa_var.long_name = "Solar Azimuth Angle"
            saa_var.units = "degrees"
            saa_var.description = "Solar azimuth angle averaged over the flight leg."
        
        yyyy_var.long_name = "Year"
        yyyy_var.units = "dimensionless"
        yyyy_var.description = "Year of the flight"

        mm_var.long_name = "Month"
        mm_var.units = "dimensionless"
        mm_var.description = "Month of the flight"

        dd_var.long_name = "Day"
        dd_var.units = "dimensionless"
        dd_var.description = "Day of the flight"
        
        time_var.long_name = "Time"
        time_var.units = "decimal hour"
        time_var.description = "Time at which the camera data was obtained. It may contain some offset due to time sync error."
        
        ncfile.title = "Gridded image"
        ncfile.description = "Hemispherical radiance derived from a airborne downward hemispherical camera on %s at %s UTC." % (date, tact.strftime("%H%M%S"))

        # Assign data to variables
        lon_var[:, :] = lonxx
        lat_var[:, :] = latyy
        vza_var[:, :] = vzagrid
        vaa_var[:, :] = vaagrid
        count_var[:, :] = ncount
        radiance_var[:, :, :] = datout if len(datout.shape) == 3 else datout[:, :, np.newaxis]
        if reflectance is not None:
            reflectance_var[:, :, :] = reflectance if len(reflectance.shape) == 3 else reflectance[:, :, np.newaxis]
        if flxdn is not None:
            down_flux_var[:] = flxdn
        if wvlc is not None:
            center_wvl_var[:] = wvlc
        if sza is not None:
            sza_var.assignValue(sza)
        if saa is not None:
            saa_var.assignValue(saa)
        yyyy_var.assignValue(date.split('-')[0])
        mm_var.assignValue(date.split('-')[1])
        dd_var.assignValue(date.split('-')[2])
        time_var.assignValue(tact.hour + tact.minute/60. + tact.second/3600. + tact.microsecond/(3600*1e6))
    
def write_angular_grid_to_nc(output_ncfile, datout, zen_yy, razi_xx, date, start_time, end_time, reflectance=None, azi_xx=None, sza=None, saa=None, flxdn=None, wvlc=None, nimg=None, alt=None):
    print('Writing to %s' % (output_ncfile))
    with nc.Dataset(output_ncfile, "w", format="NETCDF4") as ncfile:
        # Define dimensions
        zenith_dim = ncfile.createDimension("zenith", datout.shape[0])
        azimuth_dim = ncfile.createDimension("azimuth", datout.shape[1])
        ch_dim = ncfile.createDimension("ch", datout.shape[2] if len(datout.shape) == 3 else 1)

        # Define variables
        zenith_var = ncfile.createVariable("vza", "f4", ("zenith", "azimuth"))
        relative_azimuth_var = ncfile.createVariable("raa", "f4", ("zenith", "azimuth"))
        if azi_xx is not None:
            azimuth_var = ncfile.createVariable("vaa", "f4", ("zenith", "azimuth"))
        radiance_var = ncfile.createVariable("radiance", "f4", ("zenith", "azimuth", 'ch'), fill_value=np.nan)
        if reflectance is not None:
            reflectance_var = ncfile.createVariable("hdrf", "f4", ("zenith", "azimuth", 'ch'), fill_value=np.nan)
        if flxdn is not None:
            down_flux_var = ncfile.createVariable("fluxdn", "f4", ('ch'))
        if wvlc is not None:
            center_wvl_var = ncfile.createVariable("wvlc", "f4", ('ch'))
        if sza is not None:
            sza_var = ncfile.createVariable("solar_zenith_angle", "f4")
        if saa is not None:
            saa_var = ncfile.createVariable("solar_azimuth_angle", "f4")
        if alt is not None:
            alt_var = ncfile.createVariable("alt", "f4")
        yyyy_var = ncfile.createVariable("year", "i4")
        mm_var = ncfile.createVariable("month", "i4")
        dd_var = ncfile.createVariable("day", "i4")
        st_time_var = ncfile.createVariable("start time", "f4")
        en_time_var = ncfile.createVariable("end time", "f4")
        if nimg is not None:
            nimg_var = ncfile.createVariable("image number", "i4")

        # Add attributes
        zenith_var.long_name = "Viewing Zenith Angle"
        zenith_var.units = "degrees"
        zenith_var.description = "Angle from the vertical axis to the line of sight. Nadir = 0 degrees."

        relative_azimuth_var.long_name = "Relative Azimuth Angle"
        relative_azimuth_var.units = "degrees"
        relative_azimuth_var.description = "Horizontal angle between the line of sight and the principal plane. Sunglint should be at 0 degrees."

        if azi_xx is not None:
            azimuth_var.long_name = "Viewing Azimuth Angle"
            azimuth_var.units = "degrees"
            azimuth_var.description = "Angle from the north direction to the line of sight. North = 0 degrees."

        radiance_var.long_name = "Radiance"
        radiance_var.units = "W/m^2/nm/sr"
        radiance_var.description = "Radiance as a function of viewing zenith and azimuth angles."

        if reflectance is not None:
            reflectance_var.long_name = "HDRF"
            reflectance_var.units = "dimensionless"
            reflectance_var.description = "HDRF (reflectance) as a function of viewing zenith and azimuth angles."

        if flxdn is not None:
            down_flux_var.long_name = "Downward irradiance"
            down_flux_var.units = "W/m^2/nm"
            down_flux_var.description = "Downward irradiance averaged over the flight leg. The camera response function is applied."
        
        if wvlc is not None:
            center_wvl_var.long_name = "Center wavelength"
            center_wvl_var.units = "nm"
            center_wvl_var.description = "Center wavelength of the camera's spectral response function."

        if sza is not None:
            sza_var.long_name = "Solar Zenith Angle"
            sza_var.units = "degrees"
            sza_var.description = "Solar zenith angle averaged over the flight leg."

        if saa is not None:
            saa_var.long_name = "Solar Azimuth Angle"
            saa_var.units = "degrees"
            saa_var.description = "Solar azimuth angle averaged over the flight leg."
        
        if alt is not None:
            alt_var.long_name = "Altitude"
            alt_var.units = "m"
            alt_var.description = "Altitude of the aircraft averaged over the flight leg."

        yyyy_var.long_name = "Year"
        yyyy_var.units = "dimensionless"
        yyyy_var.description = "Year of the flight"

        mm_var.long_name = "Month"
        mm_var.units = "dimensionless"
        mm_var.description = "Month of the flight"

        dd_var.long_name = "Day"
        dd_var.units = "dimensionless"
        dd_var.description = "Day of the flight"

        st_time_var.long_name = "Start time"
        st_time_var.units = "decimal hour"
        st_time_var.description = "Beginning of the time frame over which the camera data was averaged."

        en_time_var.long_name = "End time"
        en_time_var.units = "decimal hour"
        en_time_var.description = "End of the time frame over which the camera data was averaged."

        if nimg is not None:
            nimg_var.long_name = "Image number"
            nimg_var.units = "dimensionless"
            nimg_var.description = "Number of images averaged to produce radiance/HDRF."

        ncfile.title = "HDRF Data"
        ncfile.description = "HDRF (reflectance) values derived from %d airborne downward hemispherical camera images on %s between %s - %s UTC." % (nimg, date, start_time, end_time)

        # Assign data to variables
        zenith_var[:, :] = zen_yy
        relative_azimuth_var[:, :] = razi_xx % 360.
        if azi_xx is not None:
            azimuth_var[:, :] = azi_xx % 360.
        radiance_var[:, :, :] = datout if len(datout.shape) == 3 else datout[:, :, np.newaxis]
        if reflectance is not None:
            reflectance_var[:, :, :] = reflectance if len(reflectance.shape) == 3 else reflectance[:, :, np.newaxis]
        if flxdn is not None:
            down_flux_var[:] = flxdn
        if wvlc is not None:
            center_wvl_var[:] = wvlc
        if sza is not None:
            sza_var.assignValue(sza)
        if saa is not None:
            saa_var.assignValue(saa)
        if alt is not None:
            alt_var.assignValue(alt)
        yyyy_var.assignValue(date.split('-')[0])
        mm_var.assignValue(date.split('-')[1])
        dd_var.assignValue(date.split('-')[2])
        st_time_var.assignValue(start_time)
        en_time_var.assignValue(end_time)
        if nimg is not None:
            nimg_var.assignValue(nimg)
    
def plot_surface_and_angular_grid_image(fn_out, lon_xx, lat_yy, ref_surface, rel_azimuth, zenith, ref_angular, tact):
    fig  = plt.figure(figsize=(8, 4.5))

    cartopy_proj = ccrs.Orthographic(central_longitude=np.nanmean(lon_xx), central_latitude=np.nanmean(lat_yy),)
    ax = fig.add_subplot(121, projection=cartopy_proj)
    xmin, xmax = np.nanmin(lon_xx[~np.isnan(ref_surface)]), np.nanmax(lon_xx[~np.isnan(ref_surface)])
    ymin, ymax = np.nanmin(lat_yy[~np.isnan(ref_surface)]), np.nanmax(lat_yy[~np.isnan(ref_surface)])
    cp = ax.pcolormesh(lon_xx, lat_yy, ref_surface, transform=ccrs.PlateCarree(), zorder=10)
    ax.set_extent([xmin, xmax, ymin, ymax])
    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.5*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.5*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
    g1.top_labels = False
    g1.right_labels = False
    ax.annotate('Surface grid reflectance', (0.01, 1.05), xycoords='axes fraction')
    cbar = fig.colorbar(cp, ax=ax, orientation='horizontal')
    cbar.set_label('Reflectance')
    
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi/2.)
    cw = ax2.pcolormesh(rel_azimuth*np.pi/180., zenith, ref_angular)
    ax2.set_rticks([30., 60., 90.])
    ax2.annotate('Angular reflectance', (-0.10, 1.05), xycoords='axes fraction')
    cbar2 = fig.colorbar(cw, ax=ax2, orientation='horizontal')
    cbar2.set_label('Reflectance')

    fig.suptitle('Camera image: ' + datetime.datetime.strftime(tact, '%Y-%m-%d %H:%M:%S.%f'), fontsize=9)
    fig.tight_layout()

    fig.savefig(fn_out, dpi=300)

def plot_angular_grid_rad_and_ref(fn_out, rel_azimuth, zenith, rad, ref, sza, flxdn, start_time, end_time=None, nimg=None, alt=None, meta={}):
    meta1 = meta.get('rad', {})
    meta2 = meta.get('ref', {})
    vmin1 = meta1.get('vmin', None)
    vmax1 = meta1.get('vmax', None)
    cmap1 = meta1.get('cmap', 'viridis')
    vmin2 = meta2.get('vmin', None)
    vmax2 = meta2.get('vmax', None)
    cmap2 = meta2.get('cmap', 'viridis')

    fig  = plt.figure(figsize=(8, 4.5))

    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(np.pi/2.)
    cw = ax1.pcolormesh(rel_azimuth*np.pi/180., zenith, rad, vmin=vmin1, vmax=vmax1, cmap=cmap1)
    ax1.set_rticks([30., 60., 90.])
    ax1.annotate('Radiance', (-0.10, 1.05), xycoords='axes fraction')
    cbar1 = fig.colorbar(cw, ax=ax1, orientation='horizontal')
    cbar1.set_label(r'Radiance $ \rm (W/m^2/nm/sr) $')

    ax1.annotate('Average SZA = %5.2f deg' % (sza), (0.01, -0.19), xycoords='axes fraction', fontsize=9)

    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_offset(np.pi/2.)
    cw = ax2.pcolormesh(rel_azimuth*np.pi/180., zenith, ref, vmin=vmin2, vmax=vmax2, cmap=cmap2)
    ax2.set_rticks([30., 60., 90.])
    ax2.annotate('Reflectance', (-0.10, 1.05), xycoords='axes fraction')
    if alt is not None: ax2.annotate('Alt: %7.1f m' % alt, ( 0.70, -0.07), xycoords='axes fraction', fontsize=9)
    cbar2 = fig.colorbar(cw, ax=ax2, orientation='horizontal')
    cbar2.set_label('Reflectance')

    ax2.annotate(r'Average Downward flux = %6.3f $ \rm (W/m^2/nm) $' % (flxdn), (0.01, -0.19), xycoords='axes fraction', fontsize=9)

    if end_time is not None:
        if nimg is not None:
            fig.suptitle('Average of %d images between %s - %s ' % (nimg, start_time, end_time), fontsize=9)
        else:
            fig.suptitle('Images averaged between %s - %s ' % (start_time, end_time), fontsize=9)
    else:
        fig.suptitle('Image at %s' % (start_time), fontsize=9)
    
    fig.tight_layout()

    fig.savefig(fn_out, dpi=300)