import os
import numpy as np
import datetime
import glob
import h5py
import pysolar
from astropy.io import fits
from pyproj import Geod as geod

class Camera_arcsix:
	def __init__(self, date):
		self.date = date

		# Constants for ARCSIX flight meta data

		self._flights = {
			'2024-05-17': {
				'description'	:	'Test flight 1',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-05-21':{
				'description'	:	'Test flight 2',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-05-23':{
				'description'	:	'Transit flight to Pituffik (Spring)',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-05-24':{
				'description'	:	'Test flight 3',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-05-28':{
				'description'	:	'Research Flight 01',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-05-30':{
				'description'	:	'Research Flight 02',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-05-31':{
				'description'	:	'Research Flight 03',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-03':{
				'description'	:	'Research Flight 04',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-05':{
				'description'	:	'Research Flight 05',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-06':{
				'description'	:	'Research Flight 06',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-07':{
				'description'	:	'Research Flight 07',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-10':{
				'description'	:	'Research Flight 08',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-11':{
				'description'	:	'Research Flight 09',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-06-13':{
				'description'	:	'Research Flight 10',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-07-08':{
				'description'	:	'Ground Test',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-07-09':{
				'description'	:	'Ground Test',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-07-22':{
				'description'	:	'Transit flight to Pituffik (Summer)',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-07-25':{
				'description'	:	'Research Flight 11',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-07-29':{
				'description'	:	'Research Flight 12',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-07-30':{
				'description'	:	'Research Flight 13',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-01':{
				'description'	:	'Research Flight 14',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-02':{
				'description'	:	'Research Flight 15',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-07':{
				'description'	:	'Research Flight 16',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-08':{
				'description'	:	'Research Flight 17',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-09':{
				'description'	:	'Research Flight 18',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-15':{
				'description'	:	'Research Flight 19',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
			'2024-08-16':{
				'description'	:	'Transit flight to Bangor (Summer)',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:	0.,
			},
		}

		self.radcal_dict_simple = {
			'red'	: {	'c0':       0.03697803089161538   ,   # red channel cal main term
					    'c2':       2.9306376815596668e-08,   # quadratic correction term
					    'lambda': 626.408                 ,   # center wavelength [nm]
					    'f_toa':    1.6687                },  # TOA solar irradiance [W/m2/nm]
			'green'	: {	'c0':       0.0379816678515955    ,   # green channel cal main term
					    'c2':       2.4174566687203538e-08,   # quadratic correction term
					    'lambda': 553.572                 ,   # center wavelength [nm]
					    'f_toa':    1.8476                },  # TOA solar irradiance [W/m2/nm]}
			'blue'	: {	'c0':       0.04097034968881991   ,   # blue channel cal main term
					    'c2':       2.3561354650281456e-08,   # quadratic correction term
					    'lambda': 492.667                 ,   # center wavelength [nm]
					    'f_toa':    1.9084                },  # TOA solar irradiance [W/m2/nm]}
			'type'	: {	'pre':    'count'                 ,   # type before conversion
					    'post':   'radiance'              }   # type after conversion
		}

		self.flipud = True
		self.fliplr = True
		self.leftpix  = np.array([2190, 1703])
		self.rightpix = np.array([1004,  162])
		self.centerpix = np.int_(0.5*(self.leftpix + self.rightpix))
		self.headvector = np.array([-(self.rightpix[1] - self.centerpix[1]),
									 (self.rightpix[0] - self.centerpix[0])])

		self.pit_off =  1.5  # deg
		self.rol_off =  1.5  # deg
		self.hed_off =  0.0  # deg rotate clockwise


		self.hsk_dict = None
		self.hrss = None
		self.lats = None
		self.lons = None
		self.alts = None
		self.pits = None
		self.rols = None
		self.heds = None

		self.fits_allfiles = None
		
		self.date_check(self.date)

		self.flight_meta = self._flights[date]
		self.flight_date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
		
		self.camera_setting = {	'centerpix'	:	self.centerpix,
								'headvector':	self.headvector,
								'pit_off'	:	self.pit_off,
								'hed_off'	:	self.hed_off,
								'rol_off'	:	self.rol_off,
								'flipud'	:	self.flipud,
								'fliplr'	:	self.fliplr,}
		
	def date_check(self, date):
		if date not in self._flights:
			message = 'Error [Meta_arcsix]: date %s is not in the list of flight dates. ' \
						+ 'Make sure it is in YYYY-MM-DD format.'
			raise OSError(message)
		else:
			if not self._flights[date]['available']:
				message = 'Error [Meta_arcsix]: date %s is not avaiable.'
				raise OSError(message)
			else:
				print('Camera images for [%s]' % self._flights[date]['description'])
	
	def load_hsk(self, location='argus'):
		if location == 'argus':
			path = '/Volumes/argus/field/arcsix/processed'
		else:
			path = location
		yyyy, mm, dd = self.date.split('-')
		hsk_fname = 'ARCSIX-HSK_P3B_%s%s%s_*.h5' % (yyyy, mm, dd)
		hsk_path = os.path.join(path, hsk_fname)
		hsk_files  = glob.glob(hsk_path)
		if len(hsk_files) == 0:
			message = 'Error [Meta_arcsix.load_hsk]: No housekeeping file found. Check the path/access to the correct directory.'
			raise OSError(message)
		elif len(hsk_files) != 1:
			message = 'Error [Meta_arcsix.load_hsk]: More than 1 housekeeping file found. Check the files in the directory.'
			raise OSError(message)
		else:
			hsk_fn = hsk_files[0]
		
		self.hsk_dict = self.read_hsk(hsk_fn)

		self.hrss = self.hsk_dict['hrs']
		self.lats = self.hsk_dict['lat']
		self.lons = self.hsk_dict['lon']
		self.alts = self.hsk_dict['alt']
		self.pits = self.hsk_dict['pit'] # [deg]
		self.rols = self.hsk_dict['rol'] # [deg]
		self.heds = self.hsk_dict['hed'] # [deg]


	def read_hsk(self, hsk_filename):
		print('Reading {}'.format(hsk_filename))
		h5f = h5py.File(hsk_filename, 'r')
		hsk_data = {'doy': self.jd_to_doy(h5f['jday'][...], 2024) ,
					'hrs': h5f['tmhr'][...]        ,
					'alt': h5f['alt'][...],
					'lat': h5f['lat'][...]    ,
					'lon': h5f['lon'][...]   ,
					'hed': h5f['ang_hed'][...],
					'rol': h5f['ang_rol'][...]  ,
					'pit': h5f['ang_pit'][...]  }
		return hsk_data
	
	def jd_to_doy(self, jd, year):
		return 1721425.5 + 365 * (year - 1) + int((year - 1) / 4)
	
	def load_fits(self, start_time, end_time, location='ARCSIX1KSS'):
		self.fits_allfiles = self.locate_allfits(location=location)

		st_dt = datetime.datetime.strptime(start_time, '%H:%M:%S')
		# if campaign.lower() == 'camp2ex' and st_dt.hour < 15: st_dt += datetime.timedelta(days=1)
		en_dt = datetime.datetime.strptime(end_time, '%H:%M:%S')
		# if campaign.lower() == 'camp2ex' and en_dt.hour < 15: en_dt += datetime.timedelta(days=1)

		self.t_offset = datetime.timedelta(	days=self.flight_meta['toff_day'], 
											seconds=self.flight_meta['toff_sec'])

		fits_list = []
		lat_list, lon_list = [], []
		for ifits, fits_fn in enumerate(self.fits_allfiles):
			tt_fn = datetime.datetime.strptime(fits_fn[-14:-6], '%H_%M_%S')
			# if campaign.lower() == 'camp2ex' and tt_fn.hour < 15: tt_fn += datetime.timedelta(days=1)
			if st_dt <= tt_fn <= en_dt:
				fits_list.append(fits_fn)
				t_fn  = datetime.datetime.strptime(self.date + ' ' + fits_fn[-14:-6], '%Y-%m-%d %H_%M_%S')
				hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + self.t_offset.total_seconds()/3600.
				lat  = np.interp(hrs, self.hrss, self.lats) # interpolate to find values for the time at which the image was taken
				lon  = np.interp(hrs, self.hrss, self.lons)
				lat_list.append(lat)
				lon_list.append(lon)
		self.lat_list = np.array(lat_list)
		self.lon_list = np.array(lon_list)
		return fits_list
	
	def locate_allfits(self, location='ARCSIX1KSS'):
		yyyy, mm, dd = self.date.split('-')
		num_rf = self.flight_meta['description'].split(' ')[-1]
		if location in ['ARCSIX1KSS', 'ARCISX3KSS']:
			if self.flight_date <= datetime.datetime(2024, 8, 3):
				path = '/Volumes/ARCSIX1KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
			else:
				path = '/Volumes/ARCSIX3KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
		elif location in ['ARCSIX2KSS', 'ARCISX4KSS']:
			if self.flight_date <= datetime.datetime(2024, 8, 3):
				path = '/Volumes/ARCSIX2KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
			else:
				path = '/Volumes/ARCSIX4KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
		else:
			path = location
		
		fits_allfiles = glob.glob(path, recursive=True)
		if len(fits_allfiles) == 0:
			message = 'Error [Meta_arcsix.load_fits]: No fits file found. Check the path/access to the correct directory.'
			raise OSError(message)
		fits_allfiles = sorted(fits_allfiles)
		return fits_allfiles
	
	# def interpolate_hsk_for_fits(self, fits_fn):
		# t_fn  = datetime.datetime.strptime(self.date + ' ' + fits_fn[-14:-6], '%Y-%m-%d %H_%M_%S')
	def interpolate_hsk_for_fits(self, date_obs_str):
		t_fn  = datetime.datetime.strptime(self.date + ' ' + date_obs_str.split('T')[1][:-1], '%Y-%m-%d %H:%M:%S.%f')
		t_act = t_fn + self.t_offset
		hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + self.t_offset.total_seconds()/3600.

		lat  = np.interp(hrs, self.hrss, self.lats) # interpolate to find values for the time at which the image was taken
		lon  = np.interp(hrs, self.hrss, self.lons)
		alt  = np.interp(hrs, self.hrss, self.alts)
		pit  = np.interp(hrs, self.hrss, self.pits)
		rol  = np.interp(hrs, self.hrss, self.rols)
		hed  = np.interp(hrs, self.hrss, self.heds)

		when = t_act.replace(tzinfo=datetime.timezone.utc)
		sza  = pysolar.solar.get_altitude(lat, lon, when)
		saa  = pysolar.solar.get_azimuth( lat, lon, when, elevation=alt)

		aircraft_status = {	'lat'		:	lat,
							'lon'		:	lon,
							'rol'		:	rol,
							'pit'		:	pit,
							'hed'		:	hed,
							'alt'		:	alt,
							'sza'		:	sza,
							'saa'		:	saa}
		return t_act, aircraft_status
