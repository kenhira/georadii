import os
import numpy as np
import datetime
import glob
import h5py
from astropy.time import Time


class LVIS_arcsix:
	def __init__(self, date):
		self.date = date

		# Constants for ARCSIX flight meta data

		self._flights = {
			'2024-05-17': {
				'description'	:	'Test flight 1',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-05-21':{
				'description'	:	'Test flight 2',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-05-23':{
				'description'	:	'Transit flight to Pituffik (Spring)',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-05-24':{
				'description'	:	'Test flight 3',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-05-28':{
				'description'	:	'Research Flight 01',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-05-30':{
				'description'	:	'Research Flight 02',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-05-31':{
				'description'	:	'Research Flight 03',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-03':{
				'description'	:	'Research Flight 04',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-05':{
				'description'	:	'Research Flight 05',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-06':{
				'description'	:	'Research Flight 06',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-07':{
				'description'	:	'Research Flight 07',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-10':{
				'description'	:	'Research Flight 08',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-11':{
				'description'	:	'Research Flight 09',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-06-13':{
				'description'	:	'Research Flight 10',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-07-08':{
				'description'	:	'Ground Test',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-07-09':{
				'description'	:	'Ground Test',
				'available'		:	False,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-07-22':{
				'description'	:	'Transit flight to Pituffik (Summer)',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-07-25':{
				'description'	:	'Research Flight 11',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-07-29':{
				'description'	:	'Research Flight 12',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-07-30':{
				'description'	:	'Research Flight 13',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-01':{
				'description'	:	'Research Flight 14',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-02':{
				'description'	:	'Research Flight 15',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-07':{
				'description'	:	'Research Flight 16',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-08':{
				'description'	:	'Research Flight 17',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-09':{
				'description'	:	'Research Flight 18',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-15':{
				'description'	:	'Research Flight 19',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
			'2024-08-16':{
				'description'	:	'Transit flight to Bangor (Summer)',
				'available'		:	True,
				'toff_day'		:	0.,
				'toff_sec'		:  18.,
			},
		}
		
		self.date_check(self.date)

		self.flight_meta = self._flights[date]
		self.flight_date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
		
		
	def date_check(self, date):
		if date not in self._flights:
			message = 'Error [LVIS_arcsix]: date %s is not in the list of flight dates. ' \
						+ 'Make sure it is in YYYY-MM-DD format.'
			raise OSError(message)
		else:
			if not self._flights[date]['available']:
				message = 'Error [LVIS_arcsix]: date %s is not avaiable.'
				raise OSError(message)
			else:
				print('LVIS data for [%s]' % self._flights[date]['description'])
	
	def load_lvis_l2(self, start_time, end_time, location='.'):
		self.txt_allfiles = self.locate_allfiles(location=location, kind='l2')

		st_dt = datetime.datetime.strptime(self.date + ' ' + start_time, '%Y-%m-%d %H:%M:%S')
		# if campaign.lower() == 'camp2ex' and st_dt.hour < 15: st_dt += datetime.timedelta(days=1)
		en_dt = datetime.datetime.strptime(self.date + ' ' + end_time, '%Y-%m-%d %H:%M:%S')
		# if campaign.lower() == 'camp2ex' and en_dt.hour < 15: en_dt += datetime.timedelta(days=1)

		self.t_offset = datetime.timedelta(	days=self.flight_meta['toff_day'], 
											seconds=self.flight_meta['toff_sec'])
		files_in_range = self.find_files_in_range(self.txt_allfiles, st_dt + self.t_offset, en_dt + self.t_offset, kind='l2')

		if len(files_in_range) == 0:
			message = 'Error [LVIS_arcsix.load_lvis_l2]: No txt file found. Check the path/access to the correct directory.'
			raise OSError(message)
		
		for ifile, file in enumerate(files_in_range):
			with open(file, 'r') as f:
				foundheader = 0
				ln = 0
				for line in f:
					ln += 1
					if line.startswith('# LFID'):
						headerfields = line.split()[1:]
						foundheader = 1
						break
				if not foundheader:
					message = 'Error [LVIS_arcsix.load_lvis_l2]: No header found in %s' % file
					raise OSError(message)
		        
		# glat_arr, glon_arr = np.array([], dtype=float), np.array([], dtype=float)
		# vza_arr, vaa_arr = np.array([], dtype=float), np.array([], dtype=float)
		# plat_arr, plon_arr = np.array([], dtype=float), np.array([], dtype=float)
		# time_arr = np.array([], dtype='datetime64')
		# intens_arr = np.zeros((0, 9), dtype=float)
		lat_low_arr, lon_low_arr, z_low_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		lat_maxamp_arr, lon_maxamp_arr, z_maxamp_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		lat_high_arr, lon_high_arr, z_high_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		lat_lowalt_arr, lon_lowalt_arr, z_lowalt_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		azimuth_arr, incident_arr, rangev_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		timev_arr = np.array([], dtype='datetime64[us]')

		basedate = datetime.datetime.strptime(self.date, '%Y-%m-%d')

		for ifile, file in enumerate(files_in_range):
			with open(file, 'r') as f:
				print('Processing file %s' % file)

				n_shot = sum(1 for line in f if not line.startswith('#'))
				print(f"Number of lines in the file: {n_shot}")

				lon_low, lat_low, z_low = np.zeros(n_shot), np.zeros(n_shot), np.zeros(n_shot)
				lon_maxamp, lat_maxamp, z_maxamp = np.zeros(n_shot), np.zeros(n_shot), np.zeros(n_shot)
				lon_high, lat_high, z_high = np.zeros(n_shot), np.zeros(n_shot), np.zeros(n_shot)
				lon_lowalt, lat_lowalt, z_lowalt = np.ones(n_shot)*np.nan, np.ones(n_shot)*np.nan, np.ones(n_shot)*np.nan
				azimuth, incident, rangev = np.zeros(n_shot), np.zeros(n_shot), np.zeros(n_shot)
				timev = np.zeros(n_shot, dtype='datetime64[us]')


				f.seek(0)
				it = 0
				for line in f:
					if not line.startswith('#'):
						items = line.split()
						lon_low[it]    = float(items[headerfields.index('LON_LOW')])
						lat_low[it]    = float(items[headerfields.index('LAT_LOW')])
						z_low[it]      = float(items[headerfields.index('Z_LOW')])
						lon_maxamp[it] = float(items[headerfields.index('LON_MAXAMP')])
						lat_maxamp[it] = float(items[headerfields.index('LAT_MAXAMP')])
						z_maxamp[it]   = float(items[headerfields.index('Z_MAXAMP')])
						lon_high[it]   = float(items[headerfields.index('LON_HIGH')])
						lat_high[it]   = float(items[headerfields.index('LAT_HIGH')])
						z_high[it]     = float(items[headerfields.index('Z_HIGH')])
						lon_lowalt[it] = float(items[headerfields.index('LON_LOW_ALT')])
						lat_lowalt[it] = float(items[headerfields.index('LAT_LOW_ALT')])
						z_lowalt[it]   = float(items[headerfields.index('Z_LOW_ALT')])
						azimuth[it]    = float(items[headerfields.index('AZIMUTH')])
						incident[it]   = float(items[headerfields.index('INCIDENTANGLE')])
						rangev[it]     = float(items[headerfields.index('RANGE')])
						timev[it] = np.datetime64(basedate + datetime.timedelta(seconds=float(items[headerfields.index('TIME')]))).astype('datetime64[us]')
						it += 1

				lat_low_arr = np.append(lat_low_arr, lat_low)
				lon_low_arr = np.append(lon_low_arr, lon_low)
				z_low_arr = np.append(z_low_arr, z_low)
				lat_maxamp_arr = np.append(lat_maxamp_arr, lat_maxamp)
				lon_maxamp_arr = np.append(lon_maxamp_arr, lon_maxamp)
				z_maxamp_arr = np.append(z_maxamp_arr, z_maxamp)
				lat_high_arr = np.append(lat_high_arr, lat_high)
				lon_high_arr = np.append(lon_high_arr, lon_high)
				z_high_arr = np.append(z_high_arr, z_high)
				lat_lowalt_arr = np.append(lat_lowalt_arr, lat_lowalt)
				lon_lowalt_arr = np.append(lon_lowalt_arr, lon_lowalt)
				z_lowalt_arr = np.append(z_lowalt_arr, z_lowalt)
				azimuth_arr = np.append(azimuth_arr, azimuth)
				incident_arr = np.append(incident_arr, incident)
				rangev_arr = np.append(rangev_arr, rangev)
				timev_arr = np.append(timev_arr, timev)

		ipixels = np.where((st_dt + self.t_offset < timev_arr) & (timev_arr < en_dt + self.t_offset))[0]
		print('Number of pixels in range:', len(ipixels))

		if len(ipixels) == 0:
			message = 'Error [LVIS_arcsix.load_lvis_l2]: No pixels found in the specified time range.'
			raise OSError(message)

		colpix  = np.zeros((len(ipixels), 1, 3))
		latpix  = np.zeros((len(ipixels)))
		lonpix  = np.zeros((len(ipixels)))
		latallpix = np.zeros((len(ipixels), 3))
		lonallpix = np.zeros((len(ipixels), 3))
		azimuthpix  = np.zeros((len(ipixels)))
		incidentpix  = np.zeros((len(ipixels)))
		rangepix  = np.zeros((len(ipixels)))
		timepix = np.zeros((len(ipixels)), dtype='datetime64[us]')

		colpix[:, 0, 0]  = z_low_arr[ipixels]
		colpix[:, 0, 1]  = z_maxamp_arr[ipixels]
		colpix[:, 0, 2]  = z_high_arr[ipixels]
		latpix[:]     = lat_maxamp_arr[ipixels]
		lonpix[:]     = lon_maxamp_arr[ipixels]
		latallpix[:, 0]  = lat_low_arr[ipixels]
		latallpix[:, 1]  = lat_maxamp_arr[ipixels]
		latallpix[:, 2]  = lat_high_arr[ipixels]
		lonallpix[:, 0]  = lon_low_arr[ipixels]
		lonallpix[:, 1]  = lon_maxamp_arr[ipixels]
		lonallpix[:, 2]  = lon_high_arr[ipixels]
		azimuthpix[:] = azimuth_arr[ipixels]
		incidentpix[:] = incident_arr[ipixels]
		rangepix[:]   = rangev_arr[ipixels]
		timepix[:]    = timev_arr[ipixels]

		img = {'data': colpix, 'type': 'altitude', 'unit': 'altitude', 'alttype': ['low', 'maxamp', 'high']}
		# latlon_meta = {'longeo': lonpix, 'latgeo': latpix}#, 'timepix': timepix}
		latlon_meta = {'latgeo': latpix, 'longeo': lonpix, 'latall': latallpix, 'lonall': lonallpix, 'azimuth': azimuthpix, 'incident': incidentpix, 'range': rangepix, 'time': timepix}
		return img, latlon_meta
	
	def load_lvis_l1b(self, start_time, end_time, location='.', load='light'):
		self.txt_allfiles = self.locate_allfiles(location=location, kind='l1b')

		st_dt = datetime.datetime.strptime(self.date + ' ' + start_time, '%Y-%m-%d %H:%M:%S')
		# if campaign.lower() == 'camp2ex' and st_dt.hour < 15: st_dt += datetime.timedelta(days=1)
		en_dt = datetime.datetime.strptime(self.date + ' ' + end_time, '%Y-%m-%d %H:%M:%S')
		# if campaign.lower() == 'camp2ex' and en_dt.hour < 15: en_dt += datetime.timedelta(days=1)

		self.t_offset = datetime.timedelta(	days=self.flight_meta['toff_day'], 
											seconds=self.flight_meta['toff_sec'])
		files_in_range = self.find_files_in_range(self.txt_allfiles, st_dt + self.t_offset, en_dt + self.t_offset, kind='l1b')

		if len(files_in_range) == 0:
			message = 'Error [LVIS_arcsix.load_lvis_l1b]: No h5 file found. Check the path/access to the correct directory.'
			raise OSError(message)
		
		lat0_arr, lon0_arr, z0_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		lat_maxamp_arr, lon_maxamp_arr, z_maxamp_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		lat1215_arr, lon1215_arr, z1215_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		rx_arr = np.zeros((0, 1216), dtype=float)
		sigmean_arr = np.array([], dtype=float)
		azimuth_arr, incident_arr, rangev_arr = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
		timev_arr = np.array([], dtype='datetime64[us]')

		basedate = datetime.datetime.strptime(self.date, '%Y-%m-%d')

		for ifile, file in enumerate(files_in_range):
			print('Processing file %s' % file)
			with h5py.File(file, 'r') as f:
				lat0 = f['LAT0'][:]
				lon0 = f['LON0'][:]
				z0 = f['Z0'][:]
				lat1215 = f['LAT1215'][:]
				lon1215 = f['LON1215'][:]
				z1215 = f['Z1215'][:]
				rx = f['RXWAVE'][:, :]
				sigmean = f['SIGMEAN'][:]
				azimuth = f['AZIMUTH'][:]
				incident = f['INCIDENTANGLE'][:]
				rangev = f['RANGE'][:]
				timev = np.array([np.datetime64(basedate + datetime.timedelta(seconds=t)) for t in f['TIME'][:]], dtype='datetime64[us]')
				
				idx_maxamp = np.argmax(rx, axis=1)
				lat_maxamp = lat0 + (lat1215 - lat0) * idx_maxamp / 1216.
				lon_maxamp = lon0 + (lon1215 - lon0) * idx_maxamp / 1216.
				z_maxamp   = z0 + (z1215 - z0) * idx_maxamp / 1216.

				# Append data to arrays
				lat0_arr = np.append(lat0_arr, lat0)
				lon0_arr = np.append(lon0_arr, lon0)
				z0_arr = np.append(z0_arr, z0)
				lat_maxamp_arr = np.append(lat_maxamp_arr, lat_maxamp)
				lon_maxamp_arr = np.append(lon_maxamp_arr, lon_maxamp)
				z_maxamp_arr = np.append(z_maxamp_arr, z_maxamp)
				lat1215_arr = np.append(lat1215_arr, lat1215)
				lon1215_arr = np.append(lon1215_arr, lon1215)
				z1215_arr = np.append(z1215_arr, z1215)
				rx_arr = np.append(rx_arr, rx, axis=0)
				sigmean_arr = np.append(sigmean_arr, sigmean)
				azimuth_arr = np.append(azimuth_arr, azimuth)
				incident_arr = np.append(incident_arr, incident)
				rangev_arr = np.append(rangev_arr, rangev)
				timev_arr = np.append(timev_arr, timev)
		
		ipixels = np.where((st_dt + self.t_offset < timev_arr) & (timev_arr < en_dt + self.t_offset))[0]
		print('Number of pixels in range:', len(ipixels))

		if len(ipixels) == 0:
			message = 'Error [LVIS_arcsix.load_lvis_l1b]: No pixels found in the specified time range.'
			raise OSError(message)
		
		colpix  = np.zeros((len(ipixels), 1, 3))
		rxpix  = np.zeros((len(ipixels), 1216))
		latpix  = np.zeros((len(ipixels)))
		lonpix  = np.zeros((len(ipixels)))
		latallpix = np.zeros((len(ipixels), 3))
		lonallpix = np.zeros((len(ipixels), 3))
		azimuthpix  = np.zeros((len(ipixels)))
		incidentpix  = np.zeros((len(ipixels)))
		rangepix  = np.zeros((len(ipixels)))
		timepix = np.zeros((len(ipixels)), dtype='datetime64[us]')


		colpix[:, 0, 0]  = z0_arr[ipixels]
		colpix[:, 0, 1]  = z_maxamp_arr[ipixels]
		colpix[:, 0, 2]  = z1215_arr[ipixels]
		rxpix[:, :]   = rx_arr[ipixels, :]
		latpix[:]     = lat_maxamp_arr[ipixels]
		lonpix[:]     = lon_maxamp_arr[ipixels]
		latallpix[:, 0]  = lat0_arr[ipixels]
		latallpix[:, 1]  = lat_maxamp_arr[ipixels]
		latallpix[:, 2]  = lat1215_arr[ipixels]
		lonallpix[:, 0]  = lon0_arr[ipixels]
		lonallpix[:, 1]  = lon_maxamp_arr[ipixels]
		lonallpix[:, 2]  = lon1215_arr[ipixels]
		azimuthpix[:] = azimuth_arr[ipixels]
		incidentpix[:] = incident_arr[ipixels]
		rangepix[:]   = rangev_arr[ipixels]
		timepix[:]    = timev_arr[ipixels]
		img = {'data': colpix, 'type': 'altitude', 'unit': 'altitude', 'alttype': ['low', 'maxamp', 'high']}
		# latlon_meta = {'longeo': lonpix, 'latgeo': latpix, 'rx': rxpix, 'timepix': timepix}
		if load.lower() == 'light':
			latlon_meta = {'latgeo': latpix, 'longeo': lonpix, 'rx': rxpix, 'time': timepix}
		elif load.lower() == 'full':
			latlon_meta = {'latgeo': latpix, 'longeo': lonpix, 'rx': rxpix, 'azimuth': azimuthpix, 'incident': incidentpix, 'range': rangepix, 'time': timepix}
		return img, latlon_meta
	
	def locate_allfiles(self, location='.', kind='l2'):
		yyyy, mm, dd = self.date.split('-')
		if True:
			# pathl1b = os.path.join(location, 'LVISF1B_ARCSIX%s_%s%s_R*.h5' % (yyyy, mm, dd)) #LVISF1B_ARCSIX2024_0605_R2503_047591.h5
			if kind.lower() == 'l2':
				pathl  = os.path.join(location, 'LVISF2_IS_ARCSIX%s_%s%s_R*.TXT' % (yyyy, mm, dd))
			elif kind.lower() == 'l1b':
				pathl  = os.path.join(location, 'LVISF1B_ARCSIX%s_%s%s_R*.h5' % (yyyy, mm, dd))
			else:
				message = 'Error [LVIS_arcsix.locate_allfiles]: kind must be either "l2" or "l1b".'
				raise ValueError(message)
		txt_allfiles = glob.glob(pathl, recursive=True)
		if len(txt_allfiles) == 0:
			message = 'Error [LVIS_arcsix.locate_allfiles]: No txt file found. Check the path/access to the correct directory.'
			raise OSError(message)
		txt_allfiles = sorted(txt_allfiles)
		return txt_allfiles

	def find_files_in_range(self, txt_files, start_time, end_time, kind='l2'):
		file_timestamps = []
		# print(txt_files, start_time, end_time)
		for file in txt_files:
			# timestamp_str = os.path.basename(file).split('_')[2]
			# timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
			if kind.lower() == 'l2':
				sec_str = os.path.basename(file).split('_')[5].split('.')[0] #LVISF2_IS_ARCSIX2024_0605_R2503_047591.TXT
			elif kind.lower() == 'l1b':
				sec_str = os.path.basename(file).split('_')[4].split('.')[0] #LVISF1B_ARCSIX2024_0605_R2503_047591.h5
			else:
				message = 'Error [LVIS_arcsix.find_files_in_range]: kind must be either "l2" or "l1b".'
				raise ValueError(message)
			time_str = "%02d%02d%02d" % (int(sec_str) // 3600, (int(sec_str) % 3600) // 60, int(sec_str) % 60)
			timestamp = datetime.datetime.strptime(self.date + ' ' + time_str, "%Y-%m-%d %H%M%S")
			file_timestamps.append((file, timestamp))
		files_all = sorted([file for file, timestamp in file_timestamps])
		files_before_start = sorted([file for file, timestamp in file_timestamps if start_time > timestamp])
		files_before_end   = sorted([file for file, timestamp in file_timestamps if timestamp < end_time])
		istart = len(files_before_start)
		iend   = len(files_before_end)
		if   istart == iend == 0:
			return []
		else:
			files_in_range = files_all[max(istart - 1, 0):iend]
			return files_in_range
