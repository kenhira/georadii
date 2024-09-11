import os
import numpy as np
import datetime
import glob
import h5py
from astropy.time import Time


class RSP_arcsix:
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
		
		self.date_check(self.date)

		self.flight_meta = self._flights[date]
		self.flight_date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
		
		
	def date_check(self, date):
		if date not in self._flights:
			message = 'Error [RSP_arcsix]: date %s is not in the list of flight dates. ' \
						+ 'Make sure it is in YYYY-MM-DD format.'
			raise OSError(message)
		else:
			if not self._flights[date]['available']:
				message = 'Error [RSP_arcsix]: date %s is not avaiable.'
				raise OSError(message)
			else:
				print('Camera images for [%s]' % self._flights[date]['description'])
	
	def load_rsp_h5(self, start_time, end_time, location='.'):
		self.h5_allfiles = self.locate_allh5(location=location)

		st_dt = datetime.datetime.strptime(self.date + ' ' + start_time, '%Y-%m-%d %H:%M:%S')
		# if campaign.lower() == 'camp2ex' and st_dt.hour < 15: st_dt += datetime.timedelta(days=1)
		en_dt = datetime.datetime.strptime(self.date + ' ' + end_time, '%Y-%m-%d %H:%M:%S')
		# if campaign.lower() == 'camp2ex' and en_dt.hour < 15: en_dt += datetime.timedelta(days=1)

		self.t_offset = datetime.timedelta(	days=self.flight_meta['toff_day'], 
											seconds=self.flight_meta['toff_sec'])
		files_in_range = self.find_files_in_range(self.h5_allfiles, st_dt + self.t_offset, en_dt + self.t_offset)

		if len(files_in_range) == 0:
			message = 'Error [RSP_arcsix.load_rsp_h5]: No h5 file found. Check the path/access to the correct directory.'
			raise OSError(message)
		
		for ifile, file in enumerate(files_in_range):
			with h5py.File(file, 'r') as h5_file:
				nscan   = h5_file['dim_Scans'][...].shape[0]
				nsector = h5_file['dim_Scene_Sectors'][...].shape[0]
				nband   = h5_file['dim_Bands'][...].shape[0]
				wvls    = h5_file['Data']['Wavelength'][...][0:nband]
        
		glat_arr, glon_arr = np.array([], dtype=float), np.array([], dtype=float)
		plat_arr, plon_arr = np.array([], dtype=float), np.array([], dtype=float)
		time_arr = np.array([], dtype='datetime64')
		intens_arr = np.zeros((0, 9), dtype=float)
		for ifile, file in enumerate(files_in_range):
			with h5py.File(file, 'r') as h5_file:
				nscan   = h5_file['dim_Scans'][...].shape[0]
				nsector = h5_file['dim_Scene_Sectors'][...].shape[0]
				nband   = h5_file['dim_Bands'][...].shape[0]
				ground_latitude  = h5_file['Geometry']['Ground_Latitude'][...][0:2, 0:nscan, 0:nsector]
				ground_longitude = h5_file['Geometry']['Ground_Longitude'][...][0:2, 0:nscan, 0:nsector]
				platform_latitude  = h5_file['Platform']['Platform_Latitude'][...][0:nscan]
				platform_longitude = h5_file['Platform']['Platform_Longitude'][...][0:nscan]
				# time_scan        = h5_file['Platform']['Fraction_of_Day'][...][0:nscan]
				time_pixel       = h5_file['Geometry']['Measurement_Time'][...][0:2, 0:nscan, 0:nsector]
				intensity_1   = h5_file['Data']['Intensity_1'][...][0:nscan, 0:nsector, 0:nband]
				intensity_2   = h5_file['Data']['Intensity_2'][...][0:nscan, 0:nsector, 0:nband]            
				intensity_avg = (intensity_1 + intensity_2) / 2
				glat_arr = np.append(glat_arr, ground_latitude[0, 0:nscan, 0:nsector].flatten())
				glon_arr = np.append(glon_arr, ground_longitude[0, 0:nscan, 0:nsector].flatten())
				plat_arr = np.append(plat_arr, platform_latitude[0:nscan].flatten())
				plon_arr = np.append(plon_arr, platform_longitude[0:nscan].flatten())
				# time_arr = np.append(time_arr, np.array([datetime.strptime(start_time_str.split(' ')[0] + ' 00:00', '%Y-%m-%d %H:%M') + datetime.timedelta(days=fofd) for fofd in time_scan[0:nscan]]))
				time_arr = np.append(time_arr, Time(time_pixel[0, 0:nscan, 0:nsector].flatten(), format='mjd').to_datetime())
				intens_arr = np.append(intens_arr, intensity_avg[0:nscan, 0:nsector, 0:nband].reshape(nscan*nsector, nband), axis=0)
		ipixels = np.where((st_dt + self.t_offset < time_arr) & (time_arr < en_dt + self.t_offset))[0]
		colpix  = np.zeros((len(ipixels), 1, 3))
		latpix  = np.zeros((len(ipixels), 1))
		lonpix  = np.zeros((len(ipixels), 1))
		timepix = np.zeros((len(ipixels), 1), dtype='datetime64')
		# print(ipixels)
		# print(wvls)
		# print(colpix.shape)
		colpix[:, 0, 0]  = intens_arr[ipixels, 3]
		colpix[:, 0, 1]  = intens_arr[ipixels, 2]
		colpix[:, 0, 2]  = intens_arr[ipixels, 1]
		latpix[:, 0]     = glat_arr[ipixels]
		lonpix[:, 0]     = glon_arr[ipixels]
		# timepix[:, 0]    = time_arr[ipixels]
		img = {'data': colpix, 'type': 'radiance', 'unit': 'radiance'}
		latlon_meta = {'longeo': lonpix, 'latgeo': latpix}#, 'timepix': timepix}
		return img, latlon_meta
	
	def locate_allh5(self, location='.'):
		yyyy, mm, dd = self.date.split('-')
		if True:
			path = os.path.join(location, 'ARCSIX-RSP-L1C_P3B_%s%s%s_R01/*.h5' % (yyyy, mm, dd))
			# path = location
		h5_allfiles = glob.glob(path, recursive=True)
		if len(h5_allfiles) == 0:
			message = 'Error [RSP_arcsix.locate_allh5]: No h5 file found. Check the path/access to the correct directory.'
			raise OSError(message)
		h5_allfiles = sorted(h5_allfiles)
		return h5_allfiles

	def find_files_in_range(self, h5_files, start_time, end_time):
		file_timestamps = []
		print(h5_files, start_time, end_time)
		for file in h5_files:
			timestamp_str = os.path.basename(file).split('_')[2]
			timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
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
