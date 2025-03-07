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
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-05-21':{
				'description'	:	'Test flight 2',
				'available'		:	False,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-05-23':{
				'description'	:	'Transit flight to Pituffik (Spring)',
				'available'		:	False,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-05-24':{
				'description'	:	'Test flight 3',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-05-28':{
				'description'	:	'Research Flight 01',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-05-30':{
				'description'	:	'Research Flight 02',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-05-31':{
				'description'	:	'Research Flight 03',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'12:32:18',
					'time2' :	'18:28:07',
					'second1':	13.5,
					'second2':	13.0,
				},
			},
			'2024-06-03':{
				'description'	:	'Research Flight 04',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-06-05':{
				'description'	:	'Research Flight 05',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'11:04:04',
					'time2' :	'18:47:02',
					'second1':	0.3,
					'second2':	1.6,
				},
			},
			'2024-06-06':{
				'description'	:	'Research Flight 06',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'11:02:59',
					'time2' :	'18:38:09',
					'second1':	0.2,
					'second2':	1.2,
				},
			},
			'2024-06-07':{
				'description'	:	'Research Flight 07',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-06-10':{
				'description'	:	'Research Flight 08',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-06-11':{
				'description'	:	'Research Flight 09',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-06-13':{
				'description'	:	'Research Flight 10',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-07-08':{
				'description'	:	'Ground Test',
				'available'		:	False,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-07-09':{
				'description'	:	'Ground Test',
				'available'		:	False,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-07-22':{
				'description'	:	'Transit flight to Pituffik (Summer)',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-07-25':{
				'description'	:	'Research Flight 11',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-07-29':{
				'description'	:	'Research Flight 12',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-07-30':{
				'description'	:	'Research Flight 13',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-01':{
				'description'	:	'Research Flight 14',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-02':{
				'description'	:	'Research Flight 15',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-07':{
				'description'	:	'Research Flight 16',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-08':{
				'description'	:	'Research Flight 17',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-09':{
				'description'	:	'Research Flight 18',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-15':{
				'description'	:	'Research Flight 19',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
			'2024-08-16':{
				'description'	:	'Transit flight to Bangor (Summer)',
				'available'		:	True,
				'time offset'	:	{
					'day'	:	0.,
					'time1' :	'06:00:00',
					'time2' :	'18:00:00',
					'second1':	0.,
					'second2':	0.,
				},
			},
		}

		self.imgdim = np.array([3096, 2080])

		self.radcal_def_dict = {
			'red'	: {	'c0'  :  1.49e-09,   # 0th order coeff (constant)
					    'c2'  :  9.20e-16,   # 2nd order coeff
					    'c4'  :  6.07e-23,   # 4th order coeff
					    'c6'  :  3.97e-28,   # 6th order coeff
					    'imgc': (1596.11,  934.36),    # center of aperture
					    'snsc': (1675.47,  942.39),    # center of sensitivity
						'maxp':  850.,                 # radius of valid circle around imgc
						'wvlc':  626.8,             }, # center wavelength [nm]
			'green'	: {	'c0'  :  1.73e-09,   # 0th order coeff (constant)
					    'c2'  :  1.02e-15,   # 2nd order coeff
					    'c4'  : -4.33e-22,   # 4th order coeff
					    'c6'  :  7.15e-28,   # 6th order coeff
					    'imgc': (1596.11,  934.36),    # center of aperture
					    'snsc': (1705.83,  930.02),    # center of sensitivity
						'maxp':  850.,                 # radius of valid circle around imgc
						'wvlc':  558.9,             }, # center wavelength [nm]
			'blue'	: {	'c0'  :  2.31e-09,   # 0th order coeff (constant)
					    'c2'  :  1.36e-15,   # 2nd order coeff
					    'c4'  : -5.81e-22,   # 4th order coeff
					    'c6'  :  7.14e-28,   # 6th order coeff
					    'imgc': (1596.11,  934.36),    # center of aperture
					    'snsc': (1677.66,  952.00),    # center of sensitivity
						'maxp':  850.,                 # radius of valid circle around imgc
						'wvlc':  498.5,             }, # center wavelength [nm]
			'type'	: {	'input' :   'count per time'        ,   # type before conversion
					    'output':   'radiance'              },  # type after conversion
			'unit'	: {	'input' :   's-1'                   ,   # type before conversion
					    'output':   'W m-2 nm-1 sr-1'       },  # type after conversion
		}

		self.rad_coef, self.rad_wvlc = self.generate_rad_calib()

		self.geomcal_def_dict = {
			'6th'	: {	'c1'  :  9.0e-02,   # 1st order coeff
					    'c2'  :  2.8e-06,   # 2nd order coeff
					    'c3'  : -2.3e-08,   # 3nd order coeff
					    'c4'  :  5.3e-11,   # 4th order coeff
					    'c5'  : -5.2e-14,   # 5th order coeff
					    'c6'  :  2.3e-17,   # 6th order coeff
					    'imgc': (1594.75,  933.27),    # center of aperture
						'maxp':  850.,              }, # radius of valid circle around imgc
			'quad'	: {	'c1'  :  8.8e-02,   # 1st order coeff
					    'c2'  :  3.1e-06,   # 2nd order coeff
					    'imgc': (1594.75,  933.27),    # center of aperture
						'maxp':  850.,              }, # radius of valid circle around imgc
			'linear': {	'c0'  :  0.09,     # linear coeff
					    'imgc': (1594.75,  933.27),    # center of aperture
						'maxp':  850.,              }, # radius of valid circle around imgc
			'type'	: {	'input' :   'image coordinate'        ,   # type before conversion
					    'output':   'field view angle'        },  # type after conversion
			'unit'	: {	'input' :   'pixel'                   ,   # type before conversion
					    'output':   'deg'                     },  # type after conversion
		}

		# self.radcal_dict_simple = {
		# 	'red'	: {	'c0':       0.03697803089161538   ,   # red channel cal main term
		# 			    'c2':       2.9306376815596668e-08,   # quadratic correction term
		# 			    'lambda': 626.408                 ,   # center wavelength [nm]
		# 			    'f_toa':    1.6687                },  # TOA solar irradiance [W/m2/nm]
		# 	'green'	: {	'c0':       0.0379816678515955    ,   # green channel cal main term
		# 			    'c2':       2.4174566687203538e-08,   # quadratic correction term
		# 			    'lambda': 553.572                 ,   # center wavelength [nm]
		# 			    'f_toa':    1.8476                },  # TOA solar irradiance [W/m2/nm]}
		# 	'blue'	: {	'c0':       0.04097034968881991   ,   # blue channel cal main term
		# 			    'c2':       2.3561354650281456e-08,   # quadratic correction term
		# 			    'lambda': 492.667                 ,   # center wavelength [nm]
		# 			    'f_toa':    1.9084                },  # TOA solar irradiance [W/m2/nm]}
		# 	'type'	: {	'pre':    'count'                 ,   # type before conversion
		# 			    'post':   'radiance'              }   # type after conversion
		# }

		self.flipud = True
		self.fliplr = True
		# self.leftpix  = np.array([2190, 1703])
		# self.rightpix = np.array([1004,  162])
		# self.centerpix = np.int_(0.5*(self.leftpix + self.rightpix))
		# self.headvector = np.array([-(self.rightpix[1] - self.centerpix[1]),
		# 							 (self.rightpix[0] - self.centerpix[0])])
		# self.centerpix  = np.array([1594.75,  933.27])
		self.headvector = np.array([ 770.   , -593.   ]) # derived as a 90 deg perpendicular line to the left-right pixel line


		self.xsph = None
		self.ysph = None
		self.zsph = None

		self.xsph, self.ysph, self.zsph = self.generate_geom_calib()
		
		# self.pit_off =  1.5  # deg
		# self.rol_off =  1.5  # deg
		# self.hed_off =  0.0  # deg rotate clockwise
		self.pit_off =  0.0  # deg
		self.rol_off = -0.1  # deg
		self.hed_off =  0.5  # deg rotate clockwise


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
		
		self.camera_misc    = {	'centerpix'	:	self.centerpix,
								'headvector':	self.headvector,
								'pit_off'	:	self.pit_off,
								'hed_off'	:	self.hed_off,
								'rol_off'	:	self.rol_off,
								'flipud'	:	self.flipud,
								'fliplr'	:	self.fliplr,
								'rad_coef'  :   self.rad_coef,
								'rad_wvlc'  :   self.rad_wvlc,
								'geom_zen'  :   self.zeniths,
								'geom_azi'  :   self.azimuths,}
		
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
	
	def generate_rad_calib(self, method='oblique_6th'):
		xx, yy = np.meshgrid(np.arange(self.imgdim[0]), np.arange(self.imgdim[1]))
		if method == 'oblique_6th':
			radcoef = {}
			wvlcs  = {}
			for key in ['red', 'green', 'blue']:
				ps_ = self.radcal_def_dict[key]['imgc']
				pc_ = self.radcal_def_dict[key]['snsc']
				r_  = self.radcal_def_dict[key]['maxp']
				c0  = self.radcal_def_dict[key]['c0']
				c2  = self.radcal_def_dict[key]['c2']
				c4  = self.radcal_def_dict[key]['c4']
				c6  = self.radcal_def_dict[key]['c6']
				xtopc = xx - pc_[0]
				ytopc = yy - pc_[1]
				xtops = xx - ps_[0]
				ytops = yy - ps_[1]
				a_ = ((pc_[0] - ps_[0])**2. + (pc_[1] - ps_[1])**2. - r_*r_)/(r_*r_)
				b_ = ((pc_[0] - ps_[0])*xtopc + (pc_[1] - ps_[1])*ytopc)/r_
				c_ = xtopc*xtopc + ytopc*ytopc
				d_ = (-b_ - np.sqrt(b_*b_ - a_*c_))/a_
				valid = xtops*xtops + ytops*ytops < r_*r_ # limit the valid region to the circle
				d_[~valid] = np.nan
				radcoef[key] = np.zeros_like(xx, dtype=float)
				radcoef[key] = c0 + c2*d_**2. + c4*d_**4. + c6*d_**6.
				wvlc = self.radcal_def_dict[key]['wvlc']
				wvlcs[key] = wvlc
			radcoef['type'] = self.radcal_def_dict['type']
			radcoef['unit'] = self.radcal_def_dict['unit']
		else:
			message = 'Error [Meta_arcsix.generate_rad_calib]: method=%s is not supported.' % (method)
			raise OSError(message)
		return radcoef, wvlcs
	
	def generate_geom_calib(self, method='6th'):
		if method not in ['6th', 'quad', 'linear']:
			message = 'Error [Meta_arcsix.generate_geom_calib]: method=%s is not supported.' % (method)
			raise OSError(message)
		xx, yy = np.meshgrid(np.arange(self.imgdim[0]), np.arange(self.imgdim[1]))
		img2d_zero = np.zeros_like(xx, dtype=float)
		if method == '6th':
			c1  = self.geomcal_def_dict[method]['c1']
			c2  = self.geomcal_def_dict[method]['c2']
			c3  = self.geomcal_def_dict[method]['c3']
			c4  = self.geomcal_def_dict[method]['c4']
			c5  = self.geomcal_def_dict[method]['c5']
			c6  = self.geomcal_def_dict[method]['c6']
			xtoc = xx - self.geomcal_def_dict[method]['imgc'][0]
			ytoc = yy - self.geomcal_def_dict[method]['imgc'][1]
			r    = np.sqrt(xtoc*xtoc + ytoc*ytoc)
			self.valid_geom = r < self.geomcal_def_dict[method]['maxp']
			self.zeniths  = np.ma.masked_where(~self.valid_geom, img2d_zero)
			self.zeniths  = np.deg2rad(self.zeniths + c1*r + c2*r*r + c3*r*r*r + c4*r*r*r*r + c5*r*r*r*r*r + c6*r*r*r*r*r*r)
		elif method == 'quad':
			c1  = self.geomcal_def_dict[method]['c1']
			c2  = self.geomcal_def_dict[method]['c2']
			xtoc = xx - self.geomcal_def_dict[method]['imgc'][0]
			ytoc = yy - self.geomcal_def_dict[method]['imgc'][1]
			r    = np.sqrt(xtoc*xtoc + ytoc*ytoc)
			self.valid_geom = r < self.geomcal_def_dict[method]['maxp']
			self.zeniths  = np.ma.masked_where(~self.valid_geom, img2d_zero)
			self.zeniths  = np.deg2rad(self.zeniths + c1*r + c2*r*r)
		elif method == 'linear':
			c0  = self.geomcal_def_dict[method]['c0']
			xtoc = xx - self.geomcal_def_dict[method]['imgc'][0]
			ytoc = yy - self.geomcal_def_dict[method]['imgc'][1]
			r    = np.sqrt(xtoc*xtoc + ytoc*ytoc)
			self.valid_geom = r < self.geomcal_def_dict[method]['maxp']
			self.zeniths  = np.ma.masked_where(~self.valid_geom, img2d_zero)
			self.zeniths  = np.deg2rad(self.zeniths + c0*r)
		self.azimuths = np.ma.masked_where(~self.valid_geom, img2d_zero)
		head_ang = np.arctan2(self.headvector[0], self.headvector[1])
		self.azimuths = self.azimuths + np.arctan2(xx - self.geomcal_def_dict[method]['imgc'][0], yy - self.geomcal_def_dict[method]['imgc'][1]) - head_ang
		self.azimuths = self.azimuths % (2.*np.pi)
		self.xsph = np.sin(self.zeniths)*np.cos(self.azimuths)*(1. if self.fliplr else -1.)
		self.ysph = np.sin(self.zeniths)*np.sin(self.azimuths)*(1. if self.flipud else -1.)
		self.zsph = np.cos(self.zeniths)
		self.centerpix = self.geomcal_def_dict[method]['imgc']
		return self.xsph, self.ysph, self.zsph
	
	def load_fits(self, start_time, end_time, location='ARCSIX1KSS'):
		self.fits_allfiles = self.locate_allfits(location=location)

		st_dt = datetime.datetime.strptime(start_time, '%H:%M:%S')
		# if campaign.lower() == 'camp2ex' and st_dt.hour < 15: st_dt += datetime.timedelta(days=1)
		en_dt = datetime.datetime.strptime(end_time, '%H:%M:%S')
		# if campaign.lower() == 'camp2ex' and en_dt.hour < 15: en_dt += datetime.timedelta(days=1)

		# self.t_offset = datetime.timedelta(	days=self.flight_meta['toff_day'], 
		# 									seconds=self.flight_meta['toff_sec'])
		self.to_t1 = datetime.datetime.strptime(self.flight_meta['time offset']['time1'], '%H:%M:%S')
		self.to_t2 = datetime.datetime.strptime(self.flight_meta['time offset']['time2'], '%H:%M:%S')
		self.to_s1 = self.flight_meta['time offset']['second1']
		self.to_s2 = self.flight_meta['time offset']['second2']
		self.to_d  = self.flight_meta['time offset']['day']

		fits_list = []
		lat_list, lon_list = [], []
		for ifits, fits_fn in enumerate(self.fits_allfiles):
			tt_fn = datetime.datetime.strptime(fits_fn[-14:-6], '%H_%M_%S')
			# if campaign.lower() == 'camp2ex' and tt_fn.hour < 15: tt_fn += datetime.timedelta(days=1)
			if st_dt <= tt_fn <= en_dt:
				fits_list.append(fits_fn)
				t_fn  = datetime.datetime.strptime(self.date + ' ' + fits_fn[-14:-6], '%Y-%m-%d %H_%M_%S')
				to_s = self.to_s1 + (tt_fn - self.to_t1).total_seconds() * (self.to_s2 - self.to_s1) / (self.to_t2 - self.to_t1).total_seconds()
				# print('Time offset: %s' % to_s)
				t_offset = datetime.timedelta(days=self.to_d, seconds=to_s)
				hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + t_offset.total_seconds()/3600.
				# hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + self.t_offset.total_seconds()/3600.
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
	
	def rad_and_geom_from_fits(self, fits_filename, flipud=None, fliplr=None, mask_fits_filename=None, mask_aircraft_shadow=True):
		image, header = self.radiance_from_fits(fits_filename, flipud=flipud, fliplr=fliplr, mask_fits_filename=mask_fits_filename)
		t_act, aircraft_status = self.interpolate_hsk_for_fits(header['DATE-OBS'])
		vza, vaa = self.calc_viewing_geometry(aircraft_status['rol'], aircraft_status['pit'], aircraft_status['hed'])
		image['geometry'] = {'vza': vza, 'vaa': vaa, 'centerpix': self.centerpix, 'type' : 'viewing geometry', 'unit' : 'radian'}
		image['t_act'] = t_act
		image['aircraft_status'] = aircraft_status
		if mask_aircraft_shadow:
			image = self.aircraft_shadow_mask(image)
		return image, t_act

	# Read image file (Fits file format) and convert to radiance
	def radiance_from_fits(self, fits_filename, flipud=None, fliplr=None, mask_fits_filename=None):
		if flipud is None:
			flipud = self.flipud
		if fliplr is None:
			fliplr = self.fliplr

		if not os.path.exists(fits_filename):
			message = 'Error: fits file {} not found.'.format(fits_filename)
			raise OSError(message)
		print('Reading {}'.format(fits_filename))
		handle = fits.open(fits_filename)
		
		fimg = self.reshapeFITS(handle[0].data)
		if flipud:
			fimg = np.flipud(fimg)
		if fliplr:
			fimg = np.fliplr(fimg)
		
		fheader = handle[0].header
		
		if mask_fits_filename is not None:
			if not os.path.exists(mask_fits_filename):
				message = 'Error: fits file {} not found.'.format(mask_fits_filename)
				raise OSError(message)
			print('Reading {}'.format(mask_fits_filename))
			handle_msk = fits.open(mask_fits_filename)
			
			fmsk = self.reshapeFITS(handle_msk[0].data)
			if flipud:
				fmsk = np.flipud(fmsk)
			if fliplr:
				fmsk = np.fliplr(fmsk)
			fheader_msk = handle_msk[0].header

			fimg[fmsk > 0.] = np.nan
		
		img = {'data': fimg, 'shape': fimg.shape, 'type': 'count', 'unit': 'unitless', 'wavelength': None}

		img_rad = self.rad_conversion(img, fheader['EXPTIME'], radcal_dict=self.rad_coef, wvlc_list=self.rad_wvlc, saturation_val=0.95*(2**fheader['BITPIX']))
		
		return img_rad, fheader
	
		# Reshape the array read from the fits file
	def reshapeFITS(self, img):
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

	def rad_conversion(self, img, exptime, radcal_dict=None, wvlc_list=None, saturation_val=None):
		if radcal_dict is None:
			radcal_dict = self.rad_coef
		if wvlc_list is None:
			wvlc_list = self.rad_wvlc
		print('Radiometric calibration coefficients are being applied...')
		if radcal_dict['type']['input'].lower() == 'count per time' and img['type'].lower() == 'count':
			if saturation_val is not None:
				img['data'][img['data'] > saturation_val] = np.nan
			img['data'][:, :, 0] =  img['data'][:, :, 0] * radcal_dict['red'][:, :]   / exptime
			img['data'][:, :, 1] =  img['data'][:, :, 1] * radcal_dict['green'][:, :] / exptime
			img['data'][:, :, 2] =  img['data'][:, :, 2] * radcal_dict['blue'][:, :]  / exptime
			img['type'] = 'radiance'
			img['unit'] = 'W m^(-2) nm^(-1) sr^(-1)'
		else:
			print(' !!! Image and calibration data types do not match! <%s> vs <%s> Skipping... ' \
						% (radcal_dict['type']['input'].lower()))
		img['wavelength'] = np.array([wvlc_list['red'], wvlc_list['green'], wvlc_list['blue']])
		return img
	
	def calc_viewing_geometry(self, rol, pit, hed): # in degrees
		R_n2c = self.R_NED2Cam(rol*np.pi/180., pit*np.pi/180., hed*np.pi/180.)
		xcar, ycar, zcar = np.einsum('ij,jkl->ikl', R_n2c.T, np.stack([self.xsph, self.ysph, self.zsph]))
		R_n2c = self.R_NED2Cam(self.pit_off*np.pi/180., self.rol_off*np.pi/180., self.hed_off*np.pi/180.)
		xcar, ycar, zcar = np.einsum('ij,jkl->ikl', R_n2c.T, np.stack([xcar, ycar, zcar]))
		xcar = np.ma.masked_where(~self.valid_geom, xcar)
		ycar = np.ma.masked_where(~self.valid_geom, ycar)
		zcar = np.ma.masked_where(~self.valid_geom, zcar)
		vza  = np.arccos(zcar)
		vaa  = np.arctan2(ycar, xcar)
		vza = vza.filled(np.nan)
		vaa = vaa.filled(np.nan)
		return vza, vaa
	
	def aircraft_shadow_mask(self, image):
		def dist(a1, a2):
			dif = (a1 - a2) % (2.*np.pi)
			return np.minimum(dif, 2.*np.pi - dif)
		aircraft_length = 36. # m
		vza = image['geometry']['vza'] # rad
		vaa = image['geometry']['vaa'] # rad
		status = image['aircraft_status']
		alt = status['alt']
		sza = np.deg2rad(status['sza'])
		saa = np.deg2rad(status['saa'])
		dvza = aircraft_length/alt*np.cos(sza)*np.cos(sza)
		dvaa = aircraft_length/alt*np.cos(sza)
		target_vza = sza
		target_vaa = saa + np.pi
		in_shadow = (vza - target_vza)**2./(dvza)**2. + dist(vaa, target_vaa)**2./(dvaa)**2. < 1
		image['data'][in_shadow] = np.nan
		return image

	def R_NED2Cam(self, roll, pitch, yaw): #roll, pitch, yaw [radian]
		cr = np.cos(roll)
		sr = np.sin(roll)
		cp = np.cos(pitch)
		sp = np.sin(pitch)
		cy = np.cos(yaw)
		sy = np.sin(yaw)
		Rr = np.matrix([[ 1.,  0.,  0.],
						[ 0.,  cr,  sr],
						[ 0., -sr,  cr]])
		Rp = np.matrix([[ cp,  0., -sp],
						[ 0.,  1.,  0.],
						[ sp,  0.,  cp]])
		Ry = np.matrix([[ cy,  sy,  0.],
						[-sy,  cy,  0.],
						[ 0.,  0.,  1.]])
		R = np.matmul(Rr, np.matmul(Rp, Ry))
		return R
	
	# def interpolate_hsk_for_fits(self, fits_fn):
		# t_fn  = datetime.datetime.strptime(self.date + ' ' + fits_fn[-14:-6], '%Y-%m-%d %H_%M_%S')
	def interpolate_hsk_for_fits(self, date_obs_str):
		t_fn  = datetime.datetime.strptime(self.date + ' ' + date_obs_str.split('T')[1][:-1], '%Y-%m-%d %H:%M:%S.%f')
		tt_fn = datetime.datetime.strptime(date_obs_str.split('T')[1][:-1], '%H:%M:%S.%f')
		to_s = self.to_s1 + (tt_fn - self.to_t1).total_seconds() * (self.to_s2 - self.to_s1) / (self.to_t2 - self.to_t1).total_seconds()
		# print('Time offset: %s' % to_s)
		t_offset = datetime.timedelta(days=self.to_d, seconds=to_s)
		t_act = t_fn + t_offset
		hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + t_offset.total_seconds()/3600.

		lat  = np.interp(hrs, self.hrss, self.lats) # interpolate to find values for the time at which the image was taken
		lon  = np.interp(hrs, self.hrss, self.lons)
		alt  = np.interp(hrs, self.hrss, self.alts)
		pit  = np.interp(hrs, self.hrss, self.pits)
		rol  = np.interp(hrs, self.hrss, self.rols)
		hed  = np.interp(hrs, self.hrss, self.heds)

		when = t_act.replace(tzinfo=datetime.timezone.utc)
		sza  = 90. - pysolar.solar.get_altitude(lat, lon, when)
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
