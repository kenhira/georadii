import os
import sys
import numpy as np
import datetime
import glob
import h5py
import pysolar
from astropy.io import fits
from pyproj import Geod as geod
import multiprocessing as mp
# import concurrent.futures

class Georadii:
	def __init__(	self,
					input_array,
					input_type=None,
					input_coordinate=None,
					input_meta=None,
					mode='semiauto'):
		self.input_array       = input_array
		self.input_type        = input_type
		self.input_coordinate  = input_coordinate
		self.input_meta        = input_meta
		self.mode              = mode

		self.camera_data  = None
		self.latlon_data  = None
		self.gridded_data = None

		if self.input_type is None:
			message = 'Error [Georadii]: input_type needs to be specified.'
			raise OSError(message)
		
		elif self.input_type == 'camera':

			if self.input_coordinate is None:
				message = 'Error [Georadii]: input_type needs to be specified.'
				raise OSError(message)

			elif self.input_coordinate == 'camera':
				self.camera_data = self.AllSkyCamera(self.input_array, self.input_meta)
				if self.mode == 'auto':
					self.geolocate()
					self.gridded()
				elif self.mode == 'semiauto':
					self.geolocate()
				elif self.mode == 'manual':
					pass

			elif self.input_coordinate == 'latlon':
				self.latlon_data = self.LatLon(self.input_array, self.input_meta)
				if self.mode == 'auto':
					self.gridded()
				elif self.mode == 'manual':
					pass

			else:
				message = 'Error [Georadii]: input_coordinate <%s> is not compatible with <%s>.' \
							% (self.input_coordinate, self.input_type)
				raise OSError(message)
			
		elif self.input_type.lower() == 'aviris':

			if self.input_coordinate is None:
				message = 'Error [Georadii]: input_type needs to be specified.'
				raise OSError(message)

			elif self.input_coordinate == 'latlon':
				self.latlon_data = self.LatLon(self.input_array, self.input_meta)
				if self.mode == 'auto':
					self.gridded()
				elif self.mode == 'manual':
					pass

			else:
				message = 'Error [Georadii]: input_coordinate <%s> is not compatible with <%s>.' \
							% (self.input_coordinate, self.input_type)
				raise OSError(message)

		elif self.input_type.lower() == 'rsp':

			message = 'Error [Georadii]: RSP data is not yet supported.'
			raise OSError(message)

		else:
			message = 'Error [Georadii]: input_type <%s> is unknown.' % self.input_type
			raise OSError(message)
	
	def geolocate(self):
		cam_class = self.camera_data
		cam_class.hemispheric_to_cartesian()
		cam_class.cartesian_to_latlon()
		latlon_meta = cam_class.cam_meta
		latlon_meta['longeo'] = cam_class.longeo
		latlon_meta['latgeo'] = cam_class.latgeo
		latlon_meta['valid_domain'] = cam_class.valid_domain
		self.latlon_data = self.LatLon(cam_class.img, latlon_meta)
		return latlon_meta['longeo'], latlon_meta['latgeo']
		
	def gridded(self, gridmeta=None, enable_mp=False): # Note: multiprocessing is currently slow
		latlon_class = self.latlon_data
		if gridmeta is None:
			grid_meta = {}
			# grid_meta['transform'] = {	'active'      : False, # Rotate the lat/lon coordinate system so that the grids are more regular
			# 							'center'      : (0.5*(np.nanmin(latlon_class.longeo) + np.nanmax(latlon_class.longeo)),
			# 											 0.5*(np.nanmin(latlon_class.latgeo) + np.nanmax(latlon_class.latgeo))),
			# 							'inclination' : 0. }
			# grid_meta['x'] = {	'min' : np.nanmin(latlon_class.longeo),
			# 					'max' : np.nanmax(latlon_class.longeo),
			# 					'incr': (np.nanmax(latlon_class.longeo) - np.nanmin(latlon_class.longeo)) / 160.} 
			# grid_meta['y']  = {	'min' : np.nanmin(latlon_class.latgeo),
			# 					'max' : np.nanmax(latlon_class.latgeo),
			# 					'incr': (np.nanmax(latlon_class.latgeo) - np.nanmin(latlon_class.latgeo)) / 160.}
			grid_meta['transform'] = {	'active'      : True, # Rotate the lat/lon coordinate system so that the grids are more regular
										'center'      : (0.5*(np.nanmin(latlon_class.longeo) + np.nanmax(latlon_class.longeo)),
														 0.5*(np.nanmin(latlon_class.latgeo) + np.nanmax(latlon_class.latgeo))),
										'inclination' : 0. }
			# f_extend = 5.
			# grid_meta['x'] = {	'min' :  0.5*f_extend*np.nanmin(latlon_class.longeo) - 0.5*f_extend*np.nanmax(latlon_class.longeo),
			# 					'max' : -0.5*f_extend*np.nanmin(latlon_class.longeo) + 0.5*f_extend*np.nanmax(latlon_class.longeo),
			# 					'incr': (np.nanmax(latlon_class.longeo) - np.nanmin(latlon_class.longeo)) / 250.} 
			# grid_meta['y']  = {	'min' :  0.5*f_extend*np.nanmin(latlon_class.latgeo) - 0.5*f_extend*np.nanmax(latlon_class.latgeo),
			# 					'max' : -0.5*f_extend*np.nanmin(latlon_class.latgeo) + 0.5*f_extend*np.nanmax(latlon_class.latgeo),
			# 					'incr': (np.nanmax(latlon_class.latgeo) - np.nanmin(latlon_class.latgeo)) / 250.}
		else:
			grid_meta = gridmeta
		# define the grid system
		if grid_meta['transform']['active']: # Rotate the lat/lon coordinate system so that the grids are more regular
			lon_center, lat_center = grid_meta['transform']['center']
			inclination = grid_meta['transform']['inclination']
			lat2geo, lon2geo = self.transform_coordinates(latlon_class.latgeo, latlon_class.longeo, lat_center, lon_center, inclination)
			if gridmeta is None:
				lonmin, lonmax, dlon = np.nanmin(lon2geo), np.nanmax(lon2geo), (np.nanmax(lon2geo) - np.nanmin(lon2geo))/250.
				latmin, latmax, dlat = np.nanmin(lat2geo), np.nanmax(lat2geo), (np.nanmax(lat2geo) - np.nanmin(lat2geo))/250.
			else:
				lonmin, lonmax, dlon = grid_meta['x']['min'], grid_meta['x']['max'], grid_meta['x']['incr']
				latmin, latmax, dlat = grid_meta['y']['min'], grid_meta['y']['max'], grid_meta['y']['incr']
			lon2_arr = np.arange(lonmin, lonmax + dlon, dlon)
			lat2_arr = np.arange(latmin, latmax + dlat, dlat)
			lon2_xx, lat2_yy = np.meshgrid(lon2_arr, lat2_arr)
			lat_yy, lon_xx = self.inverse_transform_coordinates(lat2_yy, lon2_xx, lat_center, lon_center, inclination)
			longidx = np.int_((lon2geo - lonmin + 0.5*dlon)//dlon % len(lon2_arr))
			latgidx = np.int_((lat2geo - latmin + 0.5*dlat)//dlat % len(lat2_arr))

		else:
			lonmin, lonmax, dlon = grid_meta['x']['min'], grid_meta['x']['max'], grid_meta['x']['incr']
			latmin, latmax, dlat = grid_meta['y']['min'], grid_meta['y']['max'], grid_meta['y']['incr']
			lon_arr = np.arange(lonmin, lonmax + dlon, dlon)
			lat_arr = np.arange(latmin, latmax + dlat, dlat)
			lon_xx, lat_yy = np.meshgrid(lon_arr, lat_arr)
			longidx = np.int_((latlon_class.longeo - lonmin + 0.5*dlon)//dlon % len(lon_arr))
			latgidx = np.int_((latlon_class.latgeo - latmin + 0.5*dlat)//dlat % len(lat_arr))
		
		# dimension of the gridded map
		grid_dim = (lon_xx.shape[0], lon_xx.shape[1], latlon_class.img['data'].shape[2])
		if enable_mp:
			nproc = 8
			if latlon_class.valid_domain is not None:
				x1, x2 = np.meshgrid(np.arange(latgidx.shape[1]), np.arange(latgidx.shape[0]))
				x1_ch = np.array_split(x1[latlon_class.valid_domain], nproc, axis=0)
				x2_ch = np.array_split(x2[latlon_class.valid_domain], nproc, axis=0)
				args = [(x1_ch[ich], x2_ch[ich], latgidx, longidx, latlon_class.img['data'], grid_dim) for ich in range(nproc)]
			else:
				x1, x2 = np.meshgrid(np.arange(latgidx.shape[1]), np.arange(latgidx.shape[0]))
				x1_ch = np.array_split(x1, nproc, axis=0)
				x2_ch = np.array_split(x2, nproc, axis=0)
				args = [(x1_ch[ich], x2_ch[ich], latgidx, longidx, latlon_class.img['data'], grid_dim) for ich in range(nproc)]
			with mp.Pool(processes=nproc) as pool:
				results = pool.map(self.count_and_sum, args)
			count_list, value_list = zip(*results)
			count_stack = np.stack(count_list, axis=-1)
			value_stack = np.stack(value_list, axis=-1)
			ncount, imgsum = np.sum(count_stack, axis=2), np.sum(value_stack, axis=3)
		else:
			x1, x2 = np.meshgrid(np.arange(longidx.shape[1]), np.arange(longidx.shape[0]))
			if latlon_class.valid_domain is not None:
				ncount, imgsum = self.count_and_sum((x1[np.where(latlon_class.valid_domain)], x2[np.where(latlon_class.valid_domain)], 
												latgidx, longidx, latlon_class.img['data'], grid_dim))
			else:
				ncount, imgsum = self.count_and_sum((x1, x2, latgidx, longidx, latlon_class.img['data'], grid_dim))

		imgout = np.zeros_like(imgsum)
		for ich in range(latlon_class.img['data'].shape[2]):
			imgout[:, :, ich] = imgsum[:, :, ich]/np.float64(ncount)                             # divide by the number of pixels to average the image values
		return lon_xx, lat_yy, imgout, ncount
	
	def count_and_sum(self, arg):
		x1, x2, ap, aq, av, dim_grid = arg
		ndimg1, ndimg2, ndimg3 = dim_grid
		cnt = np.zeros((ndimg1, ndimg2))
		val = np.zeros((ndimg1, ndimg2, ndimg3))
		for i1, i2 in zip(x1.flatten(), x2.flatten()):
			cnt[ap[i2, i1], aq[i2, i1]] += 1
			val[ap[i2, i1], aq[i2, i1], :] += av[i2, i1, :]
		return cnt, val


	def plot(self):
		if self.gridded_data is not None:
			pass
		elif self.latlon_data is not None:
			pass
		elif self.camera_data is not None:
			pass
		else:
			print('There is nothing to be plotted...')
	
	def transform_coordinates(self, lat_in, lon_in, lat_piv, lon_piv, inclination):
		# Convert lat_in, lon_in to Cartesian coordinates
		lat_in_rad = np.radians(lat_in)
		lon_in_rad = np.radians(lon_in)
		x_1 = np.cos(lat_in_rad) * np.cos(lon_in_rad)
		y_1 = np.cos(lat_in_rad) * np.sin(lon_in_rad)
		z_1 = np.sin(lat_in_rad)
		points = np.array([x_1, y_1, z_1])

		# Rotate around z-axis to align lon_piv with the prime meridian
		lon_piv_rad = np.radians(lon_piv)
		rotation_matrix_z = np.array([
			[np.cos(-lon_piv_rad), -np.sin(-lon_piv_rad), 0],
			[np.sin(-lon_piv_rad),  np.cos(-lon_piv_rad), 0],
			[0, 0, 1]
		])
		points = np.einsum('ij,jk...->ik...', rotation_matrix_z, points)

		# Rotate around y-axis to bring lat_piv to the equator
		lat_piv_rad = np.radians(lat_piv)
		rotation_matrix_y = np.array([
			[np.cos(-lat_piv_rad), 0, -np.sin(-lat_piv_rad)],
			[0, 1, 0],
			[np.sin(-lat_piv_rad), 0, np.cos(-lat_piv_rad)]
		])
		points = np.einsum('ij,jk...->ik...', rotation_matrix_y, points)

		# Apply inclination by rotating around x-axis
		inclination_rad = np.radians(inclination)
		rotation_matrix_x = np.array([
			[1, 0, 0],
			[0, np.cos(inclination_rad), -np.sin(inclination_rad)],
			[0, np.sin(inclination_rad), np.cos(inclination_rad)]
		])
		points = np.einsum('ij,jk...->ik...', rotation_matrix_x, points)

		# Convert the new Cartesian coordinates back to lat/lon
		new_lon = np.arctan2(points[1], points[0])
		hyp = np.sqrt(points[0]**2 + points[1]**2)
		new_lat = np.arctan2(points[2], hyp)

		# Convert radians to degrees
		new_lat = np.degrees(new_lat)
		new_lon = (np.degrees(new_lon) + 180.) % 360. - 180.

		return new_lat, new_lon


	def inverse_transform_coordinates(self, lat_in, lon_in, lat_piv, lon_piv, inclination):
		# Convert lat_in, lon_in to Cartesian coordinates
		lat_in_rad = np.radians(lat_in)
		lon_in_rad = np.radians(lon_in)
		x_new = np.cos(lat_in_rad) * np.cos(lon_in_rad)
		y_new = np.cos(lat_in_rad) * np.sin(lon_in_rad)
		z_new = np.sin(lat_in_rad)
		points = np.array([x_new, y_new, z_new])

		# Reverse rotate around x-axis (negative of inclination angle)
		inclination_rad = np.radians(inclination)
		rotation_matrix_x = np.array([
			[1, 0, 0],
			[0, np.cos(-inclination_rad), -np.sin(-inclination_rad)],
			[0, np.sin(-inclination_rad), np.cos(-inclination_rad)]
		])
		points = np.einsum('ij,jk...->ik...', rotation_matrix_x, points)

		# Reverse rotate around y-axis (positive of lat_piv)
		lat_piv_rad = np.radians(lat_piv)
		rotation_matrix_y = np.array([
			[np.cos(lat_piv_rad), 0, -np.sin(lat_piv_rad)],
			[0, 1, 0],
			[np.sin(lat_piv_rad), 0, np.cos(lat_piv_rad)]
		])
		points = np.einsum('ij,jk...->ik...', rotation_matrix_y, points)

		# Reverse rotate around z-axis (positive of lon_piv)
		lon_piv_rad = np.radians(lon_piv)
		rotation_matrix_z = np.array([
			[np.cos(lon_piv_rad), -np.sin(lon_piv_rad), 0],
			[np.sin(lon_piv_rad),  np.cos(lon_piv_rad), 0],
			[0, 0, 1]
		])
		points = np.einsum('ij,jk...->ik...', rotation_matrix_z, points)

		# Convert the final Cartesian coordinates back to original lat/lon
		lon_out = np.arctan2(points[1], points[0])
		hyp = np.sqrt(points[0]**2 + points[1]**2)
		lat_out = np.arctan2(points[2], hyp)

		# Convert radians to degrees
		lat_out = np.degrees(lat_out)
		lon_out = np.degrees(lon_out)

		return lat_out, lon_out

	class LatLon:
		def __init__(self, img, latlon_meta):
			self.img = img
			self.latlon_meta = latlon_meta

			self.longeo = self.latlon_meta['longeo']
			self.latgeo = self.latlon_meta['latgeo']

			self.valid_domain = self.latlon_meta.get('valid_domain', None)




	class AllSkyCamera:
		def __init__(self, image, cam_meta): #img, centerpix, headvector, fov=65., degperpix=0.09):

			self.cam_meta = cam_meta

			_fov       = 65.    # deg
			_degperpix =  0.09  # deg/pixel
			_degperpix2 = 0.00  # deg/pixel
			_radcaldic = {}
			_radcaldic['red']   = {	'c0':       0.03697803089161538   ,   # red channel cal main term
									'c2':       2.9306376815596668e-08,   # quadratic correction term
									'lambda': 626.408                 ,   # center wavelength [nm]
									'f_toa':    1.6687                }   # TOA solar irradiance [W/m2/nm]
			_radcaldic['green'] = {	'c0':       0.0379816678515955    ,   # green channel cal main term
									'c2':       2.4174566687203538e-08,   # quadratic correction term
									'lambda': 553.572                 ,   # center wavelength [nm]
									'f_toa':    1.8476                }   # TOA solar irradiance [W/m2/nm]
			_radcaldic['blue']  = {	'c0':       0.04097034968881991   ,   # blue channel cal main term
									'c2':       2.3561354650281456e-08,   # quadratic correction term
									'lambda': 492.667                 ,   # center wavelength [nm]
									'f_toa':    1.9084                }   # TOA solar irradiance [W/m2/nm]
			_radcaldic['type'] = {	'pre':  'raw count'                ,   # type before conversion
									'post': 'radiance'                 }   # type after conversion
			_radcaldic['exposure'] = {'time':  0.0                   }   # exposure time [sec?]

			if 'type' not in self.cam_meta:
				message = 'Error [Georadii.AllSkyCamera]: Key <type> is not found in <cam_meta>.'
				raise OSError(message)
			if 'unit' not in self.cam_meta:
				message = 'Error [Georadii.AllSkyCamera]: Key <unit> is not found in <cam_meta>.'
				raise OSError(message)
			if 'centerpix' not in self.cam_meta:
				message = 'Error [Georadii.AllSkyCamera]: Key <centerpix> is not found in <cam_meta>.'
				raise OSError(message)
			if 'headvector' not in self.cam_meta:
				message = 'Error [Georadii.AllSkyCamera]: Key <headvector> is not found in <cam_meta>.'
				raise OSError(message)

			self.zenith_limit = self.cam_meta.get('fov', _fov)             # deg
			self.degperpix    = self.cam_meta.get('degperpix',  _degperpix) # deg/pixel
			self.degperpix2   = self.cam_meta.get('degperpix2', _degperpix2) # deg/pixel
			img = {'data': image, 'type': self.cam_meta['type'], 'unit': self.cam_meta['unit']}
			self.load_image(img, self.cam_meta['centerpix'], self.cam_meta['headvector'])
			self.radcal       = self.cam_meta.get('radcal', False)
			if self.radcal:
				self.radcaldic    = self.cam_meta.get('radcaldic', _radcaldic)
				self.rad_calibration(radcal_dict=self.radcaldic, exptime=self.cam_meta['exptime'])


		def load_image(self, img, centerpix, headvector):
			self.img = img
			self.img2d_zero = np.zeros_like(self.img['data'][:, :, 0])
			self.xa, self.ya = np.arange(self.img['data'].shape[1]), np.arange(self.img['data'].shape[0])
			self.xx, self.yy = np.meshgrid(self.xa, self.ya)
			# self.r_incl = np.int_(self.zenith_limit/self.degperpix)
			if np.abs(self.degperpix2) < 1e-8:
				self.r_incl = np.int_(self.zenith_limit/self.degperpix)
			else:
				self.r_incl = np.int_((-self.degperpix + np.sqrt(self.degperpix**2. + 4.*self.degperpix2*self.zenith_limit))/(2.*self.degperpix2))
			self.r_2    = (self.xx - centerpix[0])*(self.xx - centerpix[0]) + (self.yy - centerpix[1])*(self.yy - centerpix[1])
			self.r_     = np.sqrt(self.r_2)
			self.valid_domain = (self.r_ < self.r_incl) & (~np.isnan(self.img['data'][:, :, 0]))
			self.zeniths  = np.ma.masked_where(self.r_ > self.r_incl, self.img2d_zero)
			self.azimuths = np.ma.masked_where(self.r_ > self.r_incl, self.img2d_zero)
			# self.zeniths  = self.zeniths  + self.r_*self.degperpix*np.pi/180.
			self.zeniths  = self.zeniths  + (self.r_**2.*self.degperpix2 + self.r_*self.degperpix)*np.pi/180.
			self.head_ang = np.arctan2(headvector[0], headvector[1])
			self.azimuths = (self.azimuths + np.arctan2(self.xx - centerpix[0], self.yy - centerpix[1]) - self.head_ang) % (2.*np.pi)

		# Calculate Direct-Cosine-Matrix to convert NED(North-East-Down) coordinate to
		#  camera coordinate
		# Usage: d(Cam) = R*d(NED)
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

		def hemispheric_to_cartesian(self): #rol, pit, hed, alt, rol_off=0.0, pit_off=0.0, hed_off=0.0, flipud=True, fliplr=True):
			rol     = self.cam_meta['rol']
			pit     = self.cam_meta['pit']
			hed     = self.cam_meta['hed']
			alt     = self.cam_meta['alt']
			rol_off = self.cam_meta['rol_off']
			pit_off = self.cam_meta['pit_off']
			hed_off = self.cam_meta['hed_off']
			flipud  = self.cam_meta['flipud']
			fliplr  = self.cam_meta['fliplr']
			self.xsph = np.sin(self.zeniths)*np.cos(self.azimuths)*(1. if fliplr else -1.)
			self.ysph = np.sin(self.zeniths)*np.sin(self.azimuths)*(1. if flipud else -1.)
			self.zsph = np.cos(self.zeniths)
			# self.R_n2c = np.matmul(self.R_NED2Cam(rol_off*np.pi/180., pit_off*np.pi/180., hed_off*np.pi/180.), self.R_NED2Cam(rol*np.pi/180., pit*np.pi/180., hed*np.pi/180.))
			self.R_n2c = self.R_NED2Cam(rol*np.pi/180., pit*np.pi/180., hed*np.pi/180.)
			self.xcar, self.ycar, self.zcar = np.einsum('ij,jkl->ikl', self.R_n2c.T, np.stack([self.xsph, self.ysph, self.zsph]))
			self.R_n2c = self.R_NED2Cam(pit_off*np.pi/180., rol_off*np.pi/180., hed_off*np.pi/180.)
			self.xcar, self.ycar, self.zcar = np.einsum('ij,jkl->ikl', self.R_n2c.T, np.stack([self.xcar, self.ycar, self.zcar]))
			self.xcar = np.ma.masked_where(self.r_ > self.r_incl, self.xcar)
			self.ycar = np.ma.masked_where(self.r_ > self.r_incl, self.ycar)
			self.zcar = np.ma.masked_where(self.r_ > self.r_incl, self.zcar)
			#self.vza  = np.arccos(self.zcar/np.sqrt(self.xcar**2. + self.ycar**2. + self.zcar**2.))
			self.vza  = np.arccos(self.zcar)
			self.vaa  = np.arctan2(self.ycar, self.xcar)
			self.vza = np.ma.masked_where(self.r_ > self.r_incl, self.vza)
			self.vaa = np.ma.masked_where(self.r_ > self.r_incl, self.vaa)	
			self.nflat = self.xcar*alt/self.zcar
			self.eflat = self.ycar*alt/self.zcar
			return self.vza, self.vaa

		def cartesian_to_latlon(self): #lat, lon):
			lat = self.cam_meta['lat']
			lon = self.cam_meta['lon']
			g = geod(ellps='WGS84')
			self.dist    = np.sqrt(self.nflat*self.nflat + self.eflat*self.eflat)
			self.ang_deg = np.arctan2(self.eflat, self.nflat)*180./np.pi
			self.lat_tmparr = lat*np.ones_like(self.img['data'][:, :, 0])
			self.lon_tmparr = lon*np.ones_like(self.img['data'][:, :, 0])
			self.longeo, self.latgeo, back_az = g.fwd(self.lon_tmparr, self.lat_tmparr, self.ang_deg, self.dist)
			#self.longeo = np.ma.masked_where(self.r_ > self.r_incl, self.longeo)
			#self.latgeo = np.ma.masked_where(self.r_ > self.r_incl, self.latgeo)	
			self.longeo[self.r_ > self.r_incl] = np.nan
			self.latgeo[self.r_ > self.r_incl] = np.nan
			self.distmax = 30000.
			self.longeo[self.dist > self.distmax] = np.nan
			self.latgeo[self.dist > self.distmax] = np.nan
			return self.longeo, self.latgeo

		def rad_calibration(self, radcal_dict=None, exptime=None):
			if radcal_dict is None:
				radcal_dict = self.radcaldic
			radcoef   = np.zeros_like(self.img['data'][:, :, :])
			radcoef[:, :, 0] = radcal_dict['red']['c0']   + self.r_2 * radcal_dict['red']['c2']
			radcoef[:, :, 1] = radcal_dict['green']['c0'] + self.r_2 * radcal_dict['green']['c2']
			radcoef[:, :, 2] = radcal_dict['blue']['c0']  + self.r_2 * radcal_dict['blue']['c2']
			if radcal_dict['type']['pre'].lower() == 'count' and self.img['type'].lower() == 'count':
				self.img['data'][:, :, 0] =  self.img['data'][:, :, 0] * radcoef[:, :, 0] * exptime
				self.img['data'][:, :, 1] =  self.img['data'][:, :, 1] * radcoef[:, :, 1] * exptime
				self.img['data'][:, :, 2] =  self.img['data'][:, :, 2] * radcoef[:, :, 2] * exptime
				self.img['type'] = 'radiance'
				self.img['unit'] = 'W m^(-2) sr^(-1)'
			else:
				print(' !!! Calibration data type does not match !!! Skipping... ')
			return self.img

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
def read_fits(fits_filename, flipud=False, fliplr=True, mask_fits_filename=None):
	if not os.path.exists(fits_filename):
		print('Error: {} not found.'.format(fits_filename))
		sys.exit()
	print('Reading {}'.format(fits_filename))
	handle = fits.open(fits_filename)
	fimg = reshapeFITS(handle[0].data)
	if flipud:
		fimg = np.flipud(fimg)
	if fliplr:
		fimg = np.fliplr(fimg)
	fheader = handle[0].header
	
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

	return fimg, fheader

# # Calculate Direct-Cosine-Matrix to convert NED(North-East-Down) coordinate to
# #  camera coordinate
# # Usage: d(Cam) = R*d(NED)
# def R_NED2Cam(roll, pitch, yaw): #roll, pitch, yaw [radian]
# 	cr = np.cos(roll)
# 	sr = np.sin(roll)
# 	cp = np.cos(pitch)
# 	sp = np.sin(pitch)
# 	cy = np.cos(yaw)
# 	sy = np.sin(yaw)
# 	Rr = np.matrix([[ 1.,  0.,  0.],
# 					[ 0.,  cr,  sr],
# 					[ 0., -sr,  cr]])
# 	Rp = np.matrix([[ cp,  0., -sp],
# 					[ 0.,  1.,  0.],
# 					[ sp,  0.,  cp]])
# 	Ry = np.matrix([[ cy,  sy,  0.],
# 					[-sy,  cy,  0.],
# 					[ 0.,  0.,  1.]])
# 	R = np.matmul(Rr, np.matmul(Rp, Ry))
# 	return R

# def new_get_image(self, tile):
# 	import six
# 	from PIL import Image
# 	if six.PY3:
# 		from urllib.request import urlopen, Request
# 	else:
# 		from urllib2 import urlopen
# 	url = self._image_url(tile)  # added by H.C. Winsemius
# 	req = Request(url) # added by H.C. Winsemius
# 	req.add_header('User-agent', 'your bot 0.1')
# 	# fh = urlopen(url)  # removed by H.C. Winsemius
# 	fh = urlopen(req)
# 	im_data = six.BytesIO(fh.read())
# 	fh.close()
# 	img = Image.open(im_data)
# 	img = img.convert(self.desired_tile_form)

# 	return img, self.tileextent(tile), 'lower'

# class Camera_arcsix:
# 	def __init__(self, date):
# 		self.date = date

# 		# Constants for ARCSIX flight meta data

# 		self._flights = {
# 			'2024-05-17': {
# 				'description'	:	'Test flight 1',
# 				'available'		:	False,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-05-21':{
# 				'description'	:	'Test flight 2',
# 				'available'		:	False,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-05-23':{
# 				'description'	:	'Transit flight to Pituffik (Spring)',
# 				'available'		:	False,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-05-24':{
# 				'description'	:	'Test flight 3',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-05-28':{
# 				'description'	:	'Research Flight 01',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-05-30':{
# 				'description'	:	'Research Flight 02',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-05-31':{
# 				'description'	:	'Research Flight 03',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-03':{
# 				'description'	:	'Research Flight 04',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-05':{
# 				'description'	:	'Research Flight 05',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-06':{
# 				'description'	:	'Research Flight 06',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-07':{
# 				'description'	:	'Research Flight 07',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-10':{
# 				'description'	:	'Research Flight 08',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-11':{
# 				'description'	:	'Research Flight 09',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-06-13':{
# 				'description'	:	'Research Flight 10',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-07-08':{
# 				'description'	:	'Ground Test',
# 				'available'		:	False,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-07-09':{
# 				'description'	:	'Ground Test',
# 				'available'		:	False,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-07-22':{
# 				'description'	:	'Transit flight to Pituffik (Summer)',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-07-25':{
# 				'description'	:	'Research Flight 11',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-07-29':{
# 				'description'	:	'Research Flight 12',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-07-30':{
# 				'description'	:	'Research Flight 13',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-01':{
# 				'description'	:	'Research Flight 14',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-02':{
# 				'description'	:	'Research Flight 15',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-07':{
# 				'description'	:	'Research Flight 16',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-08':{
# 				'description'	:	'Research Flight 17',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-09':{
# 				'description'	:	'Research Flight 18',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-15':{
# 				'description'	:	'Research Flight 19',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 			'2024-08-16':{
# 				'description'	:	'Transit flight to Bangor (Summer)',
# 				'available'		:	True,
# 				'toff_day'		:	0.,
# 				'toff_sec'		:	0.,
# 			},
# 		}

# 		self.radcal_dict_simple = {
# 			'red'	: {	'c0':       0.03697803089161538   ,   # red channel cal main term
# 					    'c2':       2.9306376815596668e-08,   # quadratic correction term
# 					    'lambda': 626.408                 ,   # center wavelength [nm]
# 					    'f_toa':    1.6687                },  # TOA solar irradiance [W/m2/nm]
# 			'green'	: {	'c0':       0.0379816678515955    ,   # green channel cal main term
# 					    'c2':       2.4174566687203538e-08,   # quadratic correction term
# 					    'lambda': 553.572                 ,   # center wavelength [nm]
# 					    'f_toa':    1.8476                },  # TOA solar irradiance [W/m2/nm]}
# 			'blue'	: {	'c0':       0.04097034968881991   ,   # blue channel cal main term
# 					    'c2':       2.3561354650281456e-08,   # quadratic correction term
# 					    'lambda': 492.667                 ,   # center wavelength [nm]
# 					    'f_toa':    1.9084                },  # TOA solar irradiance [W/m2/nm]}
# 			'type'	: {	'pre':    'count'                 ,   # type before conversion
# 					    'post':   'radiance'              }   # type after conversion
# 		}

# 		self.flipud = True
# 		self.fliplr = True
# 		self.leftpix  = np.array([2190, 1703])
# 		self.rightpix = np.array([1004,  162])
# 		self.centerpix = np.int_(0.5*(self.leftpix + self.rightpix))
# 		self.headvector = np.array([-(self.rightpix[1] - self.centerpix[1]),
# 									 (self.rightpix[0] - self.centerpix[0])])

# 		self.pit_off =  1.5  # deg
# 		self.rol_off =  1.5  # deg
# 		self.hed_off =  0.0  # deg rotate clockwise


# 		self.hsk_dict = None
# 		self.hrss = None
# 		self.lats = None
# 		self.lons = None
# 		self.alts = None
# 		self.pits = None
# 		self.rols = None
# 		self.heds = None

# 		self.fits_allfiles = None
		
# 		self.date_check(self.date)

# 		self.flight_meta = self._flights[date]
# 		self.flight_date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
		
# 		self.camera_setting = {	'centerpix'	:	self.centerpix,
# 								'headvector':	self.headvector,
# 								'pit_off'	:	self.pit_off,
# 								'hed_off'	:	self.hed_off,
# 								'rol_off'	:	self.rol_off,
# 								'flipud'	:	self.flipud,
# 								'fliplr'	:	self.fliplr,}
		
# 	def date_check(self, date):
# 		if date not in self._flights:
# 			message = 'Error [Meta_arcsix]: date %s is not in the list of flight dates. ' \
# 						+ 'Make sure it is in YYYY-MM-DD format.'
# 			raise OSError(message)
# 		else:
# 			if not self._flights[date]['available']:
# 				message = 'Error [Meta_arcsix]: date %s is not avaiable.'
# 				raise OSError(message)
# 			else:
# 				print('Camera images for [%s]' % self._flights[date]['description'])
	
# 	def load_hsk(self, location='argus'):
# 		if location == 'argus':
# 			path = '/Volumes/argus/field/arcsix/processed'
# 		else:
# 			path = location
# 		yyyy, mm, dd = self.date.split('-')
# 		hsk_fname = 'ARCSIX-HSK_P3B_%s%s%s_v0.h5' % (yyyy, mm, dd)
# 		hsk_path = os.path.join(path, hsk_fname)
# 		hsk_files  = glob.glob(hsk_path)
# 		if len(hsk_files) == 0:
# 			message = 'Error [Meta_arcsix.load_hsk]: No housekeeping file found. Check the path/access to the correct directory.'
# 			raise OSError(message)
# 		elif len(hsk_files) != 1:
# 			message = 'Error [Meta_arcsix.load_hsk]: More than 1 housekeeping file found. Check the files in the directory.'
# 			raise OSError(message)
# 		else:
# 			hsk_fn = hsk_files[0]
		
# 		self.hsk_dict = self.read_hsk(hsk_fn)

# 		self.hrss = self.hsk_dict['hrs']
# 		self.lats = self.hsk_dict['lat']
# 		self.lons = self.hsk_dict['lon']
# 		self.alts = self.hsk_dict['alt']
# 		self.pits = self.hsk_dict['pit'] # [deg]
# 		self.rols = self.hsk_dict['rol'] # [deg]
# 		self.heds = self.hsk_dict['hed'] # [deg]


# 	def read_hsk(self, hsk_filename):
# 		print('Reading {}'.format(hsk_filename))
# 		h5f = h5py.File(hsk_filename, 'r')
# 		hsk_data = {'doy': jd_to_doy(h5f['jday'][...], 2024) ,
# 					'hrs': h5f['tmhr'][...]        ,
# 					'alt': h5f['alt'][...],
# 					'lat': h5f['lat'][...]    ,
# 					'lon': h5f['lon'][...]   ,
# 					'hed': h5f['ang_hed'][...],
# 					'rol': h5f['ang_rol'][...]  ,
# 					'pit': h5f['ang_pit'][...]  }
# 		return hsk_data
	
# 	def load_fits(self, start_time, end_time, location='ARCSIX1KSS'):
# 		self.fits_allfiles = self.locate_allfits(location=location)

# 		st_dt = datetime.datetime.strptime(start_time, '%H:%M:%S')
# 		# if campaign.lower() == 'camp2ex' and st_dt.hour < 15: st_dt += datetime.timedelta(days=1)
# 		en_dt = datetime.datetime.strptime(end_time, '%H:%M:%S')
# 		# if campaign.lower() == 'camp2ex' and en_dt.hour < 15: en_dt += datetime.timedelta(days=1)

# 		self.t_offset = datetime.timedelta(	days=self.flight_meta['toff_day'], 
# 											seconds=self.flight_meta['toff_sec'])

# 		fits_list = []
# 		lat_list, lon_list = [], []
# 		for ifits, fits_fn in enumerate(self.fits_allfiles):
# 			tt_fn = datetime.datetime.strptime(fits_fn[-14:-6], '%H_%M_%S')
# 			# if campaign.lower() == 'camp2ex' and tt_fn.hour < 15: tt_fn += datetime.timedelta(days=1)
# 			if st_dt <= tt_fn <= en_dt:
# 				fits_list.append(fits_fn)
# 				t_fn  = datetime.datetime.strptime(self.date + ' ' + fits_fn[-14:-6], '%Y-%m-%d %H_%M_%S')
# 				hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + self.t_offset.total_seconds()/3600.
# 				lat  = np.interp(hrs, self.hrss, self.lats) # interpolate to find values for the time at which the image was taken
# 				lon  = np.interp(hrs, self.hrss, self.lons)
# 				lat_list.append(lat)
# 				lon_list.append(lon)
# 		self.lat_list = np.array(lat_list)
# 		self.lon_list = np.array(lon_list)
# 		return fits_list
	
# 	def locate_allfits(self, location='ARCSIX1KSS'):
# 		yyyy, mm, dd = self.date.split('-')
# 		num_rf = self.flight_meta['description'].split(' ')[-1]
# 		if location in ['ARCSIX1KSS', 'ARCISX3KSS']:
# 			if self.flight_date <= datetime.datetime(2024, 8, 3):
# 				path = '/Volumes/ARCSIX1KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
# 			else:
# 				path = '/Volumes/ARCSIX3KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
# 		elif location in ['ARCSIX2KSS', 'ARCISX4KSS']:
# 			if self.flight_date <= datetime.datetime(2024, 8, 3):
# 				path = '/Volumes/ARCSIX2KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
# 			else:
# 				path = '/Volumes/ARCSIX4KSS/ARCSIX_RF%s_%s_%s_%s/Camera/**/*.fits' % (num_rf, yyyy, mm, dd)
# 		else:
# 			path = location
		
# 		fits_allfiles = glob.glob(path, recursive=True)
# 		if len(fits_allfiles) == 0:
# 			message = 'Error [Meta_arcsix.load_fits]: No fits file found. Check the path/access to the correct directory.'
# 			raise OSError(message)
# 		fits_allfiles = sorted(fits_allfiles)
# 		return fits_allfiles
	
# 	# def interpolate_hsk_for_fits(self, fits_fn):
# 		# t_fn  = datetime.datetime.strptime(self.date + ' ' + fits_fn[-14:-6], '%Y-%m-%d %H_%M_%S')
# 	def interpolate_hsk_for_fits(self, date_obs_str):
# 		t_fn  = datetime.datetime.strptime(self.date + ' ' + date_obs_str.split('T')[1][:-1], '%Y-%m-%d %H:%M:%S.%f')
# 		t_act = t_fn + self.t_offset
# 		hrs   = t_fn.hour + t_fn.minute/60. + t_fn.second/3600. + self.t_offset.total_seconds()/3600.

# 		lat  = np.interp(hrs, self.hrss, self.lats) # interpolate to find values for the time at which the image was taken
# 		lon  = np.interp(hrs, self.hrss, self.lons)
# 		alt  = np.interp(hrs, self.hrss, self.alts)
# 		pit  = np.interp(hrs, self.hrss, self.pits)
# 		rol  = np.interp(hrs, self.hrss, self.rols)
# 		hed  = np.interp(hrs, self.hrss, self.heds)

# 		when = t_act.replace(tzinfo=datetime.timezone.utc)
# 		sza  = pysolar.solar.get_altitude(lat, lon, when)
# 		saa  = pysolar.solar.get_azimuth( lat, lon, when, elevation=alt)

# 		aircraft_status = {	'lat'		:	lat,
# 							'lon'		:	lon,
# 							'rol'		:	rol,
# 							'pit'		:	pit,
# 							'hed'		:	hed,
# 							'alt'		:	alt,
# 							'sza'		:	sza,
# 							'saa'		:	saa}
# 		return t_act, aircraft_status

# def find_center(xs, ys):
# 	A = np.vstack((xs, ys, np.ones((len(xs))))).T
# 	v = -(xs ** 2 + ys ** 2)
# 	u, residuals, rank, s = np.linalg.lstsq(A, v, rcond=None)
# 	cx_pred = u[0] / (-2.)
# 	cy_pred = u[1] / (-2.)
# 	r_pred = np.sqrt(cx_pred ** 2 + cy_pred ** 2 - u[2])
# 	return (cx_pred, cy_pred), r_pred

# def find_matched_keypoints(lon_xx, lat_yy, image1, image2, dmax=50):
#     import cv2
#     image1[np.isnan(image1)] = 0.0
#     image2[np.isnan(image2)] = 0.0
#     # Step 1: Load the images
#     # image1 = cv2.imread(image1_path)
#     # image2 = cv2.imread(image2_path)
#     image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') 
#     image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

#     # Step 2: Initialize SIFT detector
#     sift = cv2.SIFT_create()

#     # Step 3: Find keypoints and descriptors with SIFT
#     keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

#     # Step 4: Match descriptors using FLANN based matcher
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)

#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#     # Apply Lowe's ratio test to filter good matches
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
#     print('good_matches', len(good_matches))

#     # Step 5: Extract the locations of the matched points
#     points_image1 = []
#     points_image2 = []
    
#     for match in good_matches:
#         img1_idx = match.queryIdx
#         img2_idx = match.trainIdx
        
#         # Get the coordinates of the keypoints
#         (x1, y1) = keypoints1[img1_idx].pt
#         (x2, y2) = keypoints2[img2_idx].pt
        
#         points_image1.append((x1, y1))
#         points_image2.append((x2, y2))

#     # return points_image1, points_image2
#     m_1 = np.array([[x, y] for x, y in points_image1])
#     m_2 = np.array([[x, y] for x, y in points_image2])

#     ixm1 = np.int_(m_1[:, 0])
#     iym1 = np.int_(m_1[:, 1])
#     rxm1 = m_1[:, 0] - ixm1
#     rym1 = m_1[:, 1] - iym1
#     lonm1 = (1. - rxm1)*lon_xx[iym1, ixm1] + rxm1*lon_xx[iym1, ixm1 + 1]
#     latm1 = (1. - rym1)*lat_yy[iym1, ixm1] + rym1*lat_yy[iym1 + 1, ixm1]
#     ixm2 = np.int_(m_2[:, 0])
#     iym2 = np.int_(m_2[:, 1])
#     rxm2 = m_2[:, 0] - ixm2
#     rym2 = m_2[:, 1] - iym2
#     lonm2 = (1. - rxm2)*lon_xx[iym2, ixm2] + rxm2*lon_xx[iym2, ixm2 + 1]
#     latm2 = (1. - rym2)*lat_yy[iym2, ixm2] + rym2*lat_yy[iym2 + 1, ixm2]
#     dgrid = np.sqrt((m_2[:, 0] - m_1[:, 0])**2 + (m_2[:, 1] - m_1[:, 1])**2)
#     return lonm1[dgrid < dmax], latm1[dgrid < dmax], lonm2[dgrid < dmax], latm2[dgrid < dmax], dgrid[dgrid < dmax]

