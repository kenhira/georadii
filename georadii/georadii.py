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

			elif self.input_coordinate == 'camera' or self.input_coordinate == 'geometry':
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
		# cam_class.hemispheric_to_cartesian()
		cam_class.cartesian_to_latlon()
		latlon_meta = cam_class.cam_meta
		latlon_meta['longeo'] = cam_class.longeo
		latlon_meta['latgeo'] = cam_class.latgeo
		latlon_meta['valid_domain'] = cam_class.valid_domain
		self.latlon_data = self.LatLon(cam_class.img, latlon_meta)
		return latlon_meta['longeo'], latlon_meta['latgeo']
		
	def gridded(self, gridmeta=None, enable_mp=False, use_c=True): # Note: multiprocessing is currently slow
		latlon_class = self.latlon_data
		grid_meta = {}
		grid_meta['transform'] = {	'active'      : True, # Rotate the lat/lon coordinate system so that the grids are more regular
									'center'      : (0.5*(np.nanmin(latlon_class.longeo) + np.nanmax(latlon_class.longeo)),
														0.5*(np.nanmin(latlon_class.latgeo) + np.nanmax(latlon_class.latgeo))),
									'inclination' : 0. }
		f_extend = 5.
		grid_meta['x'] = {	'min' :  0.5*f_extend*np.nanmin(latlon_class.longeo) - 0.5*f_extend*np.nanmax(latlon_class.longeo),
							'max' : -0.5*f_extend*np.nanmin(latlon_class.longeo) + 0.5*f_extend*np.nanmax(latlon_class.longeo),
							'incr': (np.nanmax(latlon_class.longeo) - np.nanmin(latlon_class.longeo)) / 250.} 
		grid_meta['y']  = {	'min' :  0.5*f_extend*np.nanmin(latlon_class.latgeo) - 0.5*f_extend*np.nanmax(latlon_class.latgeo),
							'max' : -0.5*f_extend*np.nanmin(latlon_class.latgeo) + 0.5*f_extend*np.nanmax(latlon_class.latgeo),
							'incr': (np.nanmax(latlon_class.latgeo) - np.nanmin(latlon_class.latgeo)) / 250.}
		if gridmeta is None:
			pass
		else:
			grid_meta.update(gridmeta)
		# define the grid system
		scatdat_x = latlon_class.longeo
		scatdat_y = latlon_class.latgeo
		scatdat_data = latlon_class.img['data']
		scatdat_valid = latlon_class.valid_domain
		if grid_meta['transform']['active']: # Rotate the lat/lon coordinate system so that the grids are more regular
			lon_center, lat_center = grid_meta['transform']['center']
			inclination = grid_meta['transform']['inclination']
			tmp_scatdat_y, tmp_scatdat_x = self.transform_coordinates(scatdat_y, scatdat_x, lat_center, lon_center, inclination)
			if gridmeta is None:
				grid_xmin, grid_xmax, grid_dx = np.nanmin(tmp_scatdat_x), np.nanmax(tmp_scatdat_x), (np.nanmax(tmp_scatdat_x) - np.nanmin(tmp_scatdat_x))/250.
				grid_ymin, grid_ymax, grid_dy = np.nanmin(tmp_scatdat_y), np.nanmax(tmp_scatdat_y), (np.nanmax(tmp_scatdat_y) - np.nanmin(tmp_scatdat_y))/250.
			else:
				grid_xmin, grid_xmax, grid_dx = grid_meta['x']['min'], grid_meta['x']['max'], grid_meta['x']['incr']
				grid_ymin, grid_ymax, grid_dy = grid_meta['y']['min'], grid_meta['y']['max'], grid_meta['y']['incr']
			grid_xarr = np.arange(grid_xmin, grid_xmax, grid_dx) + 0.5*grid_dx
			grid_yarr = np.arange(grid_ymin, grid_ymax, grid_dy) + 0.5*grid_dy
			tmp_grid_xx, tmp_grid_yy = np.meshgrid(grid_xarr, grid_yarr)
			grid_yy, grid_xx = self.inverse_transform_coordinates(tmp_grid_yy, tmp_grid_xx, lat_center, lon_center, inclination)

		else:
			tmp_scatdat_x, tmp_scatdat_y = scatdat_x, scatdat_y
			grid_xmin, grid_xmax, grid_dx = grid_meta['x']['min'], grid_meta['x']['max'], grid_meta['x']['incr']
			grid_ymin, grid_ymax, grid_dy = grid_meta['y']['min'], grid_meta['y']['max'], grid_meta['y']['incr']
			grid_xarr = np.arange(grid_xmin, grid_xmax, grid_dx) + 0.5*grid_dx
			grid_yarr = np.arange(grid_ymin, grid_ymax, grid_dy) + 0.5*grid_dy
			grid_xx, grid_yy = np.meshgrid(grid_xarr, grid_yarr)
		
		grid_dim = (grid_xx.shape[0], grid_xx.shape[1], scatdat_data.shape[2] if len(scatdat_data.shape) > 2 else 0)

		datsum, ncount = self.gridding2d_core(tmp_scatdat_x, tmp_scatdat_y, scatdat_data, grid_xmin, grid_xmax, grid_dx, grid_ymin, grid_ymax, grid_dy, grid_dim, scatdat_valid=scatdat_valid, enable_mp=enable_mp, use_c=use_c)

		datout = np.zeros_like(datsum)
		if grid_dim[2] == 0:
			datout[:, :] = datsum[:, :]/np.float64(ncount)
		elif grid_dim[2] >= 1:
			for ich in range(grid_dim[2]):
				datout[:, :, ich] = datsum[:, :, ich]/np.float64(ncount)  # divide by the number of pixels to average the image values
		
		return grid_xx, grid_yy, datout, ncount
	
	def gridded_angular(self, gridmeta=None, enable_mp=False, use_c=True): # Note: multiprocessing is currently slow
		cam_class = self.camera_data
		grid_meta = {}
		grid_meta['raa'] = {	'min' :  -1.,
								'max' : 359.,
								'incr':   2.} 
		grid_meta['vza'] = {	'min' :   0.,
								'max' :  90.,
								'incr':   1.}
		grid_meta['saa'] = 0.
		if gridmeta is None:
			pass
		else:
			grid_meta.update(gridmeta)
		# define the grid system
		scatdat_raa = np.rad2deg(cam_class.vaa) - grid_meta['saa']
		scatdat_vza = np.rad2deg(cam_class.vza)
		scatdat_data = cam_class.img['data']
		scatdat_valid = cam_class.valid_domain
		grid_xmin, grid_xmax, grid_dx = grid_meta['raa']['min'], grid_meta['raa']['max'], grid_meta['raa']['incr']
		grid_ymin, grid_ymax, grid_dy = grid_meta['vza']['min'], grid_meta['vza']['max'], grid_meta['vza']['incr']
		grid_xarr = np.arange(grid_xmin, grid_xmax, grid_dx) + 0.5*grid_dx
		grid_yarr = np.arange(grid_ymin, grid_ymax, grid_dy) + 0.5*grid_dy
		grid_xx, grid_yy = np.meshgrid(grid_xarr, grid_yarr)

		grid_dim = (grid_xx.shape[0], grid_xx.shape[1], scatdat_data.shape[2] if len(scatdat_data.shape) > 2 else 0)

		if scatdat_valid is None:
			scatdat_valid = np.ones_like(scatdat_raa, dtype=bool) & (scatdat_vza < 90.)
		else:
			scatdat_valid = scatdat_valid & (scatdat_vza < 90.)
		scatdat_valid = scatdat_valid & (~np.isnan(scatdat_data).any(axis=-1) if len(scatdat_data.shape) > 2 else ~np.isnan(scatdat_data))
		
		scatdat_raa = (scatdat_raa + 360.) % 360.
		scatdat_raa[scatdat_raa > grid_meta['raa']['max']] -= 360. # change the range to grid_meta['raa']['min'] ~ grid_meta['raa']['max']

		datsum, ncount = Georadii.gridding2d_core(scatdat_raa, scatdat_vza, scatdat_data, grid_xmin, grid_xmax, grid_dx, grid_ymin, grid_ymax, grid_dy, grid_dim, scatdat_valid=scatdat_valid, enable_mp=enable_mp, use_c=use_c)

		datout = np.zeros_like(datsum)
		if grid_dim[2] == 0:
			datout[:, :] = datsum[:, :]/np.float64(ncount)
		elif grid_dim[2] >= 1:
			for ich in range(grid_dim[2]):
				datout[:, :, ich] = datsum[:, :, ich]/np.float64(ncount)  # divide by the number of pixels to average the image values
		
		return grid_xx, grid_yy, datout, ncount
	
	@staticmethod
	def gridded_angular_custom(scatdat_raa, scatdat_vza, scatdat_data, scatdat_valid=None, gridmeta=None, enable_mp=False, use_c=True): # Note: multiprocessing is currently slow
		grid_meta = {}
		grid_meta['raa'] = {	'min' :  -1.,
								'max' : 359.,
								'incr':   2.} 
		grid_meta['vza'] = {	'min' :   0.,
								'max' :  90.,
								'incr':   1.}
		grid_meta['saa'] = 0.
		if gridmeta is None:
			pass
		else:
			grid_meta.update(gridmeta)
		# define the grid system
		grid_xmin, grid_xmax, grid_dx = grid_meta['raa']['min'], grid_meta['raa']['max'], grid_meta['raa']['incr']
		grid_ymin, grid_ymax, grid_dy = grid_meta['vza']['min'], grid_meta['vza']['max'], grid_meta['vza']['incr']
		grid_xarr = np.arange(grid_xmin, grid_xmax, grid_dx) + 0.5*grid_dx
		grid_yarr = np.arange(grid_ymin, grid_ymax, grid_dy) + 0.5*grid_dy
		grid_xx, grid_yy = np.meshgrid(grid_xarr, grid_yarr)
		grid_dim = (grid_xx.shape[0], grid_xx.shape[1], scatdat_data.shape[2] if len(scatdat_data.shape) > 2 else 0)

		if scatdat_valid is None:
			scatdat_valid = np.ones_like(scatdat_raa, dtype=bool) & (scatdat_vza < 90.)
		else:
			scatdat_valid = scatdat_valid & (scatdat_vza < 90.)
		scatdat_valid = scatdat_valid & (~np.isnan(scatdat_data).any(axis=-1) if len(scatdat_data.shape) > 2 else ~np.isnan(scatdat_data))

		scatdat_raa = (scatdat_raa + 360.) % 360.
		scatdat_raa[scatdat_raa > grid_meta['raa']['max']] -= 360. # change the range to grid_meta['raa']['min'] ~ grid_meta['raa']['max']

		datsum, ncount = Georadii.gridding2d_core(scatdat_raa, scatdat_vza, scatdat_data, grid_xmin, grid_xmax, grid_dx, grid_ymin, grid_ymax, grid_dy, grid_dim, scatdat_valid=scatdat_valid, enable_mp=enable_mp, use_c=use_c)

		datout = np.zeros_like(datsum)
		if grid_dim[2] == 0:
			datout[:, :] = datsum[:, :]/np.float64(ncount)
		elif grid_dim[2] >= 1:
			for ich in range(grid_dim[2]):
				datout[:, :, ich] = datsum[:, :, ich]/np.float64(ncount)  # divide by the number of pixels to average the image values
		
		return grid_xx, grid_yy, datout, ncount
	
	@staticmethod
	def gridding2d_core(scatdat_x, scatdat_y, scatdat_data, grid_xmin, grid_xmax, grid_dx, grid_ymin, grid_ymax, grid_dy, grid_dim, scatdat_valid=None, enable_mp=False, use_c=True):
		if use_c:
			from georadii.compute import gridding2d_weight, gridding2d_count
			if scatdat_valid is not None:
				g1 = scatdat_x[scatdat_valid]
				g2 = scatdat_y[scatdat_valid]
				scatdat_validata = scatdat_data[scatdat_valid]
			else:
				g1 = scatdat_x.flatten()
				g2 = scatdat_y.flatten()
				scatdat_validata = scatdat_data.flatten() if grid_dim[2] == 0 else scatdat_data.reshape(g1.shape[0], scatdat_data.shape[2])
			datsum = gridding2d_weight(g2, g1, scatdat_validata.copy(), grid_ymin, grid_ymax, grid_dy, grid_xmin, grid_xmax, grid_dx)#.transpose(0, 1, 2)
			ncount = gridding2d_count(g2, g1, grid_ymin, grid_ymax, grid_dy, grid_xmin, grid_xmax, grid_dx)#.T
		else:
			indgx = np.int_(((scatdat_x - grid_xmin + 0.5*grid_dx)//grid_dx + grid_dim[1]) % grid_dim[1])
			indgy = np.int_(((scatdat_y - grid_ymin + 0.5*grid_dy)//grid_dy + grid_dim[0]) % grid_dim[0])
			x1temp, x2temp = np.meshgrid(np.arange(indgy.shape[1]), np.arange(indgy.shape[0]))
			if scatdat_valid is not None:
				x1 = x1temp[scatdat_valid]
				x2 = x2temp[scatdat_valid]
			else:
				x1 = x1temp
				x2 = x2temp
			if enable_mp:
				nproc = 8
				x1_ch = np.array_split(x1, nproc, axis=0)
				x2_ch = np.array_split(x2, nproc, axis=0)
				args = [(x1_ch[ich], x2_ch[ich], indgy, indgx, scatdat_data, grid_dim) for ich in range(nproc)]
				with mp.Pool(processes=nproc) as pool:
					results = pool.map(Georadii.count_and_sum, args)
				count_list, value_list = zip(*results)
				count_stack = np.stack(count_list, axis=-1)
				value_stack = np.stack(value_list, axis=-1)
				ncount, datsum = np.sum(count_stack, axis=2), np.sum(value_stack, axis=3)
			else:
				ncount, datsum = Georadii.count_and_sum((x1, x2, indgy, indgx, scatdat_data, grid_dim))
		return datsum, ncount
	
	@staticmethod
	def count_and_sum(arg):
		x1, x2, ap, aq, av, dim_grid = arg
		ndimg1, ndimg2, ndimg3 = dim_grid
		cnt = np.zeros((ndimg1, ndimg2))
		if ndimg3 == 0:
			val = np.zeros((ndimg1, ndimg2))
			for i1, i2 in zip(x1.flatten(), x2.flatten()):
				cnt[ap[i2, i1], aq[i2, i1]] += 1
				val[ap[i2, i1], aq[i2, i1]] += av[i2, i1]
		elif ndimg3 >= 1:
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
	
	@staticmethod
	def transform_coordinates(lat_in, lon_in, lat_piv, lon_piv, inclination):
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

	@staticmethod
	def inverse_transform_coordinates(lat_in, lon_in, lat_piv, lon_piv, inclination):
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
			self.earth_a = 6378137.0 # m (equatorial radius)
			self.earth_b = 6356752.3 # m (polar radius)

			self.cam_meta = cam_meta
			if self.cam_meta is None:
				self.cam_meta = {}

			# _radcaldic = {}
			# _radcaldic['red']   = {	'c0':       0.03697803089161538   ,   # red channel cal main term
			# 						'c2':       2.9306376815596668e-08,   # quadratic correction term
			# 						'lambda': 626.408                 ,   # center wavelength [nm]
			# 						'f_toa':    1.6687                }   # TOA solar irradiance [W/m2/nm]
			# _radcaldic['green'] = {	'c0':       0.0379816678515955    ,   # green channel cal main term
			# 						'c2':       2.4174566687203538e-08,   # quadratic correction term
			# 						'lambda': 553.572                 ,   # center wavelength [nm]
			# 						'f_toa':    1.8476                }   # TOA solar irradiance [W/m2/nm]
			# _radcaldic['blue']  = {	'c0':       0.04097034968881991   ,   # blue channel cal main term
			# 						'c2':       2.3561354650281456e-08,   # quadratic correction term
			# 						'lambda': 492.667                 ,   # center wavelength [nm]
			# 						'f_toa':    1.9084                }   # TOA solar irradiance [W/m2/nm]
			# _radcaldic['type'] = {	'pre':  'raw count'                ,   # type before conversion
			# 						'post': 'radiance'                 }   # type after conversion
			# _radcaldic['exposure'] = {'time':  0.0                   }   # exposure time [sec?]

			# if 'type' not in self.cam_meta:
			# 	message = 'Error [Georadii.AllSkyCamera]: Key <type> is not found in <cam_meta>.'
			# 	raise OSError(message)
			# if 'unit' not in self.cam_meta:
			# 	message = 'Error [Georadii.AllSkyCamera]: Key <unit> is not found in <cam_meta>.'
			# 	raise OSError(message)
			# if 'centerpix' not in self.cam_meta:
			# 	message = 'Error [Georadii.AllSkyCamera]: Key <centerpix> is not found in <cam_meta>.'
			# 	raise OSError(message)
			# if 'headvector' not in self.cam_meta:
			# 	message = 'Error [Georadii.AllSkyCamera]: Key <headvector> is not found in <cam_meta>.'
			# 	raise OSError(message)

			self.valid_domain = np.ones_like(image['data'][:, :, 0], dtype=bool)

			if 'geometry' in image:
				self.cam_meta['vza'] = image['geometry']['vza']
				self.cam_meta['vaa'] = image['geometry']['vaa']
				self.cam_meta['centerpix'] = image['geometry']['centerpix']
				self.cam_meta.update(image['aircraft_status'])
				self.valid_domain = self.valid_domain & (~np.isnan(self.cam_meta['vza']))
				self.valid_domain = self.valid_domain & (~np.isnan(self.cam_meta['vaa']))
				if 'fov' not in self.cam_meta:
					self.load_image(image, calc_angle='none')
				else:
					self.zenith_limit = self.cam_meta['fov']
					_degperpix =  0.09  # deg/pixel
					_degperpix2 = 0.00  # deg/pixel
					self.degperpix    = self.cam_meta.get('degperpix',  _degperpix) # deg/pixel
					self.degperpix2   = self.cam_meta.get('degperpix2', _degperpix2) # deg/pixel
					self.load_image(image, calc_angle='fov')
			else:
				_fov       = 65.    # deg
				_degperpix =  0.09  # deg/pixel
				_degperpix2 = 0.00  # deg/pixel
				_degperpix3 = 0.00  # deg/pixel
				_degperpix4 = 0.00  # deg/pixel
				_degperpix5 = 0.00  # deg/pixel
				_degperpix6 = 0.00  # deg/pixel
				self.zenith_limit = self.cam_meta.get('fov', _fov)             # deg
				self.degperpix    = self.cam_meta.get('degperpix',  _degperpix) # deg/pixel
				self.degperpix2   = self.cam_meta.get('degperpix2', _degperpix2)
				self.degperpix3   = self.cam_meta.get('degperpix3', _degperpix3)
				self.degperpix4   = self.cam_meta.get('degperpix4', _degperpix4)
				self.degperpix5   = self.cam_meta.get('degperpix5', _degperpix5)
				self.degperpix6   = self.cam_meta.get('degperpix6', _degperpix6)
				# img = {'data': image, 'type': self.cam_meta['type'], 'unit': self.cam_meta['unit']}
				self.load_image(image, calc_angle='all')
			
			self.radcal       = self.cam_meta.get('radcal', False)
			if 'rad_coef' in self.cam_meta and self.radcal:
				# self.radcaldic    = self.cam_meta.get('radcaldic', _radcaldic)
				self.rad_conversion(self.cam_meta['rad_coef'], self.cam_meta['exptime'])
			
			self.hemispheric_to_cartesian()


		def load_image(self, img, calc_angle='none'):
			self.img = img
			self.valid_domain = self.valid_domain & (~np.isnan(self.img['data'][:, :, 0]))
			self.valid_domain = self.valid_domain & (~np.isnan(self.img['data'][:, :, 1]))
			self.valid_domain = self.valid_domain & (~np.isnan(self.img['data'][:, :, 2]))
			if calc_angle == 'fov' or calc_angle == 'all':
				self.img2d_zero = np.zeros_like(self.img['data'][:, :, 0])
				self.xa, self.ya = np.arange(self.img['data'].shape[1]), np.arange(self.img['data'].shape[0])
				self.xx, self.yy = np.meshgrid(self.xa, self.ya)
				# self.r_incl = np.int_(self.zenith_limit/self.degperpix)
				if np.abs(self.degperpix2) < 1e-8:
					self.r_incl = np.int_(self.zenith_limit/self.degperpix)
				else:
					self.r_incl = np.int_((-self.degperpix + np.sqrt(self.degperpix**2. + 4.*self.degperpix2*self.zenith_limit))/(2.*self.degperpix2))
				self.r_2    = (self.xx - self.cam_meta['centerpix'][0])*(self.xx - self.cam_meta['centerpix'][0]) + (self.yy - self.cam_meta['centerpix'][1])*(self.yy - self.cam_meta['centerpix'][1])
				self.r_     = np.sqrt(self.r_2)
				self.valid_domain = self.valid_domain & (self.r_ < self.r_incl) & (~np.isnan(self.img['data'][:, :, 0]))
				self.valid_domain = self.valid_domain & (~np.isnan(self.img['data'][:, :, 1])) & (~np.isnan(self.img['data'][:, :, 2]))
				if calc_angle == 'all':
					self.zeniths  = np.ma.masked_where(self.r_ > self.r_incl, self.img2d_zero)
					self.azimuths = np.ma.masked_where(self.r_ > self.r_incl, self.img2d_zero)
					# self.zeniths  = self.zeniths  + self.r_*self.degperpix*np.pi/180.
					# self.zeniths  = self.zeniths  + (self.r_**2.*self.degperpix2 + self.r_*self.degperpix)*np.pi/180.
					self.zeniths  = self.zeniths  + (self.r_**6.*self.degperpix6 + self.r_**5.*self.degperpix5 + self.r_**4.*self.degperpix4
													+ self.r_**3.*self.degperpix3 + self.r_**2.*self.degperpix2 + self.r_*self.degperpix)*np.pi/180.
					self.head_ang = np.arctan2(self.cam_meta['headvector'][0], self.cam_meta['headvector'][1])
					self.azimuths = (self.azimuths + np.arctan2(self.xx - self.cam_meta['centerpix'][0], self.yy - self.cam_meta['centerpix'][1]) - self.head_ang) % (2.*np.pi)
			elif calc_angle == 'none':
				pass

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
			if 'vza' in self.cam_meta and 'vaa' in self.cam_meta:
				self.vza = self.cam_meta['vza']
				self.vaa = self.cam_meta['vaa']
				alt = self.cam_meta['alt']
				self.xcar = np.sin(self.vza)*np.cos(self.vaa)
				self.ycar = np.sin(self.vza)*np.sin(self.vaa)
				self.zcar = np.cos(self.vza)
				self.nflat = self.xcar*alt/self.zcar
				self.eflat = self.ycar*alt/self.zcar
			else:
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
			# self.dist    = np.sqrt(self.nflat*self.nflat + self.eflat*self.eflat)
			self.dist = self.dist_arc(lat, self.vza, self.cam_meta['alt'])
			self.ang_deg = np.arctan2(self.eflat, self.nflat)*180./np.pi
			self.lat_tmparr = lat*np.ones_like(self.img['data'][:, :, 0])
			self.lon_tmparr = lon*np.ones_like(self.img['data'][:, :, 0])
			self.longeo, self.latgeo, back_az = g.fwd(self.lon_tmparr, self.lat_tmparr, self.ang_deg, self.dist)
			#self.longeo = np.ma.masked_where(self.r_ > self.r_incl, self.longeo)
			#self.latgeo = np.ma.masked_where(self.r_ > self.r_incl, self.latgeo)	
			# self.longeo[self.r_ > self.r_incl] = np.nan
			# self.latgeo[self.r_ > self.r_incl] = np.nan
			self.longeo[~self.valid_domain] = np.nan
			self.latgeo[~self.valid_domain] = np.nan
			self.distmax = 30000.
			self.longeo[self.dist > self.distmax] = np.nan
			self.latgeo[self.dist > self.distmax] = np.nan
			return self.longeo, self.latgeo

		def rad_conversion(self, radcal_dict, exptime):
			print('Radiometric calibration coefficients are being applied...')
			# if radcal_dict is None:
			# 	radcal_dict = self.radcaldic
			# radcoef   = np.zeros_like(self.img['data'][:, :, :])
			# radcoef[:, :, 0] = radcal_dict['red']['c0']   + self.r_2 * radcal_dict['red']['c2']
			# radcoef[:, :, 1] = radcal_dict['green']['c0'] + self.r_2 * radcal_dict['green']['c2']
			# radcoef[:, :, 2] = radcal_dict['blue']['c0']  + self.r_2 * radcal_dict['blue']['c2']
			if radcal_dict['type']['input'].lower() == 'count per time' and self.img['type'].lower() == 'count':
				self.img['data'][:, :, 0] =  self.img['data'][:, :, 0] * radcal_dict['red'][:, :]   / exptime
				self.img['data'][:, :, 1] =  self.img['data'][:, :, 1] * radcal_dict['green'][:, :] / exptime
				self.img['data'][:, :, 2] =  self.img['data'][:, :, 2] * radcal_dict['blue'][:, :]  / exptime
				self.img['type'] = 'radiance'
				self.img['unit'] = 'W m^(-2) nm^(-1) sr^(-1)'
			else:
				print(' !!! Image and calibration data types do not match! <%s> vs <%s> Skipping... ' \
							% (radcal_dict['type']['input'].lower()))
			# if radcal_dict is None:
			# 	radcal_dict = self.radcaldic
			# radcoef   = np.zeros_like(self.img['data'][:, :, :])
			# radcoef[:, :, 0] = radcal_dict['red']['c0']   + self.r_2 * radcal_dict['red']['c2']
			# radcoef[:, :, 1] = radcal_dict['green']['c0'] + self.r_2 * radcal_dict['green']['c2']
			# radcoef[:, :, 2] = radcal_dict['blue']['c0']  + self.r_2 * radcal_dict['blue']['c2']
			# if radcal_dict['type']['pre'].lower() == 'count' and self.img['type'].lower() == 'count':
			# 	self.img['data'][:, :, 0] =  self.img['data'][:, :, 0] * radcoef[:, :, 0] * exptime
			# 	self.img['data'][:, :, 1] =  self.img['data'][:, :, 1] * radcoef[:, :, 1] * exptime
			# 	self.img['data'][:, :, 2] =  self.img['data'][:, :, 2] * radcoef[:, :, 2] * exptime
			# 	self.img['type'] = 'radiance'
			# 	self.img['unit'] = 'W m^(-2) sr^(-1)'
			# else:
			# 	print(' !!! Calibration data type does not match !!! Skipping... ')
			return self.img
		
		def dist_arc(self, lat, vza, alt): # lat [degree], vza [radian], alt [m]
			lat_rad = np.deg2rad(lat)
			r_e = np.sqrt((self.earth_a*np.cos(lat_rad))**2 + (self.earth_b*np.sin(lat_rad))**2)
			dist = r_e*(np.arcsin((r_e + alt)*np.sin(vza)/r_e) - vza)
			return dist

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
