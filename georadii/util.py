import os
import sys
import numpy as np
import h5py
from astropy.io import fits
import cv2

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
def read_fits(fits_filename, flipud=False, fliplr=True):
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

def find_center(xs, ys):
	A = np.vstack((xs, ys, np.ones((len(xs))))).T
	v = -(xs ** 2 + ys ** 2)
	u, residuals, rank, s = np.linalg.lstsq(A, v, rcond=None)
	cx_pred = u[0] / (-2.)
	cy_pred = u[1] / (-2.)
	r_pred = np.sqrt(cx_pred ** 2 + cy_pred ** 2 - u[2])
	return (cx_pred, cy_pred), r_pred

def find_matched_keypoints(lon_xx, lat_yy, image1, image2, dmax=50):
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
    print(d_grid.shape)
    m1 = m_1[np.where(d_grid < dmax), 0:2][0, :, :]
    m2 = m_2[np.where(d_grid < dmax), 0:2][0, :, :]
    print(m_1.shape)
    print(m1.shape)
    dgrid = d_grid[np.where(d_grid < dmax)]
    print(dgrid.shape)

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
            print(nei_idxs)
            print(dgrid[ini] < dthres, dgrid[ini], dthres)
            print(d_grid[nei_idxs])
    ini_filtered = np.array(ini_filtered)
    print("Filtered:", ini_filtered.size)
    m1 = m1[ini_filtered, :]
    m2 = m2[ini_filtered, :]
    dgrid = dgrid[ini_filtered]

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
    return lonm1, latm1, lonm2, latm2, dgrid

