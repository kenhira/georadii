import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy

from astropy.io import fits

from georadii.camera import Camera_arcsix
from georadii.util import read_fits

def get_all_image_array(date, start_time, end_time):

    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    camtool.load_hsk(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))

    # Retrieve all the image files for the specified time period
    fits_list = camtool.load_fits(start_time, end_time, location='ARCSIX2KSS')[::4] # subselecting the images (1 every 4)
    print('%d fits file(s) found' % len(fits_list))

    # Read one fits file
    img, header = read_fits(fits_list[0], flipud=False, fliplr=False)

    # Prepare the array to store all the image data 
    all_image = np.zeros((img.shape[0], img.shape[1], img.shape[2], len(fits_list)), dtype=np.float32)

    # Read and store all the image data within the specified time frame
    for ifits, fits_file in enumerate(fits_list):
        img, header = read_fits(fits_file, flipud=False, fliplr=False)
        all_image[:, :, :, ifits] = np.float32(img)/float(2**16) # between 0 and 1
    
    return all_image

if __name__ == "__main__":

    # Make the directory to store the output pngs
    dirname = 'out_mask_img'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fnum = 1
    dir2name = '%s/%04d' %(dirname, fnum)
    while os.path.exists(dir2name):
        fnum += 1
        dir2name = '%s/%04d' %(dirname, fnum)
    os.makedirs(dir2name)

    ### STEP 1: Remove most of the major feature by thresholding method
    # Select the time range over which the images are averaged
    date       = '2024-05-31'
    start_time = '15:15:00' # in or over cloud
    end_time   = '15:30:00'
    v_thres = 0.24 # Threshold value below which the pixel is obstructed
    n_neighbor = 5 # Number of pixels next to the mask to be eliminated

    # Get all images taken during the specified period
    img_all = get_all_image_array(date, start_time, end_time)

    # Define the mask
    img_in = np.mean(img_all[:, :, 1, :], axis=2) # average out high-frequency features
    mask1 = img_in < v_thres # simple thresholding approach
    mask_arr1 = np.where(mask1, 1., 0.) # pseudo-image to show where the mask is

    # 2D convolution to "extend" the mask to eliminate near-edge pixel that were missed
    kernel = np.ones((2*n_neighbor + 1, 2*n_neighbor + 1))
    mask2 = scipy.signal.convolve2d(mask_arr1, kernel, mode='same', boundary='wrap') > 0.
    mask_arr2 = np.where(mask2, 1., 0.) # pseudo-image to show where the mask is
    
    # mask the original image to show the un-masked portion
    img_masked = np.copy(img_in)
    # img_masked[mask1] = 1.
    img_masked[mask2] = 1.


    # -- Plot -- #
    fig = plt.figure(figsize=(6, 12))

    ax1 = fig.add_subplot(311)
    cq1 = ax1.imshow(img_in)
    ax1.set_title('Input')
    fig.colorbar(cq1, ax=ax1)

    ax2 = fig.add_subplot(312)
    cq2 = ax2.imshow(img_masked)
    ax2.set_title('Masked image')
    fig.colorbar(cq2, ax=ax2)

    ax3 = fig.add_subplot(313)
    cq4 = ax3.imshow(mask_arr1, cmap='Blues', alpha=0.5)
    cq3 = ax3.imshow(mask_arr2, cmap='Reds', alpha=0.5)
    ax3.set_title('Mask (original and %d-pixel extension)' % n_neighbor)

    plt.savefig('%s/img1.png' % (dir2name), dpi=300)


    ### STEP 2: Remove the downward fin and propellers with threshold for mean and std
    # Select the time range over which the images are averaged
    date       = '2024-05-31'
    start_time = '17:00:00' # in or over cloud
    end_time   = '17:15:00'
    v_thres  = 0.44 # Threshold value below which the pixel is obstructed (mean)
    v_thres2 = 0.08 # Threshold value above which the pixel is obstructed (std)
    n_neighbor = 5 # Number of pixels next to the mask to be eliminated

    # Get all images taken during the specified period
    img_all = get_all_image_array(date, start_time, end_time)

    # Define the mask
    img_in = np.mean(img_all[:, :, 1, :], axis=2) # average out high-frequency features
    ixpx, iypx = np.meshgrid(np.arange(img_all.shape[1]), np.arange(img_all.shape[0])) # convenient array with pixel indices
    mask3 = (img_in < v_thres) & (650 < ixpx) & (ixpx < 900) & (1600 < iypx) & (iypx < 1850) # simple thresholding approach
    mask_arr3 = np.where(mask3, 1., 0.) # pseudo-image to show where the mask is
    
    img_in3 = np.std(img_all[:, :, 1, :], axis=2) # locate the relatively varying pixels (for propellers)
    mask5 = (img_in3 > v_thres2) & (((900 < ixpx) & (ixpx < 1400) & (1900 < iypx) & (iypx < 2080)) \
                                | ((500 < ixpx) & (ixpx < 700) & (1250 < iypx) & (iypx < 1550)))
    mask_arr5 = np.where(mask5, 1., 0.) # pseudo-image to show where the mask is

    # 2D convolution to "extend" the mask to eliminate near-edge pixel that were missed
    kernel = np.ones((2*n_neighbor + 1, 2*n_neighbor + 1))
    mask4 = scipy.signal.convolve2d(mask_arr3, kernel, mode='same', boundary='wrap') > 0.
    mask_arr4 = np.where(mask4, 1., 0.) # pseudo-image to show where the mask is
    mask6 = scipy.signal.convolve2d(mask_arr5, kernel, mode='same', boundary='wrap') > 0.
    mask_arr6 = np.where(mask6, 1., 0.) # pseudo-image to show where the mask is
    
    # mask the original image to show the un-masked portion
    img_masked2 = np.copy(img_in)
    img_masked2[mask2] = 1.
    img_masked2[mask4] = 1.
    img_masked2[mask6] = 1.

    mask_arr7 = np.where(mask1 | mask3 | mask5, 1., 0.)
    mask_arr8 = np.where(mask2 | mask4 | mask6, 1., 0.)

    # -- Plot -- #
    fig2 = plt.figure(figsize=(12, 8))

    ax1 = fig2.add_subplot(221)
    cq1 = ax1.imshow(img_in)
    ax1.set_title('Input')
    fig2.colorbar(cq1, ax=ax1)

    ax2 = fig2.add_subplot(222)
    cq2 = ax2.imshow(img_masked2)
    ax2.set_title('Resulting mask')
    fig2.colorbar(cq2, ax=ax2)

    ax3 = fig2.add_subplot(223)
    cq4 = ax3.imshow(mask_arr3, cmap='Blues', alpha=0.5)
    cq3 = ax3.imshow(mask_arr4, cmap='Reds', alpha=0.5)
    cq4 = ax3.imshow(mask_arr5, cmap='Purples', alpha=0.5)
    cq3 = ax3.imshow(mask_arr6, cmap='Oranges', alpha=0.5)
    ax3.set_title('Fin and propellers')

    ax4 = fig2.add_subplot(224)
    cq5 = ax4.imshow(mask_arr7, cmap='Blues', alpha=0.5)
    cq6 = ax4.imshow(mask_arr8, cmap='Reds', alpha=0.5)
    ax4.set_title('Combined mask (original and %d-pixel extension)' % n_neighbor)

    plt.savefig('%s/img2.png' % (dir2name), dpi=300)


    # Save as fits file
    out_fits = True
    # out_fits = False

    img_out = mask_arr8.astype(np.uint8) # the simplest format in 

    if out_fits:
        hdu = fits.PrimaryHDU(data=img_out)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto('pixel_mask_v20241023.fits')

    plt.show()