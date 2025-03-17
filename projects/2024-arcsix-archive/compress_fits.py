import os
import glob

if __name__ == "__main__":
    
    # file_location = '/Volumes/ARCSIX2KSS/TransitFlight/*/*.fits'
    # id_flt = '20240524_tran-flt-01'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF01_2024_05_28/Camera/*/*.fits'
    # id_flt = '20240528_sci-flt-01'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF02_2024_05_30/Camera/*/*.fits'
    # id_flt = '20240530_sci-flt-02'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF03_2024_05_31/Camera/*/*.fits'
    # id_flt = '20240531_sci-flt-03'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF04_2024_06_03/Camera/*/*.fits'
    # id_flt = '20240603_sci-flt-04'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF05_2024_06_05/Camera/*/*.fits'
    # id_flt = '20240605_sci-flt-05'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF06_2024_06_06/Camera/*/*.fits'
    # id_flt = '20240606_sci-flt-06'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF07_2024_06_07/Camera/*/*.fits'
    # id_flt = '20240607_sci-flt-07'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF08_2024_06_10/Camera/*/*.fits'
    # id_flt = '20240610_sci-flt-08'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF09_2024_06_11/Camera/*/*.fits'
    # id_flt = '20240611_sci-flt-09'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF10_2024_06_13/Camera/*/*.fits'
    # id_flt = '20240613_sci-flt-10'
    file_location = '/Volumes/ARCSIX2KSS/ARCSIX_TransitReturnBangor_2024-06-17/Camera/*/*.fits'
    id_flt = '20240617_tran-flt-02'

    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_SummerTransitToPituffik/Camera/*/*.fits'
    # id_flt = '20240722_tran-flt-03'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF11_2024_07_25/Camera/*/*.fits'
    # id_flt = '20240725_sci-flt-11'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF12_2024_07_29/Camera/*.fits'
    # id_flt = '20240729_sci-flt-12'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF13_2024_07_30/Camera/*.fits'
    # id_flt = '20240730_sci-flt-13'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF14_2024_08_01/Camera/*/*.fits'
    # id_flt = '20240801_sci-flt-14'
    # file_location = '/Volumes/ARCSIX2KSS/ARCSIX_RF15_2024_08_02/Camera/*/*.fits'
    # id_flt = '20240802_sci-flt-15'
    # file_location = '/Volumes/ARCSIX4KSS/ARCSIX_RF16_2024_08_07/Camera/*/*.fits'
    # id_flt = '20240807_sci-flt-16'
    # file_location = '/Volumes/ARCSIX4KSS/ARCSIX_RF17_2024_08_08/Camera/*/*.fits'
    # id_flt = '20240808_sci-flt-17'
    # file_location = '/Volumes/ARCSIX4KSS/ARCSIX_RF18_2024_08_09/Camera/*/*.fits'
    # id_flt = '20240809_sci-flt-18'
    # file_location = '/Volumes/ARCSIX4KSS/ARCSIX_RF19_2024_08_15/Camera/*/*.fits'
    # id_flt = '20240815_sci-flt-19'
    # file_location = '/Volumes/ARCSIX4KSS/ARCSIX_SummerTransit_BGTL_BGR_2024_08_16/Camera/*/*.fits'
    # id_flt = '20240816_tran-flt-04'

    # comp_type = 'bz2'
    comp_type = 'gz'

    compress_level = 6
    # compress_level = 9

    if comp_type == 'bz2':
        import bz2
        ext = '.bz2'
    elif comp_type == 'gz':
        import gzip
        ext = '.gz'

    # Save location
    save_location = '/Volumes/ARCSIX4KSS/compress/%s/nac/fits' % (id_flt)
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    # Retrieve all the image files for the specified time period
    fits_list = glob.glob(file_location)

    nfits = len(fits_list)
    for ifits, fits_file in enumerate(fits_list):
        fits_filename = os.path.basename(fits_file)
        comp_filename = os.path.join(save_location, fits_filename + ext)
        print('Processing %d/%d: %s' % (ifits + 1, nfits, fits_filename))

        # Compress the netCDF file using bzip2
        with open(fits_file, 'rb') as f_in:
            if comp_type == 'bz2':
                with bz2.open(comp_filename, 'wb', compresslevel=compress_level) as f_out:
                    f_out.writelines(f_in)
            elif comp_type == 'gz':
                with gzip.open(comp_filename, 'wb', compresslevel=compress_level) as f_out:
                    f_out.writelines(f_in)