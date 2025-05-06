import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import pickle
import h5py
from scipy.interpolate import interp1d
from pygeodesy.geoids import GeoidPGM
from astropy.time import Time
import glob

from georadii.util import makedir_numbered, get_solar_angles_multi

def read_lvis_cam_alt(file_path, freq=1):
    pickle_fn = os.path.splitext(file_path)[0] + '_alt.pkl'
    if os.path.exists(pickle_fn):
        with open(pickle_fn, 'rb') as f:
            print('Loading from %s' % pickle_fn)
            time, lat, lon, alt, roll, pitch, head = pickle.load(f)
    else:
        with open(file_path, 'r') as f:
            foundheader = 0
            ln = 0
            for line in f:
                ln += 1
                if line.startswith('CorrTime'):
                    headerfields = line.split()[:]
                    foundheader = 1
                    break
            if not foundheader:
                message = 'Error: No header found in %s' % file_path
                raise OSError(message)
        
        header_ln = ln + 1
        data = np.loadtxt(file_path, skiprows=header_ln)

        week = data[:, 0]
        sec = data[:, 1]
        lat = data[:, 2]*180./np.pi
        lon = data[:, 3]*180./np.pi
        alt = data[:, 4]
        roll = data[:, 14]*180./np.pi
        pitch = data[:, 15]*180./np.pi
        head = data[:, 16]*180./np.pi

        # Adjust roll, pitch, and heading offsets for P-3 MetNav
        pitch +=  0.19 # deg
        roll  += -0.09 # deg
        head  +=  0.35 # deg

        head = (head + 360) % 360

        gps_epoch = datetime.datetime(1980, 1, 6)
        time = np.array([np.datetime64(gps_epoch) + np.timedelta64(int(w) * 7 * 24 * 60 * 60, 's') + np.timedelta64(int(s), 's') + np.timedelta64(int((s - int(s)) * 1e6), 'us') for w, s in zip(week, sec)])

        with open(pickle_fn, 'wb') as f:
            pickle.dump((time, lat, lon, alt, roll, pitch, head), f)
            print('Saved to %s' % pickle_fn)

    return time[::freq], lat[::freq], lon[::freq], alt[::freq], roll[::freq], pitch[::freq], head[::freq]

def gen_hsk_lvis(file_path, fname_h5):
    lvis_time_offset = datetime.timedelta(seconds=18) # GPS time offset w.r.t UTC time

    lvis_time, lvis_lat, lvis_lon, lvis_alt, lvis_roll, lvis_pitch, lvis_head = read_lvis_cam_alt(file_path, freq=100)

    lvis_time = lvis_time - np.timedelta64(lvis_time_offset, 's')

    sttm = np.datetime64(lvis_time[0].astype('datetime64[s]'))
    entm = np.datetime64(lvis_time[-1].astype('datetime64[s]'))
    time_arr = np.arange(sttm, entm, datetime.timedelta(seconds=0.2), dtype='datetime64[us]')

    tmhr_lvis = (lvis_time - np.datetime64(lvis_time[0], 'D')) / np.timedelta64(1, 'h')
    tmhr = ((time_arr - np.datetime64(lvis_time[0], 'D')) / np.timedelta64(1, 'h')).astype(float)

    lvis_l1 = interp1d(tmhr_lvis, lvis_lat, kind='linear', bounds_error=False, fill_value="nan")(tmhr)
    lvis_l2 = interp1d(tmhr_lvis, lvis_lon, kind='linear', bounds_error=False, fill_value="nan")(tmhr)
    lvis_a = interp1d(tmhr_lvis, lvis_alt, kind='linear', bounds_error=False, fill_value="nan")(tmhr)
    lvis_r = interp1d(tmhr_lvis, lvis_roll, kind='linear', bounds_error=False, fill_value="nan")(tmhr)
    lvis_p = interp1d(tmhr_lvis, lvis_pitch, kind='linear', bounds_error=False, fill_value="nan")(tmhr)
    lvis_h = interp1d(tmhr_lvis, lvis_head, kind='linear', bounds_error=False, fill_value="nan")(tmhr)

    egm08 = GeoidPGM('%s/Downloads/lvis/cam/geoids/egm2008-1.pgm' % (os.getenv('HOME')), kind=3)
    geoid_heights = egm08.height(lvis_l1, lvis_l2)
    lvis_a_orthometric = lvis_a - geoid_heights

    jday = (time_arr - np.datetime64('0001-01-01T00:00:00')) / np.timedelta64(1, 'D') + 1.

    sza, saa = get_solar_angles_multi(time_arr, lvis_l1, lvis_l2, lvis_a)

    # Save to HDF5 file
    f = h5py.File(fname_h5, 'w')
    f['lat'] = lvis_l1
    f['lon'] = lvis_l2
    f['alt'] = lvis_a_orthometric
    f['tmhr'] = tmhr
    f['ang_rol'] = lvis_r
    f['ang_pit'] = lvis_p
    f['ang_hed'] = lvis_h
    f['jday'] = jday
    f['sza']  = sza
    f['saa']  = saa
    f.close()

if __name__ == "__main__":
    # dates = ['2024-05-31', '2024-06-05',]
    # dates = ['2024-06-06',]
    dates = [
        '2024-05-17', '2024-05-21', '2024-05-23', '2024-05-24', '2024-05-28',
        '2024-05-30', '2024-05-31', '2024-06-03', '2024-06-05', '2024-06-06',
        '2024-06-07', '2024-06-10', '2024-06-11', '2024-06-13', '2024-07-08',
        '2024-07-09', '2024-07-22', '2024-07-25', '2024-07-29', '2024-07-30',
        '2024-08-01', '2024-08-02', '2024-08-07', '2024-08-08', '2024-08-09',
        '2024-08-15', '2024-08-16',
    ]
    get_hsk = True

    dirname = 'out_hsk_lvis'
    dir2name = makedir_numbered(dirname)

    for date in dates:
        # Convert the date into modified Julian date
        mjd = Time(date).mjd
        print(f"Modified Julian Date: {mjd}")
        file_string = '%s/Downloads/lvis/cam/traj%d_p241108_LVISF_*_PPP_GUI_LC_610i_j102_20Hz_v3_Canon.txt' % (os.getenv('HOME'), int(mjd))
        file_list = glob.glob(file_string)
        if len(file_list) == 1:
            file_path = file_list[0]
        else:
            print("%d files found matching the pattern." % len(file_list))
            continue

        fname_h5 = os.path.join(dir2name, "ARCSIX-HSK_P3B_%s_fromLVIS.h5" % (date.replace('-', '')))

        gen_hsk_lvis(file_path, fname_h5)

        fname_h5 = os.path.join(dir2name, "ARCSIX-HSK_P3B_%s_fromLVIS.h5" % (date.replace('-', '')))
        with h5py.File(fname_h5, 'r') as f:
            lat = f['lat'][:]
            lon = f['lon'][:]
            alt = f['alt'][:]
            tmhr = f['tmhr'][:]
            jday = f['jday'][:]
            ang_rol = f['ang_rol'][:]
            ang_pit = f['ang_pit'][:]
            ang_hed = f['ang_hed'][:]
            sza = f['sza'][:]
            saa = f['saa'][:]
        
        fname_h5 = "%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_%s_v0.h5" % (os.getenv('HOME'), date.replace('-', ''))

        if get_hsk:
            with h5py.File(fname_h5, 'r') as f:
                lat_2 = f['lat'][:]
                lon_2 = f['lon'][:]
                alt_2 = f['alt'][:]
                tmhr_2 = f['tmhr'][:]
                jday_2 = f['jday'][:]
                ang_rol_2 = f['ang_rol'][:]
                ang_pit_2 = f['ang_pit'][:]
                ang_hed_2 = f['ang_hed'][:]
                sza_2 = f['sza'][:]
                saa_2 = f['saa'][:]

        fig, ax = plt.subplots(3, 2, figsize=(12, 8))

        ax[0, 0].plot(tmhr, lat, label='Latitude (LVIS)', color='blue')
        if get_hsk:
            ax[0, 0].plot(tmhr_2, lat_2, label='Latitude (MetNav)', linestyle='--')
        ax[0, 0].set_xlabel('Time (hours)')
        ax[0, 0].set_ylabel('Latitude (deg)')
        ax[0, 0].set_title('Latitude vs Time')
        ax[0, 0].grid()
        ax[0, 0].legend()

        ax[0, 1].plot(tmhr, lon, label='Longitude (LVIS)', color='orange')
        if get_hsk:
            ax[0, 1].plot(tmhr_2, lon_2, label='Longitude (MetNav)', linestyle='--', color='red')
        ax[0, 1].set_xlabel('Time (hours)')
        ax[0, 1].set_ylabel('Longitude (deg)')
        ax[0, 1].set_title('Longitude vs Time')
        ax[0, 1].grid()
        ax[0, 1].legend()

        ax[1, 0].plot(tmhr, alt, label='Altitude (LVIS)', color='green')
        if get_hsk:
            ax[1, 0].plot(tmhr_2, alt_2, label='Altitude (MetNav)', linestyle='--', color='lightgreen')
        ax[1, 0].set_xlabel('Time (hours)')
        ax[1, 0].set_ylabel('Altitude (m)')
        ax[1, 0].set_title('Altitude vs Time')
        ax[1, 0].grid()
        ax[1, 0].legend()

        ax[1, 1].plot(tmhr, ang_rol, label='Roll (LVIS)', color='red')
        ax[1, 1].plot(tmhr, ang_pit, label='Pitch (LVIS)', color='purple')
        ax[1, 1].plot(tmhr, ang_hed, label='Heading (LVIS)', color='brown')
        if get_hsk:
            ax[1, 1].plot(tmhr_2, ang_rol_2, label='Roll (MetNav)', linestyle='--', color='pink')
            ax[1, 1].plot(tmhr_2, ang_pit_2, label='Pitch (MetNav)', linestyle='--', color='violet')
            ax[1, 1].plot(tmhr_2, ang_hed_2, label='Heading (MetNav)', linestyle='--', color='black')
        ax[1, 1].set_xlabel('Time (hours)')
        ax[1, 1].set_ylabel('Angle (deg)')
        ax[1, 1].set_title('Angles vs Time')
        ax[1, 1].grid()
        ax[1, 1].legend()

        ax[2, 0].plot(tmhr, sza, label='Solar Zenith Angle (LVIS)', color='blue')
        if get_hsk:
            ax[2, 0].plot(tmhr_2, sza_2, label='Solar Zenith Angle (MetNav)', linestyle='--', color='lightblue')
        ax[2, 0].set_xlabel('Time (hours)')
        ax[2, 0].set_ylabel('Angle (deg)')
        ax[2, 0].set_title('Solar Zenith Angle vs Time')
        ax[2, 0].grid()
        ax[2, 0].legend()

        ax[2, 1].plot(tmhr, saa, label='Solar Azimuth Angle (LVIS)', color='cyan')
        if get_hsk:
            ax[2, 1].plot(tmhr_2, saa_2, label='Solar Azimuth Angle (MetNav)', linestyle='--', color='blue')
        ax[2, 1].set_xlabel('Time (hours)')
        ax[2, 1].set_ylabel('Angle (deg)')
        ax[2, 1].set_title('Solar Azimuth Angle vs Time')
        ax[2, 1].grid()
        ax[2, 1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(dir2name, '%s_hsk.png' % (date.replace('-', ''))), dpi=300)
