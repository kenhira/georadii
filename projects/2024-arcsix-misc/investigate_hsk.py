import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
import time
import netCDF4 as nc
import bz2

from georadii.georadii import Georadii
from georadii.camera import Camera_arcsix
from georadii.util import read_fits, calc_viewing_angles, write_surface_grid_to_nc, makedir_numbered, calc_bearing

if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded
    # date       = '2024-08-15'
    # start_time = '14:20:00'
    # end_time   = '14:21:30'
    # date       = '2024-06-11'
    # start_time = '13:08:48'
    # end_time   = '13:08:55'
    date       = '2024-06-05'
    # start_time = '12:43:00'
    # end_time   = '12:44:59'
    # start_time = '13:16:00'
    # end_time   = '13:16:32'
    # start_time = '13:15:23'
    # end_time   = '13:17:22'
    # start_time = '14:16:01'
    # end_time   = '14:16:30'
    start_time = '11:00:00'
    end_time   = '19:00:00'

    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    # camtool.load_aircraft(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')), kind='HSK')
    camtool.load_aircraft(location='%s/Downloads/ARCSIX_MetNav/' % (os.getenv('HOME')), kind='MetNav')

    # Convert start and end times into decimal hours
    start_time_parts = list(map(int, start_time.split(':')))
    start_time_decimal = start_time_parts[0] + start_time_parts[1] / 60 + start_time_parts[2] / 3600
    end_time_parts = list(map(int, end_time.split(':')))
    end_time_decimal = end_time_parts[0] + end_time_parts[1] / 60 + end_time_parts[2] / 3600

    print(f"Start time in decimal hours: {start_time_decimal}")
    print(f"End time in decimal hours: {end_time_decimal}")

    # Plot camtool.lats and camtool.lons against camtool.hrss in two separate panels
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)

    # Plot latitudes vs hrss
    mask = (camtool.hrss >= start_time_decimal) & (camtool.hrss <= end_time_decimal)

    avg_lat = np.mean(camtool.lats[mask])
    avg_lon = np.mean(camtool.lons[mask])
    avg_alt = np.mean(camtool.alts[mask])

    print(f"Average Latitude: {avg_lat}")
    print(f"Average Longitude: {avg_lon}")
    print(f"Average Altitude: {avg_alt}")
    
    ax1.plot(camtool.hrss[mask], camtool.lats[mask], 'b-', label='Latitude')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Latitude', color='b')
    ax1.set_title('Latitude and Longitude vs Time')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(camtool.hrss[mask], camtool.lons[mask], 'r-', label='Longitude')
    ax2.set_ylabel('Longitude', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Plot altitude vs time in the second panel
    ax3 = fig.add_subplot(312)
    ax3.plot(camtool.hrss[mask], camtool.alts[mask], 'k-', label='Altitude')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Altitude (m)', color='k')
    ax3.tick_params(axis='y', labelcolor='k')
    ax3.legend(loc='upper left')

    # Plot pitch, roll, and heading vs time in the third panel
    ax4 = fig.add_subplot(313)
    ax4.plot(camtool.hrss[mask], camtool.pits[mask], 'g-', label='Pitch')
    ax4.plot(camtool.hrss[mask], camtool.rols[mask], 'm-', label='Roll')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Pitch/Roll (Degrees)', color='g')
    ax4.tick_params(axis='y', labelcolor='g')
    ax4.legend(loc='upper left')
    ax4.set_title('Pitch, Roll, and Heading vs Time')

    ax5 = ax4.twinx()
    ax5.plot(camtool.hrss[mask], camtool.heds[mask], 'c-', label='Heading')
    ax5.set_ylabel('Heading (Degrees)', color='c')
    ax5.tick_params(axis='y', labelcolor='c')
    ax5.legend(loc='upper right')

    plt.tight_layout()
    plt.show()