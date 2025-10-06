import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
from georadii.util import read_hsk_arcsix
from georadii.georadii import Georadii


def leg_finder(targets_dict, consecutive=10):
    """
    targets_dict: dict of {array: threshold}, e.g. {'pit': (arr1, 5.), 'rol': (arr2, 5.)}
    All criteria must be satisfied (all arrays < thresholds) for a point to be considered in a leg.
    """
    keys = list(targets_dict.keys())
    arrays = [targets_dict[k][0] for k in keys]
    thresholds = [targets_dict[k][1] for k in keys]
    # Build mask where all criteria are met
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr, thresh in zip(arrays, thresholds):
        mask &= (arr < thresh)
    group_indices = np.zeros_like(mask, dtype=int)
    group_number = 0
    in_group = False
    count = 0
    for i in range(len(mask)):
        if mask[i]:
            count += 1
            if count >= consecutive:
                if not in_group:
                    group_number += 1
                    in_group = True
                group_indices[i - count + 1:i + 1] = group_number
        else:
            if in_group:
                in_group = False
            count = 0
    return group_indices

if __name__ == "__main__":

    # date = '2024-07-29'
    date = '2024-07-30'

    # start_time = '13:52:00'
    # end_time   = '17:12:00'
    start_time = '11:00:00'
    # start_time = '12:25:00'
    # end_time   = '17:11:30'
    end_time   = '18:00:00'

    # nearby_toggle = True 
    nearby_toggle = False
    # lon_target, lat_target = -57.8, 83.645
    lon_target, lat_target = -68, 84.07

    # hsk_filename = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240531_v0.h5' % (os.getenv('HOME'))
    # hsk_filename = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240605_v0.h5' % (os.getenv('HOME'))
    # hsk_filename = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240606_v0.h5' % (os.getenv('HOME'))
    # hsk_filename = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_20240729_v0.h5' % (os.getenv('HOME'))
    hsk_filename = '%s/Downloads/ARCSIX_HSK/ARCSIX-HSK_P3B_%s_v0.h5' % (os.getenv('HOME'), date.replace('-', ''))
    hsk_data = read_hsk_arcsix(hsk_filename)

    tmhr = hsk_data['hrs'] # decimal hours
    start_hour, start_minute, start_second = map(int, start_time.split(':'))
    end_hour, end_minute, end_second = map(int, end_time.split(':'))
    start_decimal = start_hour + start_minute / 60 + start_second / 3600
    end_decimal = end_hour + end_minute / 60 + end_second / 3600
    time_mask = (tmhr >= start_decimal) & (tmhr <= end_decimal)
    flight_tmhr = tmhr[time_mask]
    flight_lon = hsk_data['lon'][time_mask]
    flight_lat = hsk_data['lat'][time_mask]
    flight_alt = hsk_data['alt'][time_mask]
    flight_pit = hsk_data['pit'][time_mask]
    flight_rol = hsk_data['rol'][time_mask]
    flight_hed = hsk_data['hed'][time_mask]

    sqrt_pit_rol = np.sqrt(flight_pit**2 + flight_rol**2)
    ascend_rate = np.gradient(flight_alt) / np.gradient(flight_tmhr * 3600.)

    leg_finder_targets = {
        'tilt': (sqrt_pit_rol, 5.),
        # 'ascend': (np.abs(ascend_rate), 3.),
    }

    in_leg = leg_finder(leg_finder_targets, consecutive=15)

    if nearby_toggle:
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        target_coords = (lat_target, lon_target)
        nearby_indices = []
        for i in range(len(flight_lat)):
            current_coords = (flight_lat[i], flight_lon[i])
            _, _, distance = geod.inv(lon_target, lat_target, current_coords[1], current_coords[0])
            distance /= 1000.  # Convert from meters to kilometers
            if distance <= 50.:
                nearby_indices.append(i)
        nearby_indices = np.array(nearby_indices)

        in_leg[~np.isin(np.arange(len(in_leg)), nearby_indices)] = 0

    ### Print Out
    leg_times = []
    for leg in np.unique(in_leg[in_leg > 0]):
        leg_mask = in_leg == leg
        leg_start_time = flight_tmhr[leg_mask][0]
        leg_end_time = flight_tmhr[leg_mask][-1]
        leg_duration = leg_end_time - leg_start_time
        leg_avg_alt = np.mean(flight_alt[leg_mask])
        leg_times.append((leg, leg_start_time, leg_end_time, leg_duration, leg_avg_alt))

    for i, (leg, start, end, duration, avg_alt) in enumerate(leg_times):
        start_h = int(start)
        start_m = int((start - start_h) * 60)
        start_s = int(((start - start_h) * 60 - start_m) * 60)
        end_h = int(end)
        end_m = int((end - end_h) * 60)
        end_s = int(((end - end_h) * 60 - end_m) * 60)
        duration_m = duration * 60
        print("Leg %2d: start= %02d:%02d:%02d, end= %02d:%02d:%02d, dur= %5.1f min, avgalt= %6.1f m" % 
              (leg, start_h, start_m, start_s, end_h, end_m, end_s, duration_m, avg_alt))
    
    for i, (leg, start, end, duration, avg_alt) in enumerate(leg_times):
        start_h = int(start)
        start_m = int((start - start_h) * 60)
        start_s = int(((start - start_h) * 60 - start_m) * 60)
        end_h = int(end)
        end_m = int((end - end_h) * 60)
        end_s = int(((end - end_h) * 60 - end_m) * 60)
        print("# start_time = '%02d:%02d:%02d' ; end_time = '%02d:%02d:%02d'" % (start_h, start_m, start_s, end_h, end_m, end_s,))

    exs_all, eys_all = Georadii.ease2_coordinates(flight_lon, flight_lat)
    exs_min, exs_max = np.min(exs_all), np.max(exs_all)
    eys_min, eys_max = np.min(eys_all), np.max(eys_all)
    print("EASE2 coordinates: x_min= %6.1f, x_max= %6.1f, y_min= %6.1f, y_max= %6.1f" % (exs_min, exs_max, eys_min, eys_max))

    ### Plot
    var_plot = in_leg
    # var_plot = flight_alt
    cartopy_proj = ccrs.Orthographic(central_longitude=np.mean(flight_lon), central_latitude=np.mean(flight_lat))
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': cartopy_proj})
    ax.plot(flight_lon, flight_lat, color='grey', label='Full Trajectory', transform=ccrs.PlateCarree(), zorder=10)
    scatter = ax.scatter(flight_lon[in_leg > 0], flight_lat[in_leg > 0], c=var_plot[in_leg > 0], s=4, 
                            cmap='jet_r', transform=ccrs.PlateCarree(), zorder=20)
    
    # Add leg numbers to the starting points of the legs
    for leg in np.unique(in_leg[in_leg > 0]):
        leg_mask = in_leg == leg
        start_lon = flight_lon[leg_mask][0]
        start_lat = flight_lat[leg_mask][0]
        ax.text(start_lon, start_lat, str(leg), color='black', fontsize=12, weight='bold', 
                transform=ccrs.PlateCarree(), zorder=30)

    g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
    g1.xlocator = FixedLocator(np.arange(-180, 180.1, 2.0))
    g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.5))
    g1.top_labels = False
    g1.right_labels = False
    ax.set_title('Legs')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.coastlines()
    ax.grid(True)

    # Set the extent of the ax panel
    ax.set_extent([np.min(flight_lon[in_leg > 0]), np.max(flight_lon[in_leg > 0]), np.min(flight_lat[in_leg > 0]), np.max(flight_lat[in_leg > 0])], crs=ccrs.PlateCarree())

    # Plot time vs altitude with colored legs
    fig, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(flight_tmhr, flight_alt, color='grey', label='Altitude', zorder=10)
    scatter2 = ax2.scatter(flight_tmhr[in_leg > 0], flight_alt[in_leg > 0], c=var_plot[in_leg > 0], s=4, cmap='jet_r', zorder=20)
    
    # Add leg numbers to the starting points of the legs
    for leg in np.unique(in_leg[in_leg > 0]):
        leg_mask = in_leg == leg
        start_time = flight_tmhr[leg_mask][0]
        start_alt = flight_alt[leg_mask][0]
        ax2.text(start_time, start_alt, str(leg), color='black', fontsize=12, weight='bold', zorder=30)
    
    ax2.set_ylim(0., None)
    ax2.set_title('Time vs Altitude with Legs')
    ax2.set_xlabel('Time (decimal hours)')
    ax2.set_ylabel('Altitude (m)')
    ax2.legend()

    plt.show()