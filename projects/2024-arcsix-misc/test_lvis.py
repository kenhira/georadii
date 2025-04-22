import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from matplotlib.ticker import FixedLocator
import time

import netCDF4 as nc

from georadii.georadii import Georadii
from georadii.lvis import LVIS_arcsix

from cartopy.crs import Projection, Mercator
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom

if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded
    # date       = '2024-05-31'
    # # start_time = '15:54:00' ; end_time = '15:57:00'
    # # start_time = '16:34:00' ; end_time = '16:36:00'
    # start_time = '16:34:00' ; end_time = '16:34:10'
    # # start_time = '16:16:00' ; end_time = '16:17:00'
    # # start_time = '16:41:46' ; end_time = '16:47:26'

    # date       = '2024-05-31'
    date       = '2024-06-05'
    # start_time = '12:43:00'
    # end_time   = '12:44:59'
    # start_time = '13:16:00'
    # end_time   = '13:16:32'
    # start_time = '13:16:00'
    # end_time   = '13:16:40'
    start_time = '13:15:23'
    end_time   = '13:15:43'
    # start_time = '14:07:01'
    # end_time   = '14:07:30'
    # start_time = '15:40:00'
    # end_time   = '15:40:30'
    # start_time = '11:00:00'
    # end_time   = '17:00:00'

    all_time = False

    # Instantiate the camera toolkit for the specified date
    lvis_tool = LVIS_arcsix(date)

    lvis_geom, lvis_meta = lvis_tool.load_lvis_l2(start_time, end_time, location='/Users/kehi6101/Downloads/lvis/l2/')

    lat_arr = lvis_meta['latgeo']
    lon_arr = lvis_meta['longeo']
    out_arr = lvis_geom['data'][:, 0, 1]
    cbarlabl = 'LVIS max-amp altitude (m)'
    # lat_arr = lvis_meta['latgeo']
    # lon_arr = lvis_meta['longeo']
    # out_arr = lvis_geom['data'][:, 0, 2] - lvis_geom['data'][:, 0, 0]
    # cbarlabl = 'LVIS alt max - min (m)'
    # lat_arr = lvis_meta['latgeo']
    # lon_arr = lvis_meta['longeo']
    # # out_arr = lvis_meta['incident']
    # # out_arr = lvis_meta['range']
    # out_arr = lvis_meta['range']*np.cos(np.deg2rad(lvis_meta['incident']))
    # cbarlabl = 'LVIS range (m)'

    if all_time:
        shot_height = lvis_meta['range']*np.cos(np.deg2rad(lvis_meta['incident']))
        lvis_time = lvis_meta['time']

        # Plot shot_height against time
        plt.figure(figsize=(10, 5))
        plt.scatter(lvis_time, shot_height, marker='o', linestyle='-', s=2, color='blue')
        plt.xlabel('Time (seconds since start)')
        plt.ylabel('Shot Height (m)')
        plt.title('Shot Height vs Time')
        plt.grid(True)
        plt.show()

    else:
        xcenter, ycenter = np.mean(lvis_meta['longeo']), np.mean(lvis_meta['latgeo'])
        xmin, xmax = np.nanmin(lvis_meta['longeo']), np.nanmax(lvis_meta['longeo'])
        ymin, ymax = np.nanmin(lvis_meta['latgeo']), np.nanmax(lvis_meta['latgeo'])

        # Scatter plot of LVIS data
        cartopy_proj = ccrs.Orthographic(central_longitude=xcenter, central_latitude=ycenter,)
        fig  = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection=cartopy_proj)

        sc = ax.scatter(
            # lvis_meta['longeo'], lvis_meta['latgeo'], 
            # c=lvis_geom['data'][:, 0, 1], 
            lon_arr, lat_arr,
            c=out_arr,
            cmap='viridis', 
            s=4, 
            transform=ccrs.PlateCarree()
        )
        g1 = ax.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 180.1, 0.2*10.**(np.round(np.log10(np.abs(xmax - xmin))))))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 0.2*10.**(np.round(np.log10(np.abs(ymax - ymin))))))
        g1.right_labels = False
        g1.top_labels = False
        
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
        cbar.set_label(cbarlabl)
        
        plt.show()