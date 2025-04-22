import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from georadii.camera import Camera_arcsix
from georadii.util import read_fits, makedir_numbered
from matplotlib import rcParams


if __name__ == "__main__":
    # Select the time range for which the image files are going to be loaded
    # date       = '2024-08-15'
    # start_time = '15:39:00'
    # end_time   = '15:39:30'
    # date       = '2024-06-11'
    # start_time = '13:08:48'
    # end_time   = '13:08:55'
    date       = '2024-05-31'
    start_time = '15:52:00'
    end_time   = '16:12:00'
    # date       = '2024-06-05'
    # start_time = '12:43:00'
    # end_time   = '12:44:59'
    # start_time = '13:16:00'
    # end_time   = '13:16:32'
    # start_time = '13:16:00'
    # end_time   = '13:16:40'
    # start_time = '13:15:23'
    # end_time   = '13:17:22'
    # start_time = '14:16:01'
    # end_time   = '14:16:05'
    # start_time = '14:42:00'
    # end_time   = '15:02:00'
    # start_time = '15:40:00'
    # end_time   = '15:41:00'
    # start_time = '16:16:01'
    # end_time   = '16:16:30'
    # date       = '2024-06-06'
    # start_time = '16:16:00' ; end_time = '16:17:00'
    # # start_time = '16:41:46' ; end_time = '16:47:26'

    # Make the directory to store the output pngs
    dirname = 'out_movie'
    dir2name = makedir_numbered(dirname)

    # Instantiate the camera toolkit for the specified date
    camtool = Camera_arcsix(date)

    # Load housekeeping file needed for identifying the aircraft status
    camtool.load_aircraft(location='%s/Downloads/ARCSIX_HSK/' % (os.getenv('HOME')))

    # Retrieve all the image files for the specified time period
    fits_list = camtool.load_fits(start_time, end_time, location='argus')[::1]

    # Setup writer
    writer = animation.FFMpegWriter(fps=30)

    speedup = 10
    # speedup = 16
    # speedup = 32

    fig, ax = plt.subplots(figsize=(6, 6))

    with writer.saving(fig, os.path.join(dir2name, "fits_movie.mp4"), dpi=200):
        rad_geom1, t_act1 = camtool.rad_and_geom_from_fits(fits_list[0], mask_aircraft_shadow=False)#, saturation_val=0.9*2**16)
        lat1 = rad_geom1['aircraft_status']['lat']
        lon1 = rad_geom1['aircraft_status']['lon']
        alt1 = int(round(rad_geom1['aircraft_status']['alt']))

        impl = ax.imshow(rad_geom1['data'], cmap='gray', vmin=0, vmax=2**16)

        for ifits, fits_file in enumerate(fits_list[1:]):
            # Extract the image file and the image metadata
            rad_geom2, t_act2 = camtool.rad_and_geom_from_fits(fits_file, mask_aircraft_shadow=False)#, saturation_val=0.9*2**16)
            dt = (t_act2 - t_act1).total_seconds()

            ax.clear()
            fig.patch.set_facecolor('black')
            impl = ax.imshow(rad_geom1['data'], cmap='gray', vmin=0, vmax=2**16)
            non_zero_indices = np.argwhere(rad_geom1['data'][:, :, 1] > 0)
            min_y, min_x = non_zero_indices.min(axis=0)
            max_y, max_x = non_zero_indices.max(axis=0)
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')
            ax.axis('off')
            rcParams['text.usetex'] = False

            ax.text(0.01, 0.99, t_act1.strftime("%Y/%m/%d\n%H:%M:%S UTC"), 
                    transform=ax.transAxes, color='white', fontsize=10.5, 
                    ha='left', va='top', family='Menlo', fontweight='bold')
            lat_dir = 'N' if lat1 >= 0 else 'S'
            lon_dir = 'E' if lon1 >= 0 else 'W'
            ax.text(0.99, 0.99, "%8.3f° %s\n%8.3f° %s\n%7d m" % (abs(lat1), lat_dir, abs(lon1), lon_dir, alt1), 
                    transform=ax.transAxes, color='white', fontsize=11.5, 
                    ha='right', va='top', family='Menlo', fontweight='bold')
            n_repeats = int(dt*writer.fps/speedup)
            for _ in range(n_repeats):
                writer.grab_frame()

            # plt.imshow(rad_geom['data'], cmap='gray', vmin=0, vmax=2**16)
            # plt.colorbar()
            # plt.title('%s %s' % (date, t_act))
            # plt.show()

            rad_geom1 = rad_geom2.copy()
            t_act1 = t_act2
            lat1 = rad_geom1['aircraft_status']['lat']
            lon1 = rad_geom1['aircraft_status']['lon']
            alt1 = rad_geom1['aircraft_status']['alt']
