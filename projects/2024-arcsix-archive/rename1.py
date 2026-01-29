"""
Code to generate archive files - bundled version
"""

import os
import glob
import shutil
import tarfile
import zipfile

directory = '../v20251021'
# directory = '.'
# directory = 'out_gridding'

nc_dir_base = './data_all'
os.makedirs(nc_dir_base, exist_ok=True)

all_subdirectories = glob.glob(os.path.join(directory, 'rf*'))
# all_subdirectories = glob.glob(os.path.join(directory, 'rf01_*'))
# all_subdirectories = glob.glob(os.path.join(directory, 'rf01_20240528_00'))
# all_subdirectories = glob.glob(os.path.join(directory, '00*'))

strf = 1
enrf = 1

all_subdirectories = [
        subdir for subdir in all_subdirectories \
        if strf <= int(os.path.basename(subdir)[2:4]) <= enrf
        ]

for subdir in sorted(all_subdirectories):
    all_targz = sorted(glob.glob(os.path.join(subdir, '*.tar.gz')))
    if not all_targz:
        continue
    
    date = os.path.basename(all_targz[0]).split('_')[2]
    print('Date:', date)
    
    png_dir = os.path.join(nc_dir_base, date + '_png') 
    os.makedirs(png_dir, exist_ok=True)

    png_file_all = sorted(glob.glob(os.path.join(subdir, '*.png')))
    for png_file in png_file_all:
        st_time = os.path.basename(png_file).split('_')[1]
        png_file_newbase = 'ARCSIX-Imagery-Nadir_P3B_%s%s_R0_quicklook.png' % (date, st_time)
        shutil.copy2(png_file, os.path.join(png_dir, png_file_newbase))

    nc_dir = os.path.join(nc_dir_base, date + '_nc')
    os.makedirs(nc_dir, exist_ok=True)
    for targz in all_targz:
        base = os.path.basename(targz)
        with tarfile.open(targz, "r:gz") as tar:
            tar.extractall(path=nc_dir)
    nc_file_all = sorted(glob.glob(os.path.join(nc_dir, '*/*.nc')))
    for nc_file in nc_file_all:
        st_time = os.path.basename(nc_file).split('_')[3]
        nc_file_newbase = 'ARCSIX-Imagery-Nadir_P3B_%s%s_R0.nc' % (date, st_time)
        shutil.move(nc_file, os.path.join(nc_dir, nc_file_newbase))

    for old_dir in glob.glob(os.path.join(nc_dir, 'gridded_nc*')):
        os.rmdir(old_dir)

    gb_all = []
    for inc, nc_file in enumerate(sorted(glob.glob(os.path.join(nc_dir, '*.nc')))):
        size_bytes = os.path.getsize(nc_file)
        size_gb = size_bytes / (1024 * 1024 * 1024)
        gb_all.append(size_gb)
    
    # threshold = 2.45 # GB
    threshold = 4.95 # GB
    group_idx = []
    current = []
    current_sum = 0.0
    for i, size in enumerate(gb_all):
        if current_sum + size <= threshold:
            current.append(i)
            current_sum += size
        else:
            group_idx.append(current)
            print('Group size (GB):', current_sum)
            current = [i]
            current_sum = size
    if current:
        group_idx.append(current)
        print('Group size (GB):', current_sum)
    
    n_groups = len(group_idx)
    print('Number of groups:', n_groups)


    for ig in range(n_groups):
        png_file_in_group = sorted(glob.glob(os.path.join(png_dir, '*.png')))[group_idx[ig][0]:group_idx[ig][-1]+1]
        nc_file_in_group = sorted(glob.glob(os.path.join(nc_dir, '*.nc')))[group_idx[ig][0]:group_idx[ig][-1]+1]
        st_time_png = os.path.basename(png_file_in_group[0]).split('_')[2][8:]
        st_time_nc = os.path.basename(nc_file_in_group[0]).split('_')[2][8:]
        assert st_time_png == st_time_nc, 'Start times do not match between png and nc files, %s vs %s' % (st_time_png, st_time_nc)
        new_zip_path = 'ARCSIX-Imagery-Nadir_P3B_%s%s_R0.zip' % (date, st_time_png)
        print('Writing to a zip file:', new_zip_path)
        with zipfile.ZipFile(new_zip_path, "w", zipfile.ZIP_DEFLATED) as zp:
            print('png...')
            for png_file in png_file_in_group:
                zp.write(png_file, arcname=os.path.basename(png_file))
            print('nc...')
            for nc_file in nc_file_in_group:
                zp.write(nc_file, arcname=os.path.basename(nc_file))
