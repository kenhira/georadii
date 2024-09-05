from setuptools import setup

setup(
    name = 'georadii',
    version = '0.0.1',
    description = 'Geolocation and Radiometric conversion toolkit for Airborne Imagery',
    classifiers = [
       'Development Status :: 1 - Planning',
       'Intended Audience :: Science/Research',
       'License :: OSI Approved :: MIT License',
       'Programming Language :: Python :: 3.11',
       'Topic :: Scientific/Engineering :: Atmospheric Science',
       ],
    keywords = 'ARCSIX',
    url = 'https://github.com/kenhira/georadii',
    author = 'Ken Hirata',
    author_email = 'ken.hirata@colorado.edu',
    license = 'MIT',
    packages = ['georadii'],
    install_requires = [
        'numpy',
        'matplotlib',
        'h5py',
        'pysolar',
        ],
    python_requires = '~=3.11',
    include_package_data = True,
    zip_safe = False
    )