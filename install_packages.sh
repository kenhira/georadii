echo "Installing packages..."

echo "Let's install NumPy..."
conda install anaconda::numpy=2.2 -y

echo "Let's install SciPy..."
conda install -c anaconda scipy -y

echo "Let's install Matplotlib..."
conda install -c anaconda matplotlib -y

echo "Let's install Cartopy..."
conda install -c anaconda cartopy -y

echo "Let's install h5py..."
conda install -c anaconda netcdf4 -y

echo "Let's install h5py..."
conda install -c anaconda h5py -y

echo "Let's install astropy..."
conda install -c anaconda astropy -y

echo "Let's install Pandas..."
conda install -c anaconda pandas -y

echo "Let's install Scikit-image..."
conda install -c conda-forge scikit-image -y

echo "Let's install Rasterio..."
conda install -c conda-forge rasterio -y

echo "Let's install pysolar (pip)..."
python -m pip install pysolar

echo "Let's install Open CV (pip)..."
python -m pip install opencv-python

echo "Required packages have been installed."
