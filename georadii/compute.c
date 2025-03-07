#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>  // Required for NumPy
#include <stdlib.h>
#include <math.h>
#include <string.h>  // For memset

// Function to compute 2D histograms with weighted inputs and cyclic boundary conditions
static PyObject* gridding2d_weight(PyObject *self, PyObject *args) {
    PyArrayObject *x_array, *y_array, *weights_array;
    double x_min, x_max, x_bin_size, y_min, y_max, y_bin_size;

    // Parse NumPy array inputs and binning parameters
    if (!PyArg_ParseTuple(args, "O!O!O!dddddd", 
                          &PyArray_Type, &x_array, 
                          &PyArray_Type, &y_array, 
                          &PyArray_Type, &weights_array,
                          &x_min, &x_max, &x_bin_size, 
                          &y_min, &y_max, &y_bin_size)) {
        return NULL;
    }

    // Ensure input arrays are of type double (float64)
    if (PyArray_TYPE(x_array) != NPY_DOUBLE || PyArray_TYPE(y_array) != NPY_DOUBLE || PyArray_TYPE(weights_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be of type float64 (double)");
        return NULL;
    }

    // Get the number of data points (num_points)
    Py_ssize_t num_points = PyArray_SIZE(x_array);
    if (num_points != PyArray_SIZE(y_array)) {
        PyErr_SetString(PyExc_ValueError, "x and y arrays must have the same length");
        return NULL;
    }

    // Check if weights_array is 1D or 2D
    int n_weights;
    int is_1d_weights = (PyArray_NDIM(weights_array) == 1);
    
    if (is_1d_weights) {
        if (PyArray_SIZE(weights_array) != num_points) {
            PyErr_SetString(PyExc_ValueError, "1D weights array must have the same length as x and y arrays");
            return NULL;
        }
        n_weights = 1;  // Treat as single histogram
    } else if (PyArray_NDIM(weights_array) == 2) {
        npy_intp *weights_shape = PyArray_DIMS(weights_array);
        if (weights_shape[0] != num_points) {  // weights shape is (num_points, n_weights)
            PyErr_SetString(PyExc_ValueError, "weights array must have shape (num_points, n_weights)");
            return NULL;
        }
        n_weights = (int)weights_shape[1];  // Extract second dimension as number of histograms
    } else {
        PyErr_SetString(PyExc_ValueError, "weights array must be either 1D (num_points,) or 2D (num_points, n_weights)");
        return NULL;
    }

    // Get raw data pointers
    double *x_data = (double *)PyArray_DATA(x_array);
    double *y_data = (double *)PyArray_DATA(y_array);
    double *weights_data = (double *)PyArray_DATA(weights_array);

    // Compute number of bins
    int x_bins = (int)ceil((x_max - x_min) / x_bin_size);
    int y_bins = (int)ceil((y_max - y_min) / y_bin_size);

    // Determine output shape
    npy_intp dims[3] = {x_bins, y_bins, n_weights};
    npy_intp dims_2d[2] = {x_bins, y_bins};  // If output is 2D
    PyObject *hist_array;

    if (is_1d_weights) {
        hist_array = PyArray_SimpleNew(2, dims_2d, NPY_DOUBLE);
    } else {
        hist_array = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    }

    double *hist_data = (double *)PyArray_DATA((PyArrayObject *)hist_array);

    // Explicitly initialize histogram to zero
    Py_ssize_t total_size = is_1d_weights ? (x_bins * y_bins) : (x_bins * y_bins * n_weights);
    memset(hist_data, 0, total_size * sizeof(double));

    // Separate loops for 1D and 2D weight cases
    if (is_1d_weights) {
        for (Py_ssize_t i = 0; i < num_points; i++) {
            int x_idx = (int)ceil((x_data[i] - x_min) / x_bin_size) - 1;
            int y_idx = (int)ceil((y_data[i] - y_min) / y_bin_size) - 1;

            // Apply cyclic boundary wrapping
            x_idx = (x_idx + x_bins) % x_bins;
            y_idx = (y_idx + y_bins) % y_bins;

            hist_data[x_idx * y_bins + y_idx] += weights_data[i];
        }
    } else {
        for (Py_ssize_t i = 0; i < num_points; i++) {
            int x_idx = (int)ceil((x_data[i] - x_min) / x_bin_size) - 1;
            int y_idx = (int)ceil((y_data[i] - y_min) / y_bin_size) - 1;

            // Apply cyclic boundary wrapping
            x_idx = (x_idx + x_bins) % x_bins;
            y_idx = (y_idx + y_bins) % y_bins;

            for (int w = 0; w < n_weights; w++) {
                hist_data[(x_idx * y_bins + y_idx) * n_weights + w] += weights_data[i * n_weights + w];
            }
        }
    }

    return hist_array;  // Return NumPy array
}

// Function to compute 2D histogram with integer output (each point contributes 1)
static PyObject* gridding2d_count(PyObject *self, PyObject *args) {
    PyArrayObject *x_array, *y_array;
    double x_min, x_max, x_bin_size, y_min, y_max, y_bin_size;

    // Parse NumPy array inputs and binning parameters
    if (!PyArg_ParseTuple(args, "O!O!dddddd", 
                          &PyArray_Type, &x_array, 
                          &PyArray_Type, &y_array, 
                          &x_min, &x_max, &x_bin_size, 
                          &y_min, &y_max, &y_bin_size)) {
        return NULL;
    }

    // Ensure input arrays are of type double (float64)
    if (PyArray_TYPE(x_array) != NPY_DOUBLE || PyArray_TYPE(y_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "x and y arrays must be of type float64 (double)");
        return NULL;
    }

    Py_ssize_t num_points = PyArray_SIZE(x_array);
    double *x_data = (double *)PyArray_DATA(x_array);
    double *y_data = (double *)PyArray_DATA(y_array);

    int x_bins = (int)ceil((x_max - x_min) / x_bin_size);
    int y_bins = (int)ceil((y_max - y_min) / y_bin_size);

    npy_intp dims[2] = {x_bins, y_bins};
    PyObject *hist_array = PyArray_SimpleNew(2, dims, NPY_INT64);
    int64_t *hist_data = (int64_t *)PyArray_DATA((PyArrayObject *)hist_array);

    // Explicitly initialize histogram to zero using memset
    memset(hist_data, 0, x_bins * y_bins * sizeof(int64_t));

    for (Py_ssize_t i = 0; i < num_points; i++) {
        int x_idx = (int)ceil((x_data[i] - x_min) / x_bin_size) - 1;
        int y_idx = (int)ceil((y_data[i] - y_min) / y_bin_size) - 1;

        // Apply cyclic boundary wrapping
        x_idx = (x_idx + x_bins) % x_bins;
        y_idx = (y_idx + y_bins) % y_bins;

        hist_data[x_idx * y_bins + y_idx] += 1;  // Integer addition
    }

    return hist_array;
}

// Register both functions
static PyMethodDef ComputeMethods[] = {
    {"gridding2d_weight", gridding2d_weight, METH_VARARGS, "Computes a weighted 2D histogram"},
    {"gridding2d_count", gridding2d_count, METH_VARARGS, "Computes a 2D histogram with integer counts"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef computemodule = {
    PyModuleDef_HEAD_INIT,
    "compute",
    NULL,
    -1,
    ComputeMethods
};

PyMODINIT_FUNC PyInit_compute(void) {
    import_array();
    return PyModule_Create(&computemodule);
}