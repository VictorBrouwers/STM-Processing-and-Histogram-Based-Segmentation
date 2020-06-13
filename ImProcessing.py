import numpy as np
from scipy import ndimage
import scipy.linalg

#This class contains different kinds of methods for image processing to refine data
class ImageProcessing(object):

    #sharpens the image by a Gaussian filter
    #requires an ndarray (data), outputs ndarray
    #(SOURCE: https://scipy-lectures.org/advanced/image_processing/#feature-extraction)
    def sharpen(self, data):
        blurred_data= ndimage.gaussian_filter(data, 3)
        filter_blurred_data = ndimage.gaussian_filter(blurred_data, 1)
        alpha = 30
        sharpened_data = blurred_data + alpha * (blurred_data - filter_blurred_data)
        return sharpened_data

    # refine by subtracting mean value from rows. Equal to poly_row of 0th order
    # requires an ndarray (data), outputs ndarray
    # will be faster than poly_row for large data sets due to more efficient memory allocation
    #(SOURCE: https://bic-berkeley.github.io/psych-214-fall-2016/subtract_means.html)
    def mean_row(self, data):
        # Data parameters
        nx = data.shape[0]
        ny = data.shape[1]

        row_means = np.mean(data, axis=1)
        row_means_col_vec = row_means.reshape((nx, 1))
        data_demeaned = data - row_means_col_vec
        return data_demeaned

    #refine image by subtracting polynomial fit from rows. Equal to mean value subtraction for 0th order
    #requires an ndarray (data) and the polynomial order of the fit as integer (order)
    #outputs ndarray
    #will be slower than mean_row for large data sets due to the inefficient for-loop TODO; optimize process
    def poly_row(self, data, order):
        data_pol = data.copy() #array to be filled with corrrected rows

        #data parameters
        nx = data.shape[0]
        ny = data.shape[1]
        for i in range(data.shape[0]): #subtracts polynomial fit of given order from every row and fills data_pol
            poly_coef = np.polyfit(range(0, ny), data[i], order)
            ffit = np.poly1d(poly_coef)
            data_pol[i] = data[i] - ffit(data[i])
        return data_pol

    # Mean plan subtraction of given order
    # requires an ndarray (data), and an integer to indicate which order is required (order)
    # outputs an ndarray with subtracted mean plane of given order
    # (SOURCE: http://inversionlabs.com/2016/03/21/best-fit-surfaces-for-3-dimensional-data.html)
    def plane_lvl_order(self, data, order):
        ny = data.shape[0] #Data parameters must be chaned for some reason. TODO; find out why
        nx = data.shape[1]
        X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
        XX = X.flatten()
        YY = Y.flatten()

        XXYY = np.c_[XX, YY]
        data_re = np.reshape(data, (nx * ny, 1))

        if order == 1:
            # best-fit linear plane
            A = np.c_[XX, YY, np.ones(data_re.shape[0])]
            C, _, _, _ = scipy.linalg.lstsq(A, data_re)  # coefficients
            print("1: ", C.shape, A.shape)

            # or expressed using matrix/vector product
            Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

        elif order == 2:
            # best-fit quadratic curve
            A = np.c_[np.ones(data_re.shape[0]), XXYY, np.prod(XXYY, axis=1), XXYY ** 2]
            C, _, _, _ = scipy.linalg.lstsq(A, data_re) #coefficients
            print("2: ",C.shape, A.shape)

            # evaluate it on a grid
            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

        Z_sub = data - Z

        return Z_sub

