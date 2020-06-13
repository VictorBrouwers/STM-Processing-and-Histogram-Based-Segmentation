from peakdetect import peakdetect
import numpy as np
from scipy import ndimage
import numpy.ma as ma
import matplotlib.pyplot as plt
from importlib import reload

#Import self-written code; see .py files. Reload should be used if plots is changed
import Plots
#reload(Plots)

#initialize plots
plot = Plots.Plot()


#This class contains methods for (processed) image analysis
class ImageAnalysis:
    # segments data by using a histogram, requires manual estimation of values based on histogram.
    # requires and ndarray (data)
    # outputs ndarrays (all_segments, all_segments_cleaned) and masks (segmx_closed)
    # SOURCE: (https://github.com/bnsreenu/python_for_microscopists/blob/master/023-histogram_segmentation_using_scikit_image.py)
    def hist_segm_man(self, data, type):
        # Data parameters
        nx = data.shape[0]
        ny = data.shape[1]

        # histogram plot for visual, manual segmentation
        plot.hist_plot(data, 255, "Manual hist")

        # Normalize values in data between 0 and 255
        data_min = np.amin(data)
        data = data - data_min * np.ones((nx, ny))
        data_max = np.amax(data)
        data = data / data_max * 255

        # Regions in image are separated based on these parameters of ndarray values; manual
        if type == 'GaAsBi, edge':
            #parameters, GaAsBi, edge
            segm1 = (data >= 0) & (data <= 22)  # red
            segm2 = (data > 110)  # green
            segm3 = (data > 22) & (data <= 110)  # blue
            segm4 = (data <= 65) #yellow

        elif type == 'GaAsBi, regular':
            #parameters, GaAsBi, regular, 0th order row fit
            segm1 = (data >= 0) & (data <= 45)  # red
            segm2 = (data > 75)  # green
            segm3 = (data > 40) & (data <= 75)  # blue
            #segm4 = (data <= 65)  # yellow

        elif type == 'InAsBi, edge':
            #parameters InAsBi, edge, 0th order row fit
            segm1 = (data > 00) & (data <= 30)  # red
            segm2 = (data > 255)  # green
            segm3 = (data > 35) & (data <= 255)  # blue

        elif type == 'InAsBi, regular':
            #parameters InAsBi, manual, 0th order row fit
            segm1 = (data > 0) & (data <= 83)  # red
            segm2 = (data > 255)  # green
            segm3 = (data > 83) & (data <= 255)  # blue

        elif type == 'GaAsN, edge':
            #GaAsN, edge, 0th order row fit
            segm1 = (data >= 0) & (data <= 11)  # red
            segm2 = (data > 30)  # green
            segm3 = (data > 11) & (data <= 30)  # blue

        elif type == 'GaAsN, regular':
            # GaAsN, regular, 0th order row fit
            segm1 = (data >= 0) & (data <= 20)  # red
            segm2 = (data > 35)  # green
            segm3 = (data > 20) & (data <= 35)  # blue

        # Create empty array and fill with segments
        all_segments = np.zeros((nx, ny, 3))

        all_segments[segm1] = (1, 0, 0)  # red
        all_segments[segm2] = (0, 1, 0)  # green
        all_segments[segm3] = (0, 0, 1)  # blue
        # all_segments[segm4] = (1,1,0) #yellow

        # Clean data; opening takes care of isolated pixels within window, closing of isolated holes within window
        segm1_opened = ndimage.binary_opening(segm1, np.ones((3, 3)))
        segm1_closed = ndimage.binary_closing(segm1_opened, np.ones((3, 3)))

        segm2_opened = ndimage.binary_opening(segm2, np.ones((3, 3)))
        segm2_closed = ndimage.binary_closing(segm2_opened, np.ones((3, 3)))

        segm3_opened = ndimage.binary_opening(segm3, np.ones((3, 3)))
        segm3_closed = ndimage.binary_closing(segm3_opened, np.ones((3, 3)))

        # Create empty array and fill with cleaned segments
        all_segments_cleaned = np.zeros((nx, ny, 3))

        all_segments_cleaned[segm1_closed] = (1, 0, 0)  # red
        all_segments_cleaned[segm2_closed] = (0, 1, 0)  # green
        all_segments_cleaned[segm3_closed] = (0, 0, 1)  # blue

        return all_segments, all_segments_cleaned, segm1_closed, segm2_closed, segm3_closed, segm3 #TODO; put closed segments in object

    #Automatically segments data using a histogram. Paramaters are globally defined.
    #Requires an ndarray (data)
    #Outputs masks and ndarray (smart_segm, smart_segm_closed) and list with minima and maxima (peaks)
    def hist_segm_auto(self, data, thr_perc, autobins, fraction):
        # Data parameters
        nx = data.shape[0]
        ny = data.shape[1]

        #note; setting bins to 255 gives a histogram corresponding to 255 intervals. Higher values are more accurate
        hist, bin_edges = np.histogram(data, bins=autobins) #autobins is global var

        # Normalize values in data between 0 and the highest value of bins
        data_min = np.amin(data)
        data = data - data_min * np.ones((nx, ny))
        data_max = np.amax(data)
        data = data / data_max * autobins

        # segmenting data based on peaks
        bins = [i for i in range(hist.shape[0])] #match amount of bins to hist_array for peakdetect

        # peaks with peakdetect
        peaks = peakdetect(hist, bins, lookahead=1)  # lookahead determines how far ahead peakdetect looks for peaks
        minima = peaks[1]
        maxima = peaks[0]
        nb_min = len(minima)  # number of minima
        #print(nb_min)

        #smarter guess segmentation
        max_vals = [] #empty array that will store maxima values
        max_x_vals = []#stores x-values corresponding to maxima values
        #find all maxima and put them in array
        for i in range(len(peaks[0])):
            max_vals.append(maxima[i][1])

        #Store minimum y-values in array
        min_vals = []
        for i in range(len(peaks[1])):
            min_vals.append(minima[i][1])

        gl_max = np.amax(max_vals) #find global maximum
        position = np.where(max_vals == gl_max)[0][0] #find index global maximum
        threshold = thr_perc * gl_max #used to circumvent local maxima in main peak, thr_perc is global var

        #when peak is at edge, enforce no change
        if position == 0:
            x_left = 0
        else:
            x_left = minima[position-1][0]

        y_left = minima[position-1][1]

        #check is minimum is sufficiently low, keep lowering until threshold (global var) is reached
        position_left = position - 1 #necessary to avoid changing position value when entering right loop later on
        while y_left > threshold:
            if position_left == 0:
                x_left = 0
                print("x_left at edge")
                break
            else:
                x_left = minima[position_left][0]
                y_left = minima[position_left][1]
                position_left -= 1 #cycle through new positions until edge

        # when peak is at edge, enforce no change
        if position == nb_min:
            x_right = 255
        else:
            x_right = minima[position+1][0]

        y_right = minima[position+1][1]
        #check is minimum is sufficiently low
        position_right = position + 1
        while y_right > threshold:
            if position_right == nb_min:
                x_right = 255
                print("x_right at edge")
                break
            else:
                x_right = minima[position_right][0]
                y_right = minima[position_right][1]
                position_right += 1 #cycle through positions until edge


        print('x_left: ', x_left,'x_right: ', x_right)
        x_right = fraction * x_right #this multiplication decreases the fine details, fraction is global var
        smart_segm = (data >= x_left) & (data <= x_right)
        smart_segm_opened = ndimage.binary_opening(smart_segm)
        smart_segm_closed = ndimage.binary_closing(smart_segm_opened)

        #display the smart mask
        try_smart = np.zeros((nx, ny, 3))
        try_smart[smart_segm] = (1, 0, 0)
        plot.im_plot(try_smart, "Mask")

        fig, ax = plt.subplots()
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Amount of values in data set')
        plt.axvline(x_left, color='r', linestyle='dashed', linewidth=2)
        plt.axvline(x_right, color='r', linestyle='dashed', linewidth=2)
        plot.hist_plot(data, autobins, "Smart mask hist")

        return smart_segm, smart_segm_closed, peaks

    #segments data based on indicated feature of ROIs (right now: size, mean value). The ROIs must be displayed by the mask
    #requires ndarray (data), ndarray (mask) and string of intended feature [see first if-statement] (feature)
    #outputs an ndarray of the original data, filtered by the features (desired, high and low)
    def feature_segm(self, data, mask, feature, factor):

        label_data, nb_labels = ndimage.label(mask)

        # Compute ROIs bases on indicated feature
        # (SOURCE: https://scipy-lectures.org/advanced/image_processing/#feature-extraction)
        if feature == 'size':
            #print('feature is: ', feature)
            segm_feature = ndimage.sum(mask, label_data, range(nb_labels + 1))  # +1 required since full image is stored, too
        if feature == 'mean value':
            #print('feature is: ', feature)
            segm_feature = ndimage.mean(data, label_data, range(1, nb_labels + 1)) #+1 required since full image is stored, too

        #delete certain entries if necessary (0th entry could be full image instead of ROI)
        if nb_labels == segm_feature.shape[0]:
            rois_feature = segm_feature
            #print('kept first entry')
        else:
            rois_feature = np.delete(segm_feature, [0])  #cut out full image
            #print('deleted first entry')

        mask = np.logical_or(rois_feature == rois_feature.max(0, keepdims=1),
                             rois_feature == rois_feature.min(0, keepdims=1))  # remove min, max value
        feature_masked = ma.masked_array(rois_feature, mask=mask)  # mask with min, max removed

        #compute mean size and sd for segmentation
        mean_feature = np.mean(feature_masked, axis=0)
        sd_feature = np.std(feature_masked, axis=0)
        print("mean", feature, ": ",  mean_feature, "sd: ", sd_feature)

        if feature == 'size':
            # print('feature is: ', feature)
            bins = int(round(np.amax(feature_masked))) #specify amount of bins for histogram
        if feature == 'mean value':
            # print('feature is: ', feature)
            #bins = nb_labels #specify amount of bins for histogram. TODO: find representative value/good value
            bins = 500

        #histogram plot of the chosen feature. x-axis currently meaningless, dummy variable
        #TODO; include mean value and standard deviation in histogram, check if correct

        # max = np.amax(feature_masked)
        # mean2 = mean_feature / max * bins
        # sd = sd_feature / max * bins
        # print('adjusted mean: ', mean2,'adjusted sd: ', sd)
        # plt.axvline(mean2, color='r', linestyle='dashed', linewidth=2)
        # plt.axvline(mean2 + sd, color='g', linestyle='dashed', linewidth=2)
        # plt.axvline(mean2 - sd, color='g', linestyle='dashed', linewidth=2)
        plot.hist_plot(feature_masked, bins, "histogram: " + feature)

        #segment sizes based on std
        factor #specifies range of values TODO: make global variable?
        segm_low = (feature_masked <= (mean_feature - factor*sd_feature))
        segm_desired = (feature_masked > (mean_feature - factor*sd_feature)) & \
                       (feature_masked <= (mean_feature + factor*sd_feature))
        segm_high = (feature_masked > (mean_feature + factor*sd_feature))

        segm_over = feature_masked > mean_feature
        segm_under = feature_masked < mean_feature

        #reassign labels
        labels = np.unique(label_data)
        label_data = np.searchsorted(labels, label_data)  # orders ROIs from 1 to until nb_labels (0 is full image)

        #Create array with desired, low and high data based on mask using method im_segm
        image_segm_des = ImageAnalysis().im_segm(data, label_data, segm_desired)
        image_segm_low = ImageAnalysis().im_segm(data, label_data, segm_low)
        image_segm_high = ImageAnalysis().im_segm(data, label_data, segm_high)
        image_segm_under = ImageAnalysis().im_segm(data, label_data, segm_under)
        image_segm_over = ImageAnalysis().im_segm(data, label_data, segm_over)

        return image_segm_des, image_segm_low, image_segm_high, image_segm_under, image_segm_over

    #Separates a dataset in segments based on its peaks
    #Requires and ndarray (data) and an object (peaks) containing maxima and minima
    #Outputs a list with intervals, an object with segmented data and an object with cleaned segmented data
    #TODO: verify whether it works (maybe data has to be normalized to hist/bins), maybe useful for automatic data sep
    def peak_segm(self, data, peaks):
        #unwrap peaks
        minima = peaks[1]
        maxima = peaks[0]
        nb_min = len(minima)

        #Store the intervals in IntervalList and create segments, based on intervals, in all_segm
        interval_list = []
        all_segm = []
        all_segm_cleaned = []
        #print('minima: ', minima)
        for i in range(nb_min):
            # Include interval of 0-[first peak], make sure loop does not fail in last iteration
            if i == 0:
                x1 = 0
                x2 = minima[i][0]
            elif (i > 0) & (i < (nb_min - 1)):
                x1 = minima[i][0]
                x2 = minima[i + 1][0]
            elif i == (nb_min - 1):
                x1 = minima[i][0]
                x2 = 255

            # Store intervals
            interval = [x1, x2]
            interval_list.append(interval)

            # Create segments based on intervals
            segm = (data > x1) & (data <= x2)
            all_segm.append(segm)

            # Clean random pixels
            # COMMENT; cleaning these data nullifies the result as there are many masks, depending on lookahead
            segm_opened = ndimage.binary_opening(segm, np.ones((3, 3)))
            segm_closed = ndimage.binary_closing(segm_opened, np.ones((3, 3)))
            all_segm_cleaned.append(segm_closed)

        return interval_list, all_segm, all_segm_cleaned

    #Creates an array with data of specific feature, indicated by the segment mask
    #Requires ndarray (data), a label of this data (label_data) and an masked ndarry (segm_mask)
    #Outpus the original data, only with the mask data in it TODO: make independent of slices
    def im_segm(self, data, label_data, segm_mask):
        nx = data.shape[0]
        ny = data.shape[1]

        # filter total data based on feature mask
        slices = ndimage.find_objects(label_data)
        slices_arr = np.array(slices)
        slices_masked = slices_arr[segm_mask]

        # create array of entire image for visualization, compile all ROIs. Columns store slices.
        # TODO: this method may cause inaccuracies since it takes slices
        image_segm = np.zeros((nx, ny))  # empty array, filled with roi-slices
        for i in range(slices_masked.shape[0]):
            slice_x = slices_masked[i][0]
            slice_y = slices_masked[i][1]
            roi = data[slice_x, slice_y]  # obtain slice of i-th roi
            #Check if roi is small enough (sometimes entire image is passed as ROI)
            nroix = roi.shape[0]
            nroiy = roi.shape[1]
            if nroix < int(0.5 * nx):
                image_segm[slice_x, slice_y] = roi  # paste i-th roi in array


        return image_segm

