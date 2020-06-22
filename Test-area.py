#Imports
import access2thematrix
import imageio
from peakdetect import peakdetect
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from skimage.filters import sobel
from scipy import ndimage
from importlib import reload
from skimage import feature
from skimage import exposure
import cv2
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


#Imports of self-written code; see different .py files. Must be reloaded if changed
import ImProcessing
#reload(ImProcessing)
import ImAnalysis
#reload(ImAnalysis)
import Plots
#reload(Plots)

#Initialization of self-written code imports
ip = ImProcessing.ImageProcessing()
ia = ImAnalysis.ImageAnalysis()
plot = Plots.Plot()


#Global
matrx_data = access2thematrix.MtrxData()
#InAsBi
InAsBi = 'images/path/to/data/file_name.extension'
#GaAsBi
GaAsBi = 'images/path/to/data/file_name.extension'
#GaAsN
GaAsN = 'images/path/to/data/file_name.extension'
#Quantum Dot
QD = 'images/path/to/data/file_name.extension'


### Beginning ###
#choose what data is analyzed
data_file = GaAsBi

#order of polynomial fit to rows
order = 0

#cuts off fraction of right side from histogram for segmentation; increasing factor decreases finer details
#POSSIBLE TODO; different thresholds for sides of peak (symmetry afffects outcome slightly)
fraction = 1

#determines accuracy of automatic segmentation; increase for more contrast in histogram
autobins = 1000

#threshold percentage; algorithm looks segments based on percentage of height of global maximum of histogram.
# Values too low can lead to infinite loops. Higher values increase noise. Keep between 0.1 and 0.03
thr_perc1 = 0.07
thr_perc2 = 0.025

### END ###


#Access data, store in 'Z'
traces, message = matrx_data.open(data_file)
im0, message = matrx_data.select_image(traces[0]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
im1, message = matrx_data.select_image(traces[1]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
im2, message = matrx_data.select_image(traces[2]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
im3, message = matrx_data.select_image(traces[3]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
Z = im0.data #select first trace

#Z = cv2.imread(r'C:\Users\s169261\PycharmProjects\Programs\images\BEP data Victor\Quantum dots\default_2020Mar05-091143_STM-STM_Spectroscopy--138_1.png', cv2.IMREAD_GRAYSCALE) #directly from png
#Z = np.mean(Z, axis=2) #way to make grayscale image, not that precise


#plot.sky_plot(Z, "raw")

#plot.im_plot(Z_dsk, "deskewed")
#plane leveling 2nd order:
Z_crop = Z[000:800, 130:800]
Z_sub1 = ip.plane_lvl_order(Z_crop, 1)


#plot.sky_plot(Z_sub1, '1st order plane fit')

#subtract mean plane and polynomial fit to rows
Z_poly1 = ip.poly_row(Z_sub1, 2)
plot.sky_plot(Z_poly1, "Row align, poly, 1st")

Z_poly1_corr = ip.plane_lvl_order(Z_poly1, 2)

# fig, ax = plt.subplots()
# ax.set_xlabel('Pixel value')
# ax.set_ylabel('Amount of values in data set')
# plt.axvline(0, color='r', linestyle='dashed', linewidth=2)
# plt.axvline(40, color='b', linestyle='dashed', linewidth=2)
# plt.axvline(75, color='g', linestyle='dashed', linewidth=2)
# plot.hist_plot(Z_poly1_corr, 255, "Z_final")

# plot.sky_plot(Z_poly1_corr, "Corrected with second plane level")
# readcm = np.load('SkyColormap.npy',allow_pickle='TRUE').item()  #opens colormap file
# sky = LinearSegmentedColormap('sky', readcm)                    #for the colormap used in python
# plt.imsave(r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\GaAsN\GaAsN9_2.png", Z_poly1_corr, cmap=sky)

# all_segments, all_segments_cleaned, segm1_closed, segm2_closed, segm3_closed, segm3 = ia.\
#     hist_segm_man(Z_poly1_corr, 'GaAsBi, regular')

#plot.im_plot(all_segments_cleaned, "Man segm")


# name = r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\Trainingset\file_correcteddata.jpeg"
# plt.imsave(name, Z_poly1_corr)
#
# name2 = r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\Trainingset\file_correcteddata.jpeg"
# Z_photoshop = plt.imread(name2)
# print(Z_photoshop.shape)
# Z_photoshop = np.mean(Z_photoshop, axis=2) #way to make grayscale image, not that precise
# plot.gr_plot(Z_photoshop, "photoshopped")
#
# sharpened_Z = ip.sharpen(Z_poly1_corr)
# edge_Z_poly = sobel(Z_poly1_corr)
# plot.gr_plot(edge_Z_poly, "Sobel Edge detection")
#
# (H, hogImage) = feature.hog(Z_photoshop, orientations=9, pixels_per_cell=(8, 8),
# 	cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1", visualize=True)
#
# feature.hog(Z_photoshop,)
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# plot.gr_plot(hogImage, "HOG")

# mini = imageio.imread(r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\Trainingset\file_6.jpeg")
# mini = np.mean(mini, axis=2) #way to make grayscale image, not that precise
# (H2, hogImage2) = feature.hog(mini, orientations=9, pixels_per_cell=(4, 4),
# 	cells_per_block=(1, 1), transform_sqrt=False, block_norm="L1", visualize=True)
# #feature.hog(mini)
# #hogImage2 = exposure.rescale_intensity(hogImage2, out_range=(0.255))
# plot.gr_plot(hogImage2, "HOG2")

#
# smart_segm, smart_segm_closed, peaks = ia.hist_segm_auto(edge_Z_poly, 0.10, 1000, 1) #gives both cleaned and non-cleaned mask)
# Z_poly_mask1 = np.invert(smart_segm) #method produces negative mask; invert
# label_Z_poly, nb_labels = ndimage.label(Z_poly_mask1)

# smart_segm_ps, smart_segm_closed_ps, peaks_ps = ia.hist_segm_auto(Z_photoshop, 0.1, 200, .95) #gives both cleaned and non-cleaned mask)
# Z_ps_mask1 = np.invert(smart_segm_ps) #method produces negative mask; invert
# label_Z_ps, nb_labels_ps = ndimage.label(Z_ps_mask1)
#
#
#Segmentation based on features
# image_segm_size, image_segm_low_size, image_segm_high_size, image_segm_under, image_segm_over = ia.\
#     feature_segm(Z_poly1_corr, Z_poly_mask1, 'size', factor=1)
# plot.sky_plot(image_segm_size, "size segmented, desired")
# plot.sky_plot(image_segm_low_size, "size segmented, low")
# plot.sky_plot(image_segm_over, "size segmented, over")
# plot.sky_plot(image_segm_under, "size segmented, under")

#
# factor1 = 1
# image_segm_mean_val, image_segm_mean_val_low, image_segm_mean_val_high, image_segm_under, image_segm_over = ia.\
#     feature_segm(image_segm_size, Z_poly_mask1, 'mean value', factor1)
#plot.im_plot(image_segm_mean_val_low, "mean value segmented, low")
#plot.sky_plot(image_segm_mean_val, "mean value segmented, desired")
#plot.sky_plot(image_segm_under, "mean value segmented, under")
#plot.sky_plot(image_segm_mean_val_high, "mean value segmented, high")


# factor2 = 1
# image_segm_mean_val2, image_segm_mean_val_low2, image_segm_mean_val_high2, image_segm_under2, image_segm_over2 =\
#     ia.feature_segm(image_segm_mean_val, Z_poly_mask1, 'mean value', factor1)
# plot.sky_plot(image_segm_mean_val2, "mean value segmented 2, desired")
# plot.sky_plot(image_segm_mean_val_low2, "mean value 2 segmented, low")
# plot.sky_plot(image_segm_under2, "mean value 2 segmented, under mean")
# plot.sky_plot(image_segm_over2, "mean value 2 segmented, over mean")

# factor3 = 1
# image_segm_mean_val3, image_segm_mean_val_low3, image_segm_mean_val_high3, image_segm_under3, image_segm_over3 =\
#     ia.feature_segm(image_segm_mean_val2, Z_poly_mask1, 'mean value', factor1)
# plot.sky_plot(image_segm_mean_val3, "mean value segmented 2, desired")
# plot.sky_plot(image_segm_mean_val_low3, "mean value 2 segmented, low")
# plot.sky_plot(image_segm_under3, "mean value 2 segmented, under mean")
# plot.sky_plot(image_segm_over3, "mean value 2 segmented, over mean")


#
# #Find region of interest enclosing object
# slice_x, slice_y = ndimage.find_objects(label_Z_ps == 5)[0]  # must begin at 2 during initial try; check this
# roi = Z_poly1_corr[slice_x, slice_y]
# name = r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\Trainingset\file_" + str(6) + '.jpeg'
# plt.imsave(name, roi)




#Cycle through all identified ROIs and save them
# i = 1
# for i in range(nb_labels):
# 	#Find region of interest enclosing object
# 	slice_x, slice_y = ndimage.find_objects(label_Z_poly == i+1)[0]  # must begin at 2 during initial try; check this
# 	#print(slice_x, slice_y)
# 	#print(i, 'out of ', nb_labels)
# 	roi = Z[slice_x, slice_y]
# 	x = roi.shape[0]
# 	y = roi.shape[1]
# 	#print(x*y)
# 	if x*y > 0:
# 		plot.im_plot(roi, "roi")
# 		name = r"C:\Users\s169261\Documents\`BEP\Alessandro\Atoms\training InAsBi\file_158_" + str(i) + '.png'
# 		plt.imsave(name, roi)
# 	i += 1
# 	#print(roi.shape)


print("end of script")




