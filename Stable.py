#Imports
import access2thematrix
import numpy as np
from skimage.filters import sobel
from scipy import ndimage
from importlib import reload

#Imports of self-written code; see different .py files. Must be reloaded if changed
import ImProcessing
reload(ImProcessing)
import ImAnalysis
reload(ImAnalysis)
import Plots
reload(Plots)

#Initialization of self-written code imports
ip = ImProcessing.ImageProcessing()
ia = ImAnalysis.ImageAnalysis()
plot = Plots.Plot()


#Global
matrx_data = access2thematrix.MtrxData()
#InAsBi
InAsBi = 'images/BEP data Victor/Set 2/InAsBi/default_2018Mar01-090147_STM-STM_Spectroscopy--8_1.Z_mtrx'
#GaAsBi
GaAsBi = 'images/BEP data Victor/default_2019Dec09-090745_STM-STM_Spectroscopy--31_1.Z_mtrx'
#GaAsN
GaAsN = 'images/BEP data Victor/Set 2/GaAsN/default_2016Jun03-093824_STM-STM_Spectroscopy--9_2.Z_mtrx'


### Beginning ###
#choose what data is analyzed
data_file = GaAsBi

#order of polynomial fit to rows
order = 1

#cuts off fraction of right side from histogram for segmentation; increasing factor decreases finer details
#POSSIBLE TODO; different thresholds for sides of peak (symmetry afffects outcome slightly)
fraction = 1

#determines accuracy of automatic segmentation; increase for more contrast in histogram
autobins = 1000

#threshold percentage; algorithm looks segments based on percentage of height of global maximum of histogram.
# Values too low can lead to infinite loops. Higher values increase noise. Keep between 0.1 and 0.03
thr_perc1 = 0.15
thr_perc2 = 0.025

### END ###


#Access data, store in 'Z'
traces, message = matrx_data.open(data_file)
im0, message = matrx_data.select_image(traces[0]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
im1, message = matrx_data.select_image(traces[1]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
im2, message = matrx_data.select_image(traces[2]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
im3, message = matrx_data.select_image(traces[3]) #contains data, width, height, y_offset, x_offset, angle and channel_name_and_unit
Z = im0.data #select first trace
# Z = imageio.imread('C:/Users/s169261/PycharmProjects/Programs/images/BEP data Victor/Set 2/9_1d.png') #directly from png
# Z = np.mean(Z, axis=2) #way to make grayscale image, not that precise




plot.gr_plot(Z, "raw")
#plane leveling 2nd order:
Z_crop = Z[000:800, 130:800]
Z_sub1 = ip.plane_lvl_order(Z_crop, 1)
plot.gr_plot(Z_sub1, '1st order plane fit')

#subtract mean plane and polynomial fit to rows
Z_poly1 = ip.poly_row(Z_sub1, order)
plot.gr_plot(Z_poly1, "Row align, poly, 1st")

Z_poly1_corr = ip.plane_lvl_order(Z_poly1, 2)
plot.gr_plot(Z_poly1_corr, "Corrected with second plane level")

sharpened_Z = ip.sharpen(Z_poly1_corr)
edge_Z_poly = sobel(Z_poly1_corr)
plot.gr_plot(edge_Z_poly, "Sobel Edge detection")



smart_segm, smart_segm_closed, peaks = ia.hist_segm_auto(Z_poly1_corr, thr_perc1, autobins, fraction) #gives both cleaned and non-cleaned mask)
Z_poly_mask1 = np.invert(smart_segm_closed) #method produces negative mask; invert
label_Z_poly, nb_labels = ndimage.label(Z_poly_mask1)


#Segmentation based on features
image_segm_size, image_segm_low_size, image_segm_high_size, image_segm_under, image_segm_over = ia.\
    feature_segm(Z_poly1_corr, Z_poly_mask1, 'size', factor=1)
# Plot().im_plot(image_segm_size, "size segmented, desired")

# factor1 = 1
# image_segm_mean_val, image_segm_mean_val_low, image_segm_mean_val_high, image_segm_under, image_segm_over = ia.\
#     feature_segm(Z_poly1_corr, Z_poly_mask1, 'mean value', factor1)
# #Plot().im_plot(image_segm_mean_val_low, "mean value segmented, low")
# plot.im_plot(image_segm_mean_val, "mean value segmented, desired")
# plot.im_plot(image_segm_under, "mean value segmented, under")
# #Plot().im_plot(image_segm_mean_val_high, "mean value segmented, high")
#
# factor2 = 0.5
# image_segm_mean_val2, image_segm_mean_val_low2, image_segm_mean_val_high2, image_segm_under2, image_segm_over2 =\
#     ia.feature_segm(image_segm_mean_val, Z_poly_mask1, 'mean value', factor2)
# plot.im_plot(image_segm_mean_val2, "mean value segmented 2, desired")
# plot.im_plot(image_segm_mean_val_low2, "mean value 2 segmented, low")
# plot.im_plot(image_segm_under2, "mean value 2 segmented, under mean")
# plot.im_plot(image_segm_over2, "mean value 2 segmented, over mean")
#


#
# #Find region of interest enclosing object
# slice_x, slice_y = ndimage.find_objects(label_Z_poly == 0)[0]
# print(slice_x,slice_y)
# roi = Z_poly[slice_x, slice_y]
# plt.imshow(roi)
# plt.title("Chosen ROI")
# plt.show()


print("end of script")




