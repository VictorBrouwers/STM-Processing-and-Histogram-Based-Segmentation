import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

readcm = np.load('SkyColormap.npy',allow_pickle='TRUE').item()  #opens colormap file
sky = LinearSegmentedColormap('sky', readcm)                    #for the colormap used in python



#This class contains different kinds of plots
class Plot(object):
    #plots in gray scale
    #requires an ndarray (data) and a title (title)
    #outputs nothing
    def gr_plot(self, data, title):
        plt.figure()
        plt.imshow(data, cmap="gray")
        plt.title(title)
        plt.show()
        return 0

    #Regular image plot
    def im_plot(self, data, title):
        plt.figure()
        plt.imshow(data)
        plt.title(title)
        plt.show()
        return 0

    #Histogram plot
    def hist_plot(self, data, bins, title):
        # TODO; find way to include proper mean, sd
        hist, bin_edges = np.histogram(data, bins=bins)
        #plt.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        plt.plot(hist)
        plt.title(title)
        plt.show()
        return 0

    #plots in SkyColormap.npy style
    def sky_plot(self, data, title):
        plt.figure()
        plt.imshow(data, cmap=sky)
        plt.title(title)
        plt.show()
        return 0

