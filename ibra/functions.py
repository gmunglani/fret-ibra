# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting, logging, data output, fitting etc
"""

import numpy as np
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import mpl_toolkits.mplot3d.axes3d as p3
import logging
from timeit import default_timer as timer
import h5py
from skimage.external.tifffile import TiffWriter
import os
from scipy.optimize import curve_fit
from sklearn import linear_model
from scipy import ndimage
from loess import loess_1d

rcParams['font.family'] = 'serif'

# Create animation of background subtraction
def background_animation(verbose,stack,work_out_path,frange):
    """Background subtraction result per frame video"""
    def data(i, stack, line):
        ax1.clear()
        line1 = ax1.plot_surface(X1, Y1, stack.im_origf[:, :, i], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax1.set_title("{} Frame: {}".format(stack.val.capitalize(), frange[i] + 1))
        ax1.set_zlim(0, np.amax(stack.im_origf))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.grid(False)

        ax2.clear()
        minmax = np.ptp(np.ravel(stack.im_backf[:,:,i]))
        line2 = ax2.plot_surface(stack.X, stack.Y, stack.im_backf[:, :, i], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax2.set_title("Min to Max (Background): {}".format(minmax))
        ax2.set_zlim(0, np.amax(stack.im_origf))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.grid(False)

        ax3.clear()
        line3 = ax3.plot_surface(X1, Y1, stack.im_framef[i, :, :], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax3.set_title("Background Subtracted Image")
        ax3.set_zlim(0, np.amax(stack.im_origf))
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.grid(False)

        ax4.clear()
        signal = stack.labelsf[:,i]
        signal[signal > 0] = 0
        psignal = -np.float32(np.sum(signal))/np.float32(stack.labelsf[:,i].size)
        ax4.set_title("Percentage of Tiles with Signal: %0.2f" % (psignal))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_zlim(0, 1)
        ax4.set_xticks([0,0.5,1])
        ax4.set_yticks([0,0.5,1])
        ax4.set_zticks([0,0.5,1])
        ax4.grid(False)
        ax4.set_xlabel('Variance', labelpad=-1)
        ax4.set_ylabel('Skewness', labelpad=-1)
        ax4.set_zlabel('Median', labelpad=-1)
        ax4.tick_params(axis="x", direction="out", pad=-2)
        ax4.tick_params(axis="y", direction="out", pad=-2)
        ax4.tick_params(axis="z", direction="out", pad=-2)
        varn = stack.propf[:,:,i]
        xyz = varn[stack.maskf[:,i]]
        xyz2 = varn[[not i for i in stack.maskf[:,i]]]
        line4 = ax4.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:, 3], c='red')
        line4 = ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 3], c='blue', s=40)

        line = [line1, line2, line3, line4]
        return line,

    # Start time
    time_start = timer()

    # Define figures, axis and initialize
    rcParams.update({'font.size': 8})
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax2 = fig.add_subplot(2,2,2,projection='3d')
    ax3 = fig.add_subplot(2,2,3,projection='3d')
    ax4 = fig.add_subplot(2,2,4,projection='3d')

    ax1.set_title("{} Frame: {}".format(stack.val, 0))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(False)

    ax2.set_title("Min to Max (Background): {}".format(0))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(False)

    ax3.set_title("Background Subtracted Image")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.grid(False)

    ax4.grid(False)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.set_title("Percentage of Tiles with Signal: {}".format(0))

    ax1.view_init(elev=15., azim=30.)
    ax2.view_init(elev=15., azim=30.)
    ax3.view_init(elev=15., azim=30.)
    ax4.view_init(elev=30., azim=230.)

    # Define grid for tiled image
    X1, Y1 = np.int16(np.meshgrid(np.arange(stack.siz2), np.arange(stack.siz1)))

    line1 = ax1.plot_surface(X1,Y1,stack.im_origf[:,:,0],cmap=cm.bwr)
    line2 = ax2.plot_surface(stack.X,stack.Y,stack.im_backf[:,:,0],cmap=cm.bwr)
    line3 = ax3.plot_surface(X1,Y1,stack.im_framef[0,:,:],cmap=cm.bwr)
    line4 = ax4.scatter(0.5, 0.5, 0.5, c='red')

    line = [line1, line2, line3, line4]

    # Set up animation
    anim = animation.FuncAnimation(fig, data, fargs=(stack,line),frames=np.arange(len(frange)), interval=20, blit=False, repeat_delay=1000)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']

    # Write animation file
    if (max(np.ediff1d(frange,to_begin=frange[0])) > 1):
        fname = work_out_path + '_' + stack.val + '_specific'
        num = 1
        while (os.path.isfile(fname+str(num)+'.avi')):
            num += 1
        else:
            anim.save(fname+str(num)+'.avi', writer=Writer(fps=2))
    else:
        anim.save(work_out_path + '_' + stack.val + '_frames' + str(frange[0]+1) + '_' + str(frange[-1]+1) + '.avi', writer=Writer(fps=2))

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start)+1)
    if (verbose):
        print((stack.val.capitalize() +" (Background Animation) Time: " + time_elapsed + " second(s)"))


def logit(path):
    """Logging data"""
    logger = logging.getLogger('back')
    hdlr = logging.FileHandler(path+'.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(20)

    return logger


def h5(data,val,path,frange):
    """Saving the image stack as a .h5 file"""
    f = h5py.File(path, 'a')

    if val in f:
        # Open existing dataset
        orig = f[val]

        if val in ('acceptor','acceptori','acceptorb'):
            orange = f.attrs['acceptor_frange']
        elif val in ('donor', 'donori', 'donorb'):
            orange = f.attrs['donor_frange']
        else:
            orange = f.attrs['ratio_frange']

        # Create dictionaries of new and existing data
        orig_dict = dict(zip(orange, orig))
        new_dict = dict(zip(frange, data))

        # Save and re-write data in the dictionary
        for key in frange:
            orig_dict[key] = new_dict[key]

        # Sort frames by increasing frame number
        orig_dict_sorted = sorted(orig_dict.items())
        res_range, res = list(zip(*orig_dict_sorted))
        res = np.array(res)

        # Delete existing HDF5 dataset
        if (val in f):
            del f[val]

    else:
        # If no stack is present, create it
        res = np.array(data)
        res_range = frange

    # Save the image pixel data and frange
    if val in ('acceptor','donor'):
        f.create_dataset(val, data=res, shape=res.shape, dtype=np.uint16, compression='gzip')
        f.attrs[val + '_frange'] = res_range
    elif val == 'ratio':
        f.create_dataset(val, data=res, shape=res.shape, dtype=np.uint8, compression='gzip')
        f.attrs[val + '_frange'] = res_range
    else:
        f.create_dataset(val, data=res, shape=res.shape, dtype=np.float16, compression='gzip')

    # Close dataset
    f.close()

def time_evolution(acceptor,donor,work_out_path,name,ylabel,h5_save):
    """Median channel intensity per frame"""
    acceptor_plot = sorted(acceptor.items())
    xa, ya = list(zip(*acceptor_plot))
    xplot = [x + 1 for x in xa]

    # Sort frames for plotting
    donor_plot = sorted(donor.items())
    _, yd = list(zip(*donor_plot))

    vals = ['acceptori','donori','acceptornz','donornz']
    if (ylabel == 'Median Intensity/Bit Depth'):
        names = vals[:2]
        dec = 1
    elif (ylabel == 'Foreground/Total Image Pixels'):
        names = vals[2:]
        dec = 2

    if (h5_save):
        # Convert to arrays
        ya = np.array(ya)
        yd = np.array(yd)

        # Open HDF5 dataset
        f = h5py.File(work_out_path+'_ratio_back.h5', 'a')

        # Save dictionary data in HDF5 dataset
        if (names[0] in f):
            del f[names[0]]
        f.create_dataset(names[0], data=ya, shape=ya.shape, dtype=np.uint16, compression='gzip')

        if (names[1] in f):
            del f[names[1]]
        f.create_dataset(names[1], data=yd, shape=yd.shape, dtype=np.uint16, compression='gzip')
        f.close()


    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(xplot,ya,c=(0.62745098, 0.152941176, 0.498039216),marker='*')
    ax.plot(xplot,yd,c=(1,0.517647059,0),marker='*')

    plt.ylabel(ylabel,labelpad=15, fontsize=22)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=dec))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    plt.yticks(fontsize=18)
    plt.xlabel('Frame Number',labelpad=15, fontsize=22)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=18)

    plt.legend(['Acceptor','Donor'],fancybox=None,fontsize=18)
    plt.savefig(work_out_path + name, bbox_inches='tight')


def block(data,size):
    """Reshape image stack for faster processing"""
    return (data.reshape(data.shape[0] // size, size, -1, size)
            .swapaxes(1, 2)
            .reshape(-1, size, size))


def tiff(data,path):
    """Write out a TIFF stack"""
    with TiffWriter(path) as tif:
        for i in range(data.shape[0]):
            tif.save(data[i,:,:], compress=6)


def exp_func(x, a, b, c):
    """Exponential Function"""
    return a * np.exp(-b * x) + c


def bleach_fit(brange,frange,intensity,fitter):
    """Fit decay in intensity for bleach correction"""
    intensity_values = np.array([intensity[x] for x in brange])

    # Choose type of decay
    if (fitter == 'linear'):
        # Fitting regularized linear model
        reg = linear_model.Ridge(alpha=1000,fit_intercept=True)
        try:
            reg.fit(brange.reshape(-1, 1), intensity_values.reshape(-1, 1))
        except:
            raise ValueError('Fit not found - try a larger range')
        pred = reg.predict(frange.reshape(-1, 1))

    elif (fitter == 'exponential'):
        # Fitting exponential model
        guess = (intensity[0], 0.001, 0)
        try:
            popt, _ = curve_fit(exp_func, brange, intensity_values, p0=guess)
        except:
            raise ValueError('Fit not found - try a larger range')
        pred = exp_func(frange, *popt)

    elif (fitter == 'loess'):
        # Fitting loess model
        try:
            _, pred, _ = loess_1d.loess_1d(brange, intensity_values, xnew=None, degree=1, frac=0.5, npoints=None, rotate=False, sigy=None)
        except:
            raise ValueError('Fit not found - try a larger range')

    # Bleach corrected intensity values
    corr = np.divide(pred[0], pred)

    return corr


def ratio_calc(acceptorc,donorc):
    # Divide acceptor by donor stack
    ratio = np.true_divide(acceptorc, donorc, out=np.zeros_like(acceptorc, dtype=np.float16), where=donorc != 0)
    ratio = np.nan_to_num(ratio)

    # Flatten array to find intensity percentiles
    ratio_flat = np.ravel(ratio)
    if np.nonzero(ratio_flat)[0].size == 0:
        raise Exception("Ratio image is only background. Check the acceptor/donor processed stacks")

    perc = np.percentile(ratio_flat[np.nonzero(ratio_flat)], [10, 90], interpolation='nearest')

    # Find 10th/90th percentile ratio and additive constant for scaling
    perc_ratio = perc[0] / perc[1]
    const = 0.123 * (1 - perc_ratio)

    # Rescale ratio 10th percentile - 25, 90th percentile - 230 intensity values respectively
    ratio = 230.0 * (ratio / perc[1] - perc_ratio + const) / (1.0 - perc_ratio + const)

    # Set max/min values and apply median filter
    ratio[ratio <= 0.0] = 0.0
    ratio[ratio >= 255.0] = 255.0
    ratio = ndimage.median_filter(np.uint8(ratio), size=5)

    return ratio