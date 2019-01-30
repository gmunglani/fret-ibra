# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting, logging, data output, fitting etc
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import mpl_toolkits.mplot3d.axes3d as p3
import logging
from timeit import default_timer as timer
import h5py
from skimage.external.tifffile import TiffWriter
import os
from scipy.optimize import curve_fit
from sklearn import linear_model

rcParams['font.family'] = 'serif'

# Create animation of background subtraction
def background_animation(verbose,stack, work_out_path):
    """Background subtraction result per frame video"""
    def data(i, stack, line):
        ax1.clear()
        line1 = ax1.plot_surface(stack.X, stack.Y, stack.im_medianf[:, :, i], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax1.set_title("{} Frame: {}".format(stack.val, stack.frange[i] + 1))
        ax1.set_zlim(0, np.amax(stack.im_medianf))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.grid(False)

        ax2.clear()
        minmax = np.ptp(np.ravel(stack.im_backf[:,:,i]))
        line2 = ax2.plot_surface(stack.X, stack.Y, stack.im_backf[:, :, i], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax2.set_title("Min to Max (Background): {}".format(minmax))
        ax2.set_zlim(0, np.amax(stack.im_medianf))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.grid(False)

        ax3.clear()
        line3 = ax3.plot_surface(X1, Y1, stack.im_framef[i, :, :], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax3.set_title("Background Subtracted Image")
        ax3.set_zlim(0, np.amax(stack.im_medianf))
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
        ax4.grid(False)
        ax4.set_xlabel('Variance', labelpad=10)
        ax4.set_ylabel('Skewness', labelpad=10)
        ax4.set_zlabel('Median', labelpad=10)
        varn = stack.propf[:,:,i]
        xyz = varn[stack.maskf[:,i]]
        xyz2 = varn[[not i for i in stack.maskf[:,i]]]
        line4 = ax4.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:, 3], c='blue')
        line4 = ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 3], c='red', s=80)

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

    line1 = ax1.plot_surface(stack.X,stack.Y,stack.im_medianf[:,:,0],cmap=cm.bwr)
    line2 = ax2.plot_surface(stack.X,stack.Y,stack.im_backf[:,:,0],cmap=cm.bwr)
    line3 = ax3.plot_surface(X1,Y1,stack.im_framef[0,:,:],cmap=cm.bwr)
    line4 = ax4.scatter(0.5, 0.5, 0.5, c='red')

    line = [line1, line2, line3, line4]

    # Set up animation
    anim = animation.FuncAnimation(fig, data, fargs=(stack,line),frames=np.arange(len(stack.frange)), interval=20, blit=False, repeat_delay=1000)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']

    # Write animation file
    if (max(np.ediff1d(stack.frange, to_begin=stack.frange[0])) > 1):
        fname = work_out_path + '_' + stack.val + '_manual'
        num = 1
        while (os.path.isfile(fname+str(num)+'.avi')):
            num += 1
        else:
            anim.save(fname+str(num)+'.avi', writer=Writer(fps=2))
    else:
        anim.save(work_out_path + '_' + stack.val + '_frames_' + str(stack.frange[0]+1) + '_' + str(stack.frange[-1]+1) + '.avi', writer=Writer(fps=2))

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start))
    if (verbose):
        print(stack.val+" (Background Animation) Time: " + time_elapsed + " seconds")

def logit(path):
    """Logging data"""
    logger = logging.getLogger('back')
    hdlr = logging.FileHandler(path+'.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(20)

    return logger

def h5(frange,data,val,path,fstart=0):
    """Saving the image stack as a .h5 file"""
    f = h5py.File(path, 'a')
    if (max(np.ediff1d(frange, to_begin=frange[0])) > 1):
        # For manually selected frames, replace data already present in h5 file
        orig = f[val]
        for i,j in enumerate(frange):
            orig[j] = data[i]
    else:
        # For continuous frames, delete key and replace
        if (val in f):
            del f[val]
        f.create_dataset(val, data=data, shape=data.shape, dtype=np.uint16, compression='gzip')
        # Save the starting frame in the original TIFF stack
        if (val+'_fstart' not in f.attrs and fstart != 0):
            f.attrs[val+'_fstart'] = fstart
    f.close()

def intensity_plot(nfrange,YFPi,CFPi,path):
    """Median channel intensity per frame"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nfrange,YFPi,c='r',marker='*')
    ax.plot(nfrange,CFPi,c='b',marker='*')

    plt.xlabel('Frame Number',labelpad=15, fontsize=28)
    plt.ylabel('Median Channel Intensity',labelpad=15, fontsize=28)
    plt.xticks(fontsize=18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yticks(fontsize=18)
    plt.legend(['Acceptor','Donor'],fancybox=None,fontsize=18)
    plt.savefig(path, bbox_inches='tight')

def pixel_count(nfrange,YFPc,CFPc,path):
    """Non-zero pixel count per frame"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nfrange,YFPc,c='r',marker='*')
    ax.plot(nfrange,CFPc,c='b',marker='*')

    plt.xlabel('Frame Number',labelpad=15, fontsize=28)
    plt.ylabel('Non-Zero Pixel Count',labelpad=15, fontsize=28)
    plt.xticks(fontsize=18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yticks(fontsize=18)
    plt.legend(['Acceptor','Donor'],fancybox=None,fontsize=18)
    plt.savefig(path, bbox_inches='tight')

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

def prealloc(frange):
    """Pre-allocate arrays for frame metrics"""
    fframes = len(frange)
    YFPnz = np.empty(fframes,dtype=np.uint64)
    CFPnz = np.empty(fframes,dtype=np.uint64)
    YFPi = np.zeros(fframes,dtype=np.uint16)
    CFPi = np.zeros(fframes,dtype=np.uint16)

    return YFPnz,CFPnz,YFPi,CFPi

def exp_func(x, a, b, c):
    """Exponential Function"""
    return a * np.exp(-b * x) + c

def bleach_fit(brange,frange,intensity,fitter):
    """Fit decay in intensity for bleach correction"""
    if (fitter == 'linear'):
        # Fitting regularized linear model
        reg = linear_model.Ridge(alpha=10000,fit_intercept=True)
        try:
            reg.fit(brange.reshape(-1, 1), intensity.reshape(-1, 1))
        except:
            raise ValueError('Fit not found - try a larger range')
        pred = reg.predict(frange.reshape(-1, 1))
    elif (fitter == 'exponential'):
        # Fitting exponential model
        guess = (intensity[0], 0.001, 0)
        try:
            popt, tmp = curve_fit(exp_func, brange, intensity, p0=guess)
        except:
            raise ValueError('Fit not found - try a larger range')
        pred = exp_func(frange, *popt)

    # Bleach corrected intensity values
    corr = np.divide(pred[0], pred)

    return corr