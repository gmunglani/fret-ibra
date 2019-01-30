# -*- coding: utf-8 -*-
"""
Background subtraction with the DBSCAN clustering algorithm
using the higher moments of the intensity distribution per tile
"""

from __future__ import print_function, division

import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import cv2
import math
import pims
import os
from functions import background_animation, logit, h5, block, tiff
from timeit import default_timer as timer
from skimage.external.tifffile import TiffWriter

# #############################################################################
# Create single frame class with frame count
class frame(object):
    def __init__(self,im_stack,count):
        self.im_frame = np.asarray(im_stack[count])
        self.count = count


# Create total time stack class
class stack(frame):
        # Set eps and image type
    def __init__(self,work_inp_path,val,eps,win,start,stop,manual):
        self.val = val
        self.eps = eps
        self.saver = 0

        # Choose between the entire range and manual frames
        if (manual[0] != 0):
            manual = [x - 1 for x in manual]
            self.frange = manual
        else:
            self.frange = np.arange(start-1,stop)

        # Import frames
        im_path =  work_inp_path + '_' + self.val + '.tif'
        self.im_stack = pims.TiffStack_pil(im_path)

        # Find frame size and set window size
        self.siz1,self.siz2 = self.im_stack.frame_shape
        window = win if self.eps > 0.01 else win*2
        self.dim = np.int16(self.siz2/window)
        self.height = np.int16(window)
        self.width = np.int16(self.siz1/self.dim)

        # Create underlying background mesh
        self.X, self.Y = np.int16(np.meshgrid(np.arange(self.height), np.arange(self.width)))
        self.XY = np.column_stack((np.ravel(self.X),np.ravel(self.Y)))


    # Preallocate arrays for speed on a per frame basis
    def metric_prealloc(self):
        length = len(self.frange)
        rows = self.height*self.width
        self.im_medianf = np.empty((self.width,self.height,length),dtype=np.float32)
        self.propf = np.empty((rows,4,length),dtype=np.float32)
        self.maskf = np.empty((rows,length),dtype=np.bool)
        self.labelsf = np.empty((rows,length),dtype=np.int8)
        self.im_backf = np.empty((self.width,self.height,length),dtype=np.int16)
        self.im_framef = np.empty((length,self.siz1,self.siz2),dtype=np.uint16)


    # Calculate pixel properties per tile
    def properties(self,count):
        # Divide frame into tiles and preallocate properties array
        self.ind = frame(self.im_stack,count)
        tile_prop = np.empty([self.width*self.height,4],dtype=np.float32)
        self.ind.im_tile = block(self.ind.im_frame,self.dim)

        # Calculate 3 moments of the pixel intensity distributions and the median intensity
        for i in range(tile_prop.shape[0]):
            im_tile_flat = np.ravel(self.ind.im_tile[i,:,:])
            tile_prop[i,0] = sp.stats.moment(im_tile_flat,moment=2,axis=0)
            tile_prop[i,1] = sp.stats.moment(im_tile_flat,moment=3,axis=0)
            tile_prop[i,2] = sp.stats.moment(im_tile_flat,moment=4,axis=0)
            tile_prop[i,3] = np.median(im_tile_flat)

        self.ind.im_median = np.copy(tile_prop[:,3])

        # Fine the minimum and peak to peak values of the intensities
        tile_min = np.amin(tile_prop,axis=0)
        tile_ptp = np.ptp(tile_prop,axis=0)

        # Mean normalization of pixel properties
        for j in range(tile_prop.shape[1]):
            tile_prop[:,j] = map(lambda x : (x - tile_min[j])/tile_ptp[j], tile_prop[:,j])

        self.ind.tile_prop = tile_prop


    # Cluster tiles into background and signal
    def clustering(self):
        db = DBSCAN(eps=self.eps, min_samples=int(self.height*1.25)).fit(self.ind.tile_prop)
        self.ind.core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.ind.core_samples_mask[db.core_sample_indices_] = True
        self.ind.labels = np.int8(db.labels_)


    # Subtract median background from frame intensities
    def subtraction(self):
        # Build a median intensity mask
        im_median_mask = np.multiply(self.ind.im_median,(self.ind.labels+1))
        pos_front = np.int16(np.where(im_median_mask==0)[0])
        XY_back = np.delete(self.XY, pos_front, axis=0)
        im_median_mask_back = np.delete(im_median_mask, pos_front, axis=0)
        self.ind.im_frame = np.zeros([self.siz1,self.siz2])

        try:
            # Interpolate signal intensities over background-only tiles
            self.ind.XY_interp_back = np.uint16(griddata(XY_back, im_median_mask_back, (self.X, self.Y), method='nearest'))

            # Subtract median intensity values on a tile by tile basis
            for i, j in enumerate(self.ind.XY_interp_back.flat):
                rem = int(np.floor(i / self.height))
                mod = i % self.height
                self.ind.im_frame[rem * self.dim:(rem + 1) * self.dim, mod * self.dim:(mod + 1) * self.dim] = np.subtract(self.ind.im_tile[i,:,:], j)

            # Remove negative values
            low_values_flags = self.ind.im_frame > 65000
            self.ind.im_frame[low_values_flags] = 0

        except:
            # Change the eps value if DBSCAN does not work
            self.ind.XY_interp_back = np.zeros((self.width,self.height))
            self.logger.error("".join((self.val,'_eps: ',str(self.eps),', frame: ',str(self.ind.count+1)," (eps value too low)")))


    # Use a bilateral smoothing filter to preserve edges
    def filter(self):
        self.ind.im_frame = np.float32(self.ind.im_frame)
        filtered = cv2.bilateralFilter(self.ind.im_frame, np.int16(math.ceil(9 * self.siz2 / 1280)), self.width*0.5, self.width*0.5)
        self.ind.im_frame = np.uint16(filtered)


    # Update metrics on a per frame basis
    def metric_update(self):
        self.im_medianf[:,:,self.saver] = np.reshape(self.ind.im_median,(self.width,self.height))
        self.im_backf[:,:,self.saver] = self.ind.XY_interp_back
        self.im_framef[self.saver,:,:] = self.ind.im_frame
        self.propf[:,:,self.saver] = self.ind.tile_prop
        self.maskf[:,self.saver]  = self.ind.core_samples_mask.tolist()
        self.labelsf[:,self.saver] = self.ind.labels
        self.saver += 1


    # Use log file to print frame metrics
    def logger_update(self,logger,h5_save,time_elapsed):
        if (max(np.ediff1d(self.frange, to_begin=self.frange[0])) > 1):
            logger.info('(Background Subtraction) ' + self.val + '_eps: ' + str(self.eps) + ', frames: ' + ",".join(
                map(str, [x + 1 for x in self.frange])) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
        else:
            logger.info('(Background Subtraction) ' + self.val + '_eps: ' + str(self.eps) + ', frames: ' + str(self.frange[0]+1) + '-' + str(
                self.frange[-1]+1) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))


def background(verbose,logger,work_inp_path,work_out_path,eps,win,anim_save,h5_save,tiff_save,start,stop,manual):
    # Run through the subtraction on a per frame basis
    val = ['YFP','CFP']
    # Run through both donor and acceptor channels depending on the eps values
    for a,b in zip(val,eps):
        if (b == 0):
            continue

        # Start time
        time_start = timer()

        # Create stack class from input TIFF file
        all = stack(work_inp_path,a,b,win,start,stop,manual)

        # Preallocation of tile metrics
        all.metric_prealloc()

        # Run through the processing workflow
        for count in all.frange:
            if (verbose):
                print(a +' (Background Subtraction) Frame Number: ' + str(count + 1))
            all.properties(count)
            all.clustering()
            all.subtraction()
            all.filter()
            all.metric_update()

        # End time
        time_end = timer()
        time_elapsed = str(int(time_end - time_start))
        if (verbose):
            print(a+" (Background Subtraction) Time: " + time_elapsed + " seconds")

        # Update log file with background subtraction data
        all.logger_update(logger,h5_save,time_elapsed)

        # Save animation of frame metrics
        if (anim_save):
            background_animation(verbose,all,work_out_path)

        # Save background subtracted stack as HDF5
        if (h5_save):
            h5(all.frange,all.im_framef,a,work_out_path + '_back.h5',fstart=start)
            if (verbose):
                print("Saving " + a + " stack in " + work_out_path + + '.h5')

        if (tiff_save):
            tiff(all.im_framef, work_out_path + '_' + a + '.tif')
            if (verbose):
                print("Saving "+a+" TIFF stack in " + work_out_path + '_' + a + '.tif')

