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
import concurrent.futures
from itertools import repeat

# #############################################################################

# Create image stack class
class stack():
        # Set eps and image type
    def __init__(self,work_inp_path,val,ext):
        stack.val = val

        # Import stack
        im_path =  work_inp_path + '_' + stack.val + '.' + ext
        self.im_stack = pims.open(im_path)
        stack.siz1, stack.siz2 = self.im_stack.frame_shape

    # Set class frame parameters
    @classmethod
    def set_frame_parameters(cls,win):
        # Find frame size and set window size
        cls.dim = np.int16(cls.siz2/win)
        cls.height = np.int16(win)
        cls.width = np.int16(cls.siz1/cls.dim)

        # Create underlying background mesh
        cls.X, cls.Y = np.int16(np.meshgrid(np.arange(cls.height), np.arange(cls.width)))
        cls.XY = np.column_stack((np.ravel(cls.X),np.ravel(cls.Y)))

        # Setup grid for intensity weighted centroid calculation
        grid = np.indices((cls.dim, cls.dim))
        offset = (cls.dim - 1)*0.5
        stack.dist_grid = np.sqrt(np.square(np.subtract(grid[0], offset)) + np.square(np.subtract(grid[1], offset)))

    # Set class constants
    @classmethod
    def set_class_constants(cls,verbose,res,logger,frange,eps):
        cls.verbose = verbose
        cls.res = res
        cls.logger = logger
        cls.frange = frange
        cls.eps = eps


    # Preallocate arrays for speed
    def metric_prealloc(self):
        length = len(stack.frange)
        rows = self.height*self.width
        self.im_origf = np.empty((self.siz1, self.siz2, length), dtype=np.uint16)
        self.propf = np.empty((rows,5,length),dtype=np.float32)
        self.maskf = np.empty((rows,length),dtype=np.bool)
        self.labelsf = np.empty((rows,length),dtype=np.int8)
        self.im_backf = np.empty((self.width,self.height,length),dtype=np.int16)
        self.im_framef = np.empty((length,self.siz1,self.siz2),dtype=np.uint16)


    # Update metrics on a per frame basis
    def metric_update(self,result):
        pos = result[0]
        self.im_origf[:,:,pos] = result[1]
        self.im_backf[:,:,pos] = result[2]
        self.im_framef[pos,:,:] = result[3]
        self.propf[:,:,pos] = result[4]
        self.maskf[:,pos] = result[5].tolist()
        self.labelsf[:,pos] = result[6]


    # Use log file to print frame metrics
    def logger_update(self,h5_save,time_elapsed):
        if (max(np.ediff1d(stack.frange,to_begin=stack.frange[0])) > 1):
            stack.logger.info('(Background Subtraction) ' + stack.val + '_eps: ' + str(stack.eps) + ', frames: ' + ",".join(
                map(str, [x + 1 for x in stack.frange])) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
        else:
            stack.logger.info('(Background Subtraction) ' + stack.val + '_eps: ' + str(stack.eps) + ', frames: ' + str(stack.frange[0]+1) + '-' + str(
                stack.frange[-1]+1) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))


    # Run background subtraction stack workflow
    def stack_workflow(self,parallel):
        if (parallel):
            # Create frame class and submit processes
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for pos, count in enumerate(stack.frange):
                    fr = frame(np.asarray(self.im_stack[count]), count, pos)
                    process = executor.submit(fr.frame_workflow)
                    futures.append(process)

            # Combine parallelized output into updated metrics
            for future in concurrent.futures.as_completed(futures):
                self.metric_update(future.result())

        else:
            for pos, count in enumerate(stack.frange):
                # Initialize frame
                fr = frame(np.asarray(self.im_stack[count]), count, pos)

                # Run frame processing
                result = fr.frame_workflow()

                # Update metrics
                self.metric_update(result)


# Create single image frame class
class frame(stack):
    def __init__(self,im_frame,count,pos):
        self.im_frame = im_frame
        self.im_frame_orig = im_frame
        self.count = count
        self.pos = pos

    # Calculate pixel properties per tile
    def properties(self):
        # Divide frame into tiles and preallocate properties array
        tile_prop = np.empty([super().width*super().height,5],dtype=np.float32)
        self.im_tile = block(self.im_frame,super().dim)

        # Calculate 3 moments of the pixel intensity distributions, the median intensity, and the contour centriod distance
        for i in range(tile_prop.shape[0]):
            im_tile_flat = np.ravel(self.im_tile[i,:,:])
            # Moments of the intensity distribution
            tile_prop[i,0] = sp.stats.moment(im_tile_flat,moment=2,axis=0)
            tile_prop[i,1] = sp.stats.moment(im_tile_flat,moment=3,axis=0)
            tile_prop[i,2] = sp.stats.moment(im_tile_flat,moment=4,axis=0)
            tile_prop[i,3] = np.median(im_tile_flat)

            # Intensity weighted centroid (spatial) calculation
            centroid_intensity = np.multiply(self.im_tile[i, :, :], super().dist_grid)
            tile_prop[i,4] = np.sum(np.uint32(centroid_intensity))

        self.im_median = np.copy(tile_prop[:,3])

        # Fine the minimum and peak to peak values of the intensities
        tile_min = np.amin(tile_prop,axis=0)
        tile_ptp = np.ptp(tile_prop,axis=0)

        # Mean normalization of pixel properties
        for j in range(tile_prop.shape[1]):
            tile_prop[:,j] = list(map(lambda x : (x - tile_min[j])/tile_ptp[j], tile_prop[:,j]))

        self.tile_prop = tile_prop


    # Cluster tiles into background and signal
    def clustering(self):
        db = DBSCAN(eps=super().eps, min_samples=int(super().height*1.25)).fit(self.tile_prop)
        self.core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.core_samples_mask[db.core_sample_indices_] = True
        self.labels = np.int8(db.labels_)


    # Subtract median background from frame intensities
    def subtraction(self):
        # Build a median intensity mask
        im_median_mask = np.multiply(self.im_median,(self.labels+1))
        pos_front = np.int16(np.where(im_median_mask==0)[0])
        XY_back = np.delete(super().XY, pos_front, axis=0)
        im_median_mask_back = np.delete(im_median_mask, pos_front, axis=0)
        self.im_frame = np.zeros([super().siz1,super().siz2])

        try:
            # Interpolate signal intensities over background-only tiles
            self.XY_interp_back = np.uint16(griddata(XY_back, im_median_mask_back, (super().X, super().Y), method='nearest'))

            # Subtract median intensity values on a tile by tile basis
            for i, j in enumerate(self.XY_interp_back.flat):
                rem = int(np.floor(i / super().height))
                mod = i % super().height
                self.im_frame[rem * super().dim:(rem + 1) * super().dim, mod * super().dim:(mod + 1) * super().dim] = np.subtract(self.im_tile[i,:,:], j)

                # Remove negative values
                self.im_frame[self.im_frame > np.amax(self.im_frame_orig)] = 0
                self.im_frame[self.im_frame < 0] = 0

        except:
            # Change the eps value if DBSCAN does not work
            self.XY_interp_back = np.zeros((super().width,super().height))
            stack.logger.error("".join((super().val,'_eps: ',str(super().eps),', frame: ',str(self.count+1)," (eps value too low)")))


    # Use a bilateral smoothing filter to preserve edges
    def filter(self):
        filtered = cv2.bilateralFilter(np.float32(self.im_frame), np.int16(math.ceil(9 * super().siz2 / 320)), super().width*0.5, super().width*0.5)
        self.im_frame = np.uint16(filtered)

    # Run frame background subtraction workflow
    def frame_workflow(self):
        if (super().verbose):
            print((super().val.capitalize() + ' (Background Subtraction) Frame Number: ' + str(self.count + 1)))
        self.properties()
        self.clustering()
        self.subtraction()
        self.filter()

        return (self.pos, self.im_frame_orig, self.XY_interp_back, self.im_frame, self.tile_prop, self.core_samples_mask, self.labels)


def background(verbose,logger,work_inp_path,work_out_path,ext,res,module,eps,win,parallel,anim_save,h5_save,tiff_save,frange):
    # Run through the donor/acceptor subtraction on a per frame basis
    if module == 0:
        val = 'acceptor'
    else:
        val = 'donor'

    # Start time
    time_start = timer()

    # Create stack class from input TIFF file
    all = stack(work_inp_path,val,ext)

    # Frame number check
    assert (max(frange) < len(all.im_stack)), "frame numbers not found in input TIFF stack"

    # Assign frame parameters
    all.set_frame_parameters(win)

    # Assign class constants
    all.set_class_constants(verbose,res,logger,frange,eps)

    # Preallocation of tile metrics
    all.metric_prealloc()

    # Run image processing workflow
    all.stack_workflow(parallel)

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start)+1)
    if (verbose):
        print((val.capitalize() +" (Background Subtraction) Time: " + time_elapsed + " second(s)"))

    # Update log file with background subtraction data
    all.logger_update(h5_save,time_elapsed)

    # Save animation of frame metrics
    if (anim_save):
        background_animation(verbose,all,work_out_path,frange)

    # Save background subtracted stack as HDF5
    if (h5_save):
        h5_time_start = timer()
        h5(all.im_framef,val,work_out_path + '_back.h5',frange=frange)
        h5_time_end = timer()

        if (verbose):
            print(("Saving " + val.capitalize() + " HDF5 stack in " + work_out_path + '.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start)+1) + " second(s)]"))

    # Save background-subtracted acceptor/donor images as TIFF
    if (tiff_save):
        tiff_time_start = timer()
        tiff(all.im_framef, work_out_path + '_' + val + '_back.tif')
        tiff_time_end = timer()

        if (verbose):
            print(("Saving " + val.capitalize() + " TIFF stack in " + work_out_path + '_back_' + val + '.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]"))

