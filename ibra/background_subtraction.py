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

# Create total image stack class
class stack():
        # Set eps and image type
    def __init__(self,work_inp_path,val):
        stack.val = val

        # Import stack
        im_path =  work_inp_path + '_' + stack.val + '.tif'
        self.im_stack = pims.TiffStack_pil(im_path)
        stack.siz1,stack.siz2 = self.im_stack.frame_shape

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


    # Set class constants
    @classmethod
    def set_class_constants(cls,verbose,res,logger,frange,eps):
        cls.verbose = verbose
        cls.res = res
        cls.logger = logger
        cls.frange = frange
        cls.eps = eps


    # Preallocate arrays for speed on a per frame basis
    def metric_prealloc(self):
        length = len(stack.frange)
        rows = self.height*self.width
        self.im_medianf = np.empty((self.width,self.height,length),dtype=np.float32)
        self.propf = np.empty((rows,5,length),dtype=np.float32)
        self.maskf = np.empty((rows,length),dtype=np.bool)
        self.labelsf = np.empty((rows,length),dtype=np.int8)
        self.im_backf = np.empty((self.width,self.height,length),dtype=np.int16)
        self.im_framef = np.empty((length,self.siz1,self.siz2),dtype=np.uint16)


    # Update metrics on a per frame basis
    def metric_update(self,result):
        pos = result[0]
        self.im_medianf[:,:,pos] = np.reshape(result[1],(self.width,self.height))
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


    # Run background subtraction workflow
    def stack_workflow(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for pos, count in enumerate(stack.frange):
                fr = frame(np.asarray(self.im_stack[count]), count, pos)
                process = executor.submit(fr.frame_workflow)
                futures.append(process)

        for future in concurrent.futures.as_completed(futures):
            self.metric_update(future.result())

# Create single frame class with frame count
class frame(stack):
    def __init__(self,im_frame,count,pos):
        self.im_frame = im_frame
        self.count = count
        self.pos = pos

    # Calculate pixel properties per tile
    def properties(self):
        # Divide frame into tiles and preallocate properties array
        tile_prop = np.empty([super().width*super().height,5],dtype=np.float32)
        self.im_tile = block(self.im_frame,super().dim)

        # Create thresholded temporary frame for extracting centroids
        mult = np.float16(255) / np.float16(super().res)
        im_frame_con = np.uint8(np.float16(self.im_frame) * mult)
        _,im_frame_con_thresh = cv2.threshold(im_frame_con, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        im_tile_res = block(im_frame_con_thresh,super().dim)

        # Calculate 3 moments of the pixel intensity distributions, the median intensity, and the contour centriod distance
        for i in range(tile_prop.shape[0]):
            im_tile_flat = np.ravel(self.im_tile[i,:,:])
            tile_prop[i,0] = sp.stats.moment(im_tile_flat,moment=2,axis=0)
            tile_prop[i,1] = sp.stats.moment(im_tile_flat,moment=3,axis=0)
            tile_prop[i,2] = sp.stats.moment(im_tile_flat,moment=4,axis=0)
            tile_prop[i,3] = np.median(im_tile_flat)

            # Find contours and the default centroid value
            contours, _ = cv2.findContours(im_tile_res[i,:,:], 1, 2)
            center = im_tile_res.shape[2]/2

            # Find the contour centroid and distance from the default
            try:
                M = cv2.moments(contours[0])
                tile_prop[i,4] = math.sqrt(((int(M['m10'] / M['m00']) - center) ** 2) + ((int(M['m01'] / M['m00']) - center) ** 2))
            except:
                tile_prop[i,4] = 0


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
            low_values_flags = self.im_frame > 65000
            self.im_frame[low_values_flags] = 0

        except:
            # Change the eps value if DBSCAN does not work
            self.XY_interp_back = np.zeros((super().width,super().height))
            stack.logger.error("".join((super().val,'_eps: ',str(super().eps),', frame: ',str(self.count+1)," (eps value too low)")))


    # Use a bilateral smoothing filter to preserve edges
    def filter(self):
        self.im_frame = np.float32(self.im_frame)
        filtered = cv2.bilateralFilter(self.im_frame, np.int16(math.ceil(9 * super().siz2 / 320)), super().width*0.5, super().width*0.5)
        self.im_frame = np.uint16(filtered)

    # Run frame background subtraction workflow
    def frame_workflow(self):
        if (stack.verbose):
            print((super().val.capitalize() + ' (Background Subtraction) Frame Number: ' + str(self.count + 1)))
        self.properties()
        self.clustering()
        self.subtraction()
        self.filter()

        return (self.pos,self.im_median,self.XY_interp_back,self.im_frame,self.tile_prop,self.core_samples_mask,self.labels)


def background(verbose,logger,work_inp_path,work_out_path,res,module,eps,win,anim_save,h5_save,tiff_save,frange):
    # Run through the donor/acceptor subtraction on a per frame basis
    if module == 0:
        val = 'acceptor'
    else:
        val = 'donor'

    assert (eps > 0), "eps value must be a positive float between 0 and 1"
    assert (eps <= 1), "eps value must be a positive float between 0 and 1"

    # Start time
    time_start = timer()

    # Create stack class from input TIFF file
    all = stack(work_inp_path,val)

    assert (max(frange) < len(all.im_stack)), "frame numbers not found in input TIFF stack"

    # Assign frame parameters
    all.set_frame_parameters(win)

    # frange = [10, 12, 45, 34]
    # Assign class constants
    all.set_class_constants(verbose,res,logger,frange,eps)

    # Preallocation of tile metrics
    all.metric_prealloc()

    # Run image processing workflow
    all.stack_workflow()

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start))
    if (verbose):
        print((val.capitalize() +" (Background Subtraction) Time: " + time_elapsed + " seconds"))

    # Update log file with background subtraction data
    all.logger_update(h5_save,time_elapsed)

    # Save animation of frame metrics
    if (anim_save):
        background_animation(verbose,all,work_out_path,frange)

    # Save background subtracted stack as HDF5
    if (h5_save):
        h5(all.im_framef,val,work_out_path + '_back.h5',frange=frange)
        if (verbose):
            print(("Saving " + val.capitalize() + " HDF5 stack in " + work_out_path + '.h5'))

    # Save background-subtracted acceptor/donor images as TIFF
    if (tiff_save):
        tiff(all.im_framef, work_out_path + '_' + val + '.tif')
        if (verbose):
            print(("Saving " + val.capitalize() + " TIFF stack in " + work_out_path + '_back_' + val + '.tif'))

