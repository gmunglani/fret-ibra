# -*- coding: utf-8 -*-
"""
Image registration, union, ratiometric processing and bleach correction
"""

import numpy as np
import imreg_dft as ird
import cv2
from functions import h5, logit, intensity_plot, pixel_count, tiff, prealloc, bleach_fit
from timeit import default_timer as timer
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import ndimage

# Bleach correction module
def bleach(verbose,logger,work_out_path,YFP_range,CFP_range,fitter,h5_save,tiff_save):
    # Start time
    time_start = timer()

    # Input YFP and CFP stacks
    try:
        f = h5py.File(work_out_path + '_back_ratio.h5', 'r')
        YFP = np.array(f['YFP'])
        CFP = np.array(f['CFP'])
        YFPi = np.array(f['YFPi'])
        CFPi = np.array(f['CFPi'])
        fstart = int(f.attrs['YFP_fstart'])
        f.close()
    except:
        raise ImportError(work_out_path + "_back_ratio.h5 not found")

    # Testing input values
    nframes = YFP.shape[0]
    assert (YFP_range[0] >= fstart), "YFP_bleach_range should be within ratio processed range"
    assert (YFP_range[1] <= fstart+nframes), "YFP_bleach_range should be within ratio processed range"
    assert (CFP_range[0] >= fstart), "CFP_bleach_range should be within ratio processed range"
    assert (CFP_range[1] <= fstart+nframes), "CFP_bleach_range should be within ratio processed range"

    # Fit and correct donor channel intensity
    if (YFP_range[1] > YFP_range[0]):
        # Range of frames to fit(brange), and range to correct(frange)
        YFP_brange = np.arange(YFP_range[0] - fstart, YFP_range[1] - fstart + 1)
        YFP_frange = np.arange(YFP_brange[0],fstart+nframes-1)

        # Find correction multiplier
        YFP_corr = bleach_fit(YFP_brange,YFP_frange,YFPi[YFP_brange],fitter)

        # Update intensity images, median intensity, and bleaching correction factor
        YFP[YFP_frange,:,:] = np.uint16(np.multiply(YFP[YFP_frange,:,:],YFP_corr.reshape(-1,1,1)))
        YFPi[YFP_frange] = np.uint16(np.multiply(YFPi[YFP_frange],YFP_corr.reshape(-1)))
        YFPb = np.concatenate((np.ones(nframes-len(YFP_frange)),YFP_corr.reshape(-1)),axis=0)

    # Fit and correct acceptor channel intensity
    if (CFP_range[1] > CFP_range[0]):
        # Range of frames to fit(brange), and range to correct(frange)
        CFP_brange = np.arange(CFP_range[0] - fstart, CFP_range[1] - fstart + 1)
        CFP_frange = np.arange(CFP_brange[0],fstart+nframes-1)

        # Find correction multiplier
        CFP_corr = bleach_fit(CFP_brange,CFP_frange,CFPi[CFP_brange],fitter)

        # Update intensity images, median intensity, and bleaching correction factor
        CFP[CFP_frange,:,:] = np.uint16(np.multiply(CFP[CFP_frange,:,:],CFP_corr.reshape(-1,1,1)))
        CFPi[CFP_frange] = np.uint16(np.multiply(CFPi[CFP_frange],CFP_corr.reshape(-1)))
        CFPb = np.concatenate((np.ones(nframes-len(CFP_frange)),CFP_corr.reshape(-1)),axis=0)

    # Create plot to showcase median intensity over frame number after bleaching
    frange = np.arange(nframes)
    nfrange = [x + fstart + 1 for x in frange]
    intensity_plot(nfrange,YFPi,CFPi,work_out_path+'_intensity_bleach.png')

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start))
    if (verbose):
        print("(Bleach Correction) Time: " + time_elapsed + " seconds")

    # Update log file
    logger.info('(Bleach Correction) ' + 'YFP_bleach_frames: ' + str(YFP_frange[0]+1) + '-' + str(YFP_frange[-1]+1)
                + ', CFP_bleach_frames: ' + str(CFP_frange[0]+1) + '-' + str(CFP_frange[-1]+1)
                + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Calculate 8-bit ratio image with bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        ratio = np.true_divide(YFP, CFP, out=np.zeros_like(YFP,dtype=np.float16), where=CFP!= 0)
        ratio = np.nan_to_num(ratio)
        ratio = np.uint8(ratio * 255.0 / np.amax(ratio))

    # Save bleaching correction factors and bleach corrected ratio image
    if (h5_save):
        h5(frange,YFPb,'YFPb',work_out_path+'_back_ratio.h5')
        h5(frange,CFPb,'CFPb',work_out_path+'_back_ratio.h5')
        h5(frange,ratio,'Ratio',work_out_path+'_back_ratio.h5')
        if (verbose):
            print("Saving YFP and CFP correction factors and Ratio stack in " + work_out_path+'_back_ratio.h5')

    # Save bleach corrected ratio image as TIFF
    if (tiff_save):
        tiff(ratio, work_out_path + '_back_ratio_bleach.tif')
        if (verbose):
            print("Saving bleached ratio TIFF stack in " + work_out_path + '_back_ratio_bleach.tif')

def ratio(verbose,logger,work_out_path,crop,res,register,union,h5_save,tiff_save,start,stop,manual):
    # Start time
    time_start = timer()

    # Input background subtracted image stack
    try:
        f = h5py.File(work_out_path + '_back.h5', 'r')
    except:
        raise ImportError(work_out_path + "_back.h5 not found")

    # Input YFP stack
    try:
        YFP = np.array(f['YFP'])
        fstart = int(f.attrs['YFP_fstart'])
    except:
        raise ImportError("YFP stack background not processed")

    # Input CFP stack
    try:
        CFP = np.array(f['CFP'])
        CFP_fstart = int(f.attrs['CFP_fstart'])
    except:
        raise ImportError("CFP stack background not processed")

    f.close()
    nframes, Xdim, Ydim = YFP.shape

    # Testing input values
    assert (YFP.shape == CFP.shape), "YFP and CFP stacks have different sizes"
    assert (fstart == CFP_fstart), "YFP and CFP stacks have different start points"
    assert (start >= fstart), "Background subtracted stacks have not been processed for this start value"
    assert (stop <= fstart+nframes), "Background subtracted stacks have not been processed for this stop value"
    assert (crop[1] >= crop[0]), "crop[1] must be greater than crop[0]"
    assert (crop[3] >= crop[2]), "crop[3] must be greater than crop[2]"
    assert (crop[0] >= 0), "crop[0] must be >= 0"
    assert (crop[1] <= Xdim), "crop[1] must be <= than the width of the image"
    assert (crop[2] >= 0), "crop[2] must be >= 0"
    assert (crop[3] <= Ydim), "crop[3] must be <= than the width of the image"

    # Set default values for crop
    if (crop[1] == 0):
        crop[1] = Xdim
    if (crop[3] == 0):
        crop[3] = Ydim

    # Choose between continuous range and manually picking specific frames
    manual_flag = False
    if (manual[0] != 0):
        # Set range
        frange = [x - fstart - 1 for x in manual]

        # Input earlier h5 file if present
        try:
            f2 = h5py.File(work_out_path + '_back_ratio.h5', 'r')
            YFPnz = np.array(f2['YFPnz'])
            CFPnz = np.array(f2['CFPnz'])
            YFPi = np.array(f2['YFPi'])
            CFPi = np.array(f2['CFPi'])
            f2.close()
        except:
            # Pre-allocate metric arrays
            YFPnz, CFPnz, YFPi, CFPi = prealloc(frange)
            # Set manual_flag if only specific frames are background subtracted and ratio processed
            manual_flag = True
    else:
        # Set frange for continuous range and pre-allocate metric arrays
        frange = np.arange(start-fstart,stop-fstart+1)
        YFPnz, CFPnz, YFPi, CFPi = prealloc(frange)

    # Image crop
    YFPc = YFP[:,crop[0]:crop[1],crop[2]:crop[3]]
    CFPc = CFP[:,crop[0]:crop[1],crop[2]:crop[3]]

    # Loop through frange
    mult = np.float16(255)/np.float16(res)
    for frame,count in enumerate(frange):
        if (verbose):
            print ("(Ratio Processing) Frame Number: " + str(count+fstart))

        # Image registration for donor channel
        if (register):
            trans = ird.translation(YFPc[count,:,:], CFPc[count,:,:])
            tvec = trans["tvec"].round(4)
            CFPc[count,:,:] = np.round(ird.transform_img(CFPc[count,:,:], tvec=tvec))

        # Otsu thresholding
        tmp, A_thresh = cv2.threshold(np.uint8(np.float16(YFPc[count,:,:])*mult), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tmp, B_thresh = cv2.threshold(np.uint8(np.float16(CFPc[count,:,:])*mult), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Setting values below threshold to zero
        YFPc[count,:,:] *= A_thresh/255
        CFPc[count,:,:] *= B_thresh/255

        # Consider only foreground pixel intensity overlapping between donor and acceptor channels to ensure channels overlap perfectly
        if (union):
            # Create mask for overlapping pixels
            C = np.multiply(A_thresh,B_thresh)
            C[C > 0] = 1

            # Set non-overlapping pixels to zero
            YFPc[count,:,:] *= C
            CFPc[count,:,:] *= C

        # Special indexing for only manually picked frame processing
        if (manual_flag):
            pos = frame
        else:
            pos = count

        # Count number of non-zero pixels per frame
        YFPnz[pos] = np.count_nonzero(A_thresh)
        CFPnz[pos] = np.count_nonzero(B_thresh)

        # Find the median non-zero intensity pixels per frame
        YFPi[pos] = ndimage.median(YFPc[count,:,:],labels=C)
        CFPi[pos] = ndimage.median(CFPc[count,:,:],labels=C)

    # Create plot to showcase median intensity over frame number and the number of non-zero pixels per channel (NON-bleach corrected)
    nfrange = [x + fstart + 1 for x in frange]
    intensity_plot(nfrange,YFPi,CFPi,work_out_path+'_intensity_nonbleach.png')
    pixel_count(nfrange,YFPnz,CFPnz,work_out_path+'_pixelcount.png')

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start))
    if (verbose):
        print("(Ratio Processing) Time: " + time_elapsed + " seconds")

    # Update log file to save stack metrics
    if (max(np.ediff1d(frange, to_begin=frange[0])) > 1):
        logger.info('(Ratio Processing) ' + 'frames: ' + ",".join(
            map(str, nfrange)) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
    else:
        logger.info('(Ratio Processing) ' + 'frames: ' + str(nfrange[0]-1) + '-' + str(frange[-1]-1) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Calculate 8-bit ratio image with NON-bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        ratio = np.true_divide(YFPc, CFPc, out=np.zeros_like(YFPc,dtype=np.float16), where=CFPc!= 0)
        ratio = np.nan_to_num(ratio)
        ratio = np.uint8(ratio * 255.0 / np.amax(ratio))

    # Save processed images, non-zero pixel count, median intensity and ratio processed images in HDF5 format
    if (h5_save):
        h5(frange,YFPc,'YFP',work_out_path+'_back_ratio.h5',fstart=fstart)
        h5(frange,CFPc,'CFP',work_out_path+'_back_ratio.h5')
        h5(frange,YFPnz,'YFPnz',work_out_path+'_back_ratio.h5')
        h5(frange,CFPnz,'CFPnz',work_out_path+'_back_ratio.h5')
        h5(frange,YFPi,'YFPi',work_out_path+'_back_ratio.h5')
        h5(frange,CFPi,'CFPi',work_out_path+'_back_ratio.h5')
        h5(frange,ratio, 'Ratio', work_out_path + '_back_ratio.h5')
        if (verbose):
            print("Saving YFP, CFP and Ratio stacks in " + work_out_path+'_back_ratio.h5')

    # Save NON-bleach corrected ratio image as TIFF
    if (tiff_save):
        tiff(ratio, work_out_path + '_back_ratio.tif')
        if (verbose):
            print("Saving unbleached ratio TIFF stack in " + work_out_path + '_back_ratio.tif')