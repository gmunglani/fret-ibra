# -*- coding: utf-8 -*-
"""
Image registration, union, ratiometric processing and bleach correction
"""

import numpy as np
import numpy.testing as test
import imreg_dft as ird
import cv2
from functions import h5, logit, plot_function, tiff, bleach_fit
from timeit import default_timer as timer
import h5py
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

# Bleach correction module
def bleach(verbose,logger,work_out_path,acceptor_bound,donor_bound,fitter,h5_save,tiff_save):
    # Start time
    time_start = timer()

    # Input acceptor and donor stacks
    try:
        f3 = h5py.File(work_out_path + '_back_ratio.h5', 'r')
        ratio_frange = np.array(f3.attrs['ratio_frange'])
        acceptor = np.array(f3['acceptor'])
        donor = np.array(f3['donor'])
        acceptori = dict(zip(ratio_frange, np.array(f3['acceptori'])))
        donori = dict(zip(ratio_frange, np.array(f3['donori'])))
        f3.close()
    except:
        raise ImportError(work_out_path + "_back_ratio.h5 not found")

    # Testing input values
    acceptor_bound = np.subtract(acceptor_bound,1)
    donor_bound = np.subtract(donor_bound,1)

    assert (sum(~np.isin(acceptor_bound,ratio_frange)) == 0), "acceptor_bleach_range should be within ratio processed range"
    assert (sum(~np.isin(donor_bound,ratio_frange)) == 0), "donor_bleach_range should be within ratio processed range"

    # Fit and correct donor channel intensity
    nframes = acceptor.shape[0]
    if (acceptor_bound[1] > acceptor_bound[0]):
        # Range of frames to fit(brange), and range to correct(frange)
        acceptor_brange = np.arange(acceptor_bound[0], acceptor_bound[1] + 1)
        acceptor_frange = np.arange(acceptor_brange[0], ratio_frange[-1] + 1)

        # Find correction multiplier
        acceptor_corr = bleach_fit(acceptor_brange,acceptor_frange,acceptori,fitter)

        # Update intensity images, median intensity, and bleaching correction factor
        acceptorb = np.concatenate((np.ones(nframes-len(acceptor_frange)),acceptor_corr.reshape(-1)),axis=0)
        acceptor[ratio_frange-acceptor_frange[0],:,:] = np.uint16(np.multiply(acceptor[ratio_frange-acceptor_frange[0],:,:],acceptorb.reshape(-1,1,1)))
        acceptori_frange = np.array([acceptori[x] for x in ratio_frange])
        acceptori = dict(zip(ratio_frange, np.uint16(np.multiply(acceptori_frange,acceptorb.reshape(-1)))))

        # Save acceptor bleaching factors
        if (h5_save):
            h5(acceptorb,'acceptorb',work_out_path+'_back_ratio.h5',ratio_frange,flag=False)
            print("Saving acceptor bleaching correction factors in " + work_out_path + '_back_ratio.h5')

    # Fit and correct acceptor channel intensity
    if (donor_bound[1] > donor_bound[0]):
        donor_brange = np.arange(donor_bound[0], donor_bound[1] + 1)
        donor_frange = np.arange(donor_brange[0],ratio_frange[-1] + 1)

        # Find correction multiplier
        donor_corr = bleach_fit(donor_brange,donor_frange,donori,fitter)

        # Update intensity images, median intensity, and bleaching correction factor
        donorb = np.concatenate((np.ones(nframes-len(donor_frange)),donor_corr.reshape(-1)),axis=0)
        donor[ratio_frange-donor_frange[0],:,:] = np.uint16(np.multiply(donor[ratio_frange-donor_frange[0],:,:],donorb.reshape(-1,1,1)))
        donori_frange = np.array([donori[x] for x in ratio_frange])
        donori = dict(zip(ratio_frange, np.uint16(np.multiply(donori_frange,donorb.reshape(-1)))))

        # Save donor bleaching factors
        if (h5_save):
            h5(donorb,'donorb',work_out_path+'_back_ratio.h5',ratio_frange,flag=False)
            print("Saving donor bleaching correction factors in " + work_out_path + '_back_ratio.h5')

    # Create plot to show median intensity over frame number after bleaching
    nfrange = plot_function(acceptori,donori,work_out_path+'_intensity_bleach.png','Median Channel Intensity')[0]

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start))

    if (verbose):
        print("(Bleach Correction) Time: " + time_elapsed + " seconds")

    # Update log file
    logger.info('(Bleach Correction) ' + 'acceptor_bleach_frames: ' + str(acceptor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1)
                + ', donor_bleach_frames: ' + str(donor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1)
                + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Calculate 8-bit ratio image with bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        ratio = np.true_divide(acceptor, donor, out=np.zeros_like(acceptor,dtype=np.float16), where=donor!= 0)
        ratio = np.nan_to_num(ratio)
        ratio = np.uint8(ratio * 255.0 / np.amax(ratio))

        # Save bleach corrected ratio image
        if (h5_save):
            h5(ratio,'ratio',work_out_path+'_back_ratio.h5',nfrange,flag=False)
            
            if (verbose):
                print("Saving ratio stack in " + work_out_path+'_back_ratio.h5')

        # Save bleach corrected ratio image as TIFF
        if (tiff_save):
            tiff(ratio, work_out_path + '_back_ratio_bleach.tif')
            
            if (verbose):
                print("Saving bleached ratio TIFF stack in " + work_out_path + '_back_ratio_bleach.tif')

def ratio(verbose,logger,work_out_path,crop,res,register,union,h5_save,tiff_save,frange):
    # Start time
    time_start = timer()

    # Input background subtracted image stack
    try:
        f = h5py.File(work_out_path + '_back.h5', 'r')
    except:
        raise ImportError(work_out_path + "_back.h5 not found")

    # Input acceptor stack
    try:
        acceptor = np.array(f['acceptor'])
        acceptor_frange = np.array(f.attrs['acceptor_frange'])
    except:
        raise ImportError("acceptor stack background not processed")

    # Input donor stack
    try:
        donor = np.array(f['donor'])
        donor_frange = np.array(f.attrs['donor_frange'])
    except:
        raise ImportError("donor stack background not processed")

    f.close()

    # Find frame dimensions and intersection between processed frames and input frames
    Ydim, Xdim = acceptor.shape[1:]
    brange = np.intersect1d(frange,acceptor_frange,return_indices=True)[2]

    # Set default values for crop
    if (crop[2] == 0):
        crop[2] = Ydim
    if (crop[3] == 0):
        crop[3] = Xdim

    # Testing input values
    test.assert_array_equal (acceptor_frange,donor_frange), "acceptor and donor stacks have differing frame numbers"
    assert (sum(~np.isin(frange,acceptor_frange)) == 0), "background subtracted stacks have not been processed for all frame values"
    assert (crop[2] >= crop[0]), "crop[2] must be greater than crop[0]"
    assert (crop[3] >= crop[1]), "crop[3] must be greater than crop[1]"
    assert (crop[0] >= 0), "crop[0] must be >= 0"
    assert (crop[2] <= Ydim), "crop[2] must be <= than the width of the image"
    assert (crop[1] >= 0), "crop[1] must be >= 0"
    assert (crop[3] <= Xdim), "crop[3] must be <= than the height of the image"

    # Image crop
    acceptorc = acceptor[:,crop[1]:crop[3],crop[0]:crop[2]]
    donorc = donor[:,crop[1]:crop[3],crop[0]:crop[2]]

    # Search for saved ratio images
    try:
        # Input files into dictionaries
        f2 = h5py.File(work_out_path + '_back_ratio.h5', 'r')
        ratio_frange = np.array(f2.attrs['ratio_frange'])
        acceptornz = dict(zip(ratio_frange, np.array(f2['acceptornz'])))
        donornz = dict(zip(ratio_frange, np.array(f2['donornz'])))
        acceptori = dict(zip(ratio_frange, np.array(f2['acceptori'])))
        donori = dict(zip(ratio_frange, np.array(f2['donori'])))
        f2.close()
    except:
        # Initialize empty dictionaries
        acceptornz, donornz, acceptori, donori = {},{},{},{}


    # Loop through frames
    mult = np.float16(255)/np.float16(res)
    for count,frame in zip(frange,brange):
        if (verbose):
            print ("(Ratio Processing) Frame Number: " + str(count+1))

        # Image registration for donor channel
        if (register):
            trans = ird.translation(acceptorc[frame,:,:], donorc[frame,:,:])
            tvec = trans["tvec"].round(4)
            donorc[frame,:,:] = np.round(ird.transform_img(donorc[frame,:,:], tvec=tvec))

        # Otsu thresholding
        tmp, A_thresh = cv2.threshold(np.uint8(np.float16(acceptorc[frame,:,:])*mult), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tmp, B_thresh = cv2.threshold(np.uint8(np.float16(donorc[frame,:,:])*mult), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Setting values below threshold to zero
        acceptorc[frame,:,:] *= A_thresh/255
        donorc[frame,:,:] *= B_thresh/255

        # Consider only foreground pixel intensity overlapping between donor and acceptor channels to ensure channels overlap perfectly
        if (union):
            # Create mask for overlapping pixels
            C = np.multiply(A_thresh,B_thresh)
            C[C > 0] = 1

            # Set non-overlapping pixels to zero
            acceptorc[frame,:,:] *= C
            donorc[frame,:,:] *= C

        # Count number of non-zero pixels per frame
        acceptornz[count] = np.count_nonzero(A_thresh)
        donornz[count] = np.count_nonzero(B_thresh)

        # Find the median non-zero intensity pixels per frame
        acceptori[count] = ndimage.median(acceptorc[frame,:,:],labels=C)
        donori[count] = ndimage.median(donorc[frame,:,:],labels=C)

    # Create plot to showcase median intensity over frame number and the number of non-zero pixels per channel (NON-bleach corrected)
    nbrange, ai, di = plot_function(acceptori,donori,work_out_path+'_intensity_nonbleach.png','Median Channel Intensity')
    nbrange, anz, dnz = plot_function(acceptornz,donornz,work_out_path+'_pixelcount.png','Non-Zero Pixel Count')

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start))
    if (verbose):
        print("(Ratio Processing) Time: " + time_elapsed + " seconds")

    # Update log file to save stack metrics
    print_range = [x + 1 for x in frange]
    if (max(np.ediff1d(frange,to_begin=frange[0])) > 1):
        logger.info('(Ratio Processing) ' + 'frames: ' + ",".join(map(str, print_range)) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
    else:
        logger.info('(Ratio Processing) ' + 'frames: ' + str(print_range[0]) + '-' + str(print_range[-1]) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Calculate 8-bit ratio image with NON-bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        ratio = np.true_divide(acceptorc, donorc, out=np.zeros_like(acceptorc,dtype=np.float16), where=donorc!= 0)
        ratio = np.nan_to_num(ratio)
        ratio = np.uint8(ratio * 255.0 / np.amax(ratio))

        # Save processed images, non-zero pixel count, median intensity and ratio processed images in HDF5 format
        if (h5_save):
            h5(acceptorc,'acceptor',work_out_path+'_back_ratio.h5',nbrange,flag=False)
            h5(donorc,'donor',work_out_path+'_back_ratio.h5',nbrange,flag=False)
            h5(ratio, 'ratio', work_out_path + '_back_ratio.h5',nbrange,flag=True)
    
            h5(anz,'acceptornz',work_out_path+'_back_ratio.h5',nbrange,flag=False)
            h5(dnz,'donornz',work_out_path+'_back_ratio.h5',nbrange,flag=False)
            h5(ai,'acceptori',work_out_path+'_back_ratio.h5',nbrange,flag=False)
            h5(di,'donori',work_out_path+'_back_ratio.h5',nbrange,flag=False)
    
            if (verbose):
                print("Saving acceptor, donor and ratio stacks in " + work_out_path+'_back_ratio.h5')
    
        # Save NON-bleach corrected ratio image as TIFF
        if (tiff_save):
            tiff(ratio, work_out_path + '_back_ratio.tif')
    
            if (verbose):
                print("Saving unbleached ratio TIFF stack in " + work_out_path + '_back_ratio.tif')