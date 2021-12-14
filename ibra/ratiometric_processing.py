# -*- coding: utf-8 -*-
"""
Image registration, union, ratiometric processing and bleach correction
"""

import numpy as np
import numpy.testing as test
import imreg_dft as ird
import cv2
from functions import h5, logit, time_evolution, tiff, bleach_fit, ratio_calc
from timeit import default_timer as timer
import h5py
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

# Bleach correction module
def bleach(verbose,logger,work_out_path,acceptor_bound,donor_bound,fitter,h5_save,tiff_save,frange):
    # Start time
    time_start = timer()

    # Input acceptor and donor stacks
    try:
        f3 = h5py.File(work_out_path + '_ratio_back.h5', 'r')
        ratio_frange = np.array(f3.attrs['ratio_frange'])
        acceptor = np.array(f3['acceptor'])
        donor = np.array(f3['donor'])

        acceptori = dict(zip(ratio_frange, np.array(f3['acceptori'])))
        donori = dict(zip(ratio_frange, np.array(f3['donori'])))
        f3.close()
    except:
        raise ImportError(work_out_path + "_ratio_back.h5 not found")

    # Fit and correct acceptor channel intensity
    nframes = acceptor.shape[0]
    if (acceptor_bound[1] > acceptor_bound[0]):
        # Asset range of frames
        acceptor_bound = np.subtract(acceptor_bound, 1)
        assert (sum(~np.isin(acceptor_bound,ratio_frange)) == 0), "acceptor_bleach_range should be within processed frame range"

        # Range of frames to fit(brange), and range to correct(frange)
        acceptor_brange = np.arange(acceptor_bound[0], acceptor_bound[1] + 1)
        acceptor_frange = np.arange(acceptor_brange[0], ratio_frange[-1] + 1)

        # Find correction multiplier
        acceptor_corr = bleach_fit(acceptor_brange,acceptor_frange,acceptori,fitter)

        # Update bleaching correction factor
        acceptorb = np.concatenate((np.ones(nframes-len(acceptor_frange)),acceptor_corr.reshape(-1)),axis=0)

        # Update image
        acceptor[:,:,:] = np.uint16(np.multiply(acceptor[:,:,:],acceptorb.reshape(-1,1,1)))

        # Update image median intensity
        acceptori_frange = np.array([acceptori[x] for x in ratio_frange])
        acceptori = dict(zip(ratio_frange, np.float16(np.multiply(acceptori_frange,acceptorb.reshape(-1)))))

        # Save acceptor bleaching factors
        if (h5_save):
            h5(acceptorb,'acceptorb',work_out_path+'_ratio_back.h5',ratio_frange)

    # Fit and correct donor channel intensity
    if (donor_bound[1] > donor_bound[0]):
        # Assert range of frames
        donor_bound = np.subtract(donor_bound, 1)
        assert (sum(~np.isin(donor_bound, ratio_frange)) == 0), "donor_bleach_range should be within processed frame range"

        # Range of frames to fit(brange), and range to correct(frange)
        donor_brange = np.arange(donor_bound[0], donor_bound[1] + 1)
        donor_frange = np.arange(donor_brange[0],ratio_frange[-1] + 1)

        # Find correction multiplier
        donor_corr = bleach_fit(donor_brange,donor_frange,donori,fitter)

        # Update bleaching correction factor
        donorb = np.concatenate((np.ones(nframes-len(donor_frange)),donor_corr.reshape(-1)),axis=0)

        # Update image
        donor[:,:,:] = np.uint16(np.multiply(donor[:,:,:],donorb.reshape(-1,1,1)))

        # Update image median intensity
        donori_frange = np.array([donori[x] for x in ratio_frange])
        donori = dict(zip(ratio_frange, np.float16(np.multiply(donori_frange,donorb.reshape(-1)))))

        # Save donor bleaching factors
        if (h5_save):
            h5(donorb,'donorb',work_out_path+'_ratio_back.h5',ratio_frange)

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start)+1)

    if (verbose):
        print("(Bleach Correction) Time: " + time_elapsed + " second(s)")

    # Update log file
    logger.info('(Bleach Correction) ' + 'acceptor_bleach_frames: ' + str(acceptor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1)
                + ', donor_bleach_frames: ' + str(donor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1)
                + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Create plot to show median intensity over frame number after bleaching
    time_evolution(acceptori,donori,work_out_path,'_intensity_bleach.png','Median Intensity/Bit Depth',h5_save=False)

    # Calculate 8-bit ratio image with bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        # Calculate ratio stack
        ratio = ratio_calc(acceptor,donor)

        # Save bleach corrected ratio image
        if (h5_save):
            h5_time_start = timer()
            h5(ratio,'ratio',work_out_path+'_ratio_back.h5',frange)
            h5_time_end = timer()

            if (verbose):
                print("Saving Ratio stack in " + work_out_path+'_ratio_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]")

        # Save bleach corrected ratio image as TIFF
        if (tiff_save):
            tiff_time_start = timer()
            tiff(ratio, work_out_path + '_ratio_back_bleach.tif')
            tiff_time_end = timer()

            if (verbose):
                print("Saving bleached Ratio TIFF stack in " + work_out_path + '_ratio_back_bleach.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]")


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
        raise ImportError("Acceptor stack background not processed")

    # Input donor stack
    try:
        donor = np.array(f['donor'])
        donor_frange = np.array(f.attrs['donor_frange'])
    except:
        raise ImportError("Donor stack background not processed")

    f.close()

    # Find frame dimensions and intersection between processed frames and input frames
    Ydim, Xdim = acceptor.shape[1:]
    brange = np.intersect1d(frange,acceptor_frange,return_indices=True)[2]

    # Set default values for crop
    if (crop[2] == 0):
        crop[2] = Xdim
    if (crop[3] == 0):
        crop[3] = Ydim

    # Testing input values
    test.assert_array_equal (acceptor_frange,donor_frange), "Acceptor and Donor stacks have different frame numbers"
    assert (sum(~np.isin(frange,acceptor_frange)) == 0), "background subtracted stacks have not been processed for all frame values"
    assert (crop[2] >= crop[0]), "crop[2] must be greater than crop[0]"
    assert (crop[3] >= crop[1]), "crop[3] must be greater than crop[1]"
    assert (crop[0] >= 0), "crop[0] must be >= 0"
    assert (crop[2] <= Xdim), "crop[2] must be <= than the width of the image"
    assert (crop[1] >= 0), "crop[1] must be >= 0"
    assert (crop[3] <= Ydim), "crop[3] must be <= than the height of the image"

    # Image crop
    acceptorc = acceptor[:,crop[1]:crop[3],crop[0]:crop[2]]
    donorc = donor[:,crop[1]:crop[3],crop[0]:crop[2]]

    # Search for saved ratio images
    try:
        # Input files into dictionaries
        f2 = h5py.File(work_out_path + '_ratio_back.h5', 'r')
        ratio_frange = np.array(f2.attrs['ratio_frange'])

        acceptori = dict(list(zip(ratio_frange, np.array(f2['acceptori']))))
        donori = dict(list(zip(ratio_frange, np.array(f2['donori']))))
        f2.close()
    except:
        # Initialize empty dictionaries for intensities
        acceptori, donori = {},{}

    # Initialize empty dictionaries for pixel counts
    acceptornz, donornz = {},{}

    # Set up constants for loop
    mult = np.float16(255)/np.float16(res)
    ires = 100/np.float16(res)
    ipix = 100/(Xdim*Ydim)

    # Loop through frames
    for count,frame in list(zip(frange,brange)):
        if (verbose):
            print ("(Ratio Processing) Frame Number: " + str(count+1))

        # Image registration for donor channel
        if (register):
            trans = ird.translation(acceptorc[frame,:,:], donorc[frame,:,:])
            tvec = trans["tvec"].round(4)
            donorc[frame,:,:] = np.round(ird.transform_img(donorc[frame,:,:], tvec=tvec))

        # Thresholding
        acceptors = np.uint8(np.float16(acceptorc[frame, :, :]) * mult)
        donors = np.uint8(np.float16(donorc[frame, :, :]) * mult)

        # Check for max image intensity
        if np.amax(acceptors) + np.max(donors) > 70:
            # Otsu thresholding for normal intensity images
            _, A_thresh = cv2.threshold(acceptors, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _, B_thresh = cv2.threshold(donors, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            # Simple thresholding for low intensity images
            _, A_thresh = cv2.threshold(acceptors, 2, 255, cv2.THRESH_BINARY)
            _, B_thresh = cv2.threshold(donors, 2, 255, cv2.THRESH_BINARY)

        # Setting values below threshold to zero
        acceptorc[frame,:,:] *= np.uint16(A_thresh/255)
        donorc[frame,:,:] *= np.uint16(B_thresh/255)

        # Consider only foreground pixel intensity overlapping between donor and acceptor channels to ensure channels overlap perfectly
        if (union):
            # Create mask for overlapping pixels
            C = np.multiply(A_thresh, B_thresh)
            C[C > 0] = 1

            # Set non-overlapping pixels to zero
            acceptorc[frame,:,:] *= C
            donorc[frame,:,:] *= C

        # Count number of non-zero pixels by total pixels per frame
        acceptornz[count] = np.count_nonzero(A_thresh)*ipix
        donornz[count] = np.count_nonzero(B_thresh)*ipix

        # Find the ratio of the median non-zero intensity pixels and the bit depth per frame for the acceptor stack
        if (np.amax(acceptorc[frame,:,:]) > 0.0):
            acceptori[count] = ndimage.median(acceptorc[frame,:,:], labels = A_thresh/255)*ires
        else:
            acceptori[count] = 0

        # Find the ratio of the median non-zero intensity pixels and the bit depth per frame for the donor stack
        if (np.amax(donorc[frame,:,:])> 0.0):
            donori[count] = ndimage.median(donorc[frame,:,:], labels = B_thresh/255)*ires
        else:
            donori[count] = 0

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start)+1)
    if (verbose):
        print(("(Ratio Processing) Time: " + time_elapsed + " second(s)"))

    # Update log file to save stack metrics
    print_range = [x + 1 for x in frange]
    if (max(np.ediff1d(frange,to_begin=frange[0])) > 1):
        logger.info('(Ratio Processing) ' + 'frames: ' + ",".join(list(map(str, print_range))) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
    else:
        logger.info('(Ratio Processing) ' + 'frames: ' + str(print_range[0]) + '-' + str(print_range[-1]) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))


    # Create plot to showcase median intensity over frame number and the number of foreground pixels per channel (NON-bleach corrected)
    time_evolution(acceptori,donori,work_out_path,'_intensity_nonbleach.png','Median Intensity/Bit Depth',h5_save)
    time_evolution(acceptornz,donornz,work_out_path,'_pixelcount.png','Foreground/Total Image Pixels',h5_save)

    # Calculate 8-bit ratio image with NON-bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        # Calculate ratio stack
        ratio = ratio_calc(acceptorc,donorc)

        # Save processed images, non-zero pixel count, median intensity and ratio processed images in HDF5 format
        if (h5_save):
            acceptori_brange = np.array([acceptori[a] for a in brange])
            donori_brange = np.array([donori[a] for a in brange])

            h5_time_start = timer()
            h5(acceptorc[brange,:,:],'acceptor',work_out_path+'_ratio_back.h5',frange)
            h5(donorc[brange,:,:],'donor',work_out_path+'_ratio_back.h5',frange)
            h5(acceptori_brange, 'acceptori', work_out_path + '_ratio_back.h5', frange)
            h5(donori_brange, 'donori', work_out_path + '_ratio_back.h5', frange)
            h5(ratio[brange,:,:], 'ratio', work_out_path + '_ratio_back.h5',frange)
            h5_time_end = timer()

            if (verbose):
                print(("Saving Acceptor, Donor and Ratio stacks in " + work_out_path+'_ratio_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]"))
    
        # Save NON-bleach corrected ratio image as TIFF
        if (tiff_save):
            tiff_time_start = timer()
            tiff(ratio, work_out_path + '_ratio_back.tif')
            tiff_time_end = timer()

            if (verbose):
                print(("Saving unbleached Ratio TIFF stack in " + work_out_path + '_ratio_back.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]"))