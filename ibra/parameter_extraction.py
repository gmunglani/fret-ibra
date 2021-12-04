import os
import configparser
from functions import logit
import background_subtraction as bs
import ratiometric_processing as rp
import numpy as np

def main_extract(cfname,tiff_save,verbose,h5_save,anim_save):
    # Initialize config files
    config = configparser.ConfigParser()
    config.read(cfname)

    # Initialize input/output paths
    inp_path = config['File Parameters'].get('input_path').encode("utf-8").decode()
    fname = config['File Parameters'].get('filename').encode("utf-8").decode()
    ext = config['File Parameters'].get('extension').encode("utf-8").decode()
    current_path = os.getcwd()

    # Finalize input/output paths
    if inp_path[:2] == '..':
        work_inp_path = current_path[:-5] + inp_path[2:]
    elif inp_path[0] == '.':
        work_inp_path = current_path[:-5] + inp_path[1:]
    else:
        work_inp_path = inp_path

    # Ensure that input path exists
    if not os.path.exists(work_inp_path):
        raise IOError("Input path does not exist")

    work_inp_path += '/' + fname
    work_out_path = current_path + '/' + fname + '/'
    if not os.path.exists(work_out_path):
        os.makedirs(work_out_path)
    work_out_path += fname

    # Input options for continuous or manual frames
    frames = config['File Parameters'].get('frames')
    if (':' in frames):
        start, stop = frames.split(':')
        start = int(start)
        stop = int(stop)
        assert (stop >= start), "last frame should be greater than the first frame"
        frange = np.arange(start - 1, stop)
    else:
        frange = frames.split(',')
        frange = np.array([int(x) - 1 for x in frange])

    assert (min(frange) >= 0), "frames should only contain positive integers"

    # Input modules
    module = int(config['Modules'].get('option'))

    assert (module >= 0), "option should be between 0 and 4"
    assert (module <= 4), "option should be between 0 and 4"

    # Input TIFF file resolution
    resolution = int(config['File Parameters'].get('resolution'))
    res_types = [8, 12, 16]

    assert (resolution in res_types), "resolution must be 8, 12, or 16-bit"
    res = np.power(2, resolution) - 1

    # Input parallel option
    parallel = config['File Parameters'].getboolean('parallel')

    # Open log file
    logger = logit(work_out_path)

    # Background module options
    if (module <= 1 or module == 3):
        # Input window tile size and eps values for DBSCAN clustering algorithm
        win = int(config['Background Parameters'].get('nwindow'))
        eps = float(config['Background Parameters'].get('eps'))

        assert (win >= 10), "nwindow should be between 10 and 50"
        assert (win <= 50), "nwindow should be between 10 and 50"
        assert (eps > 0), "eps value must be a positive float between 0 and 1"
        assert (eps <= 1), "eps value must be a positive float between 0 and 1"
        assert (int(anim_save == True) + int(h5_save == True) > 0), "animation and/or h5_save must be activated"

        # Run the background subtraction algorithm for either acceptor or donor stack
        if module == 2:
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, module, eps, win, parallel, anim_save,
                      h5_save, tiff_save, frange)
        # Automated background + ratio modules
        elif module == 3:
            # Run the background subtraction algorithm for the acceptor stack
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, 0, eps, win, parallel, anim_save,
                          h5_save, tiff_save, frange)

            # Run the background subtraction algorithm for the donor stack
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, 1, eps, win, parallel, anim_save,
                          h5_save, tiff_save, frange)

    # Ratio image module
    if (module == 2 or module == 3):
        # Input crop dimensions
        crop = config['Ratio Parameters'].get('crop').split(',')
        crop = list(map(int, crop))

        # Input options for image registration and the union between donor and accepter channels, and output option for saving in HDF5
        register = config['Ratio Parameters'].getboolean('register')
        union = config['Ratio Parameters'].getboolean('union')

        # Run the ratio image processing algorithm
        rp.ratio(verbose, logger, work_out_path, crop, res, register, union, h5_save, tiff_save, frange)

    # Bleach correction module
    if (module == 4):
        # Input the bleaching range for donor and accepter channels
        acceptor_bound = config['Bleach Parameters'].get('acceptor_bleach_range').split(':')
        donor_bound = config['Bleach Parameters'].get('donor_bleach_range').split(':')
        acceptor_bound = list(map(int, acceptor_bound))
        donor_bound = list(map(int, donor_bound))

        assert (acceptor_bound[1] >= acceptor_bound[
            0]), "acceptor_bleach_range last frame should be >= acceptor_bleach_range first frame"
        assert (donor_bound[1] >= donor_bound[
            0]), "donor_bleach_range last frame should be >= donor_bleach_range first frame"

        # Input bleach correction for fitting and correcting image median intensity
        fitter = config['Bleach Parameters'].get('fit')
        fits = ['linear', 'exponential']

        assert (fitter in fits), "fit should be either linear or exponential"

        # Run bleach correction algorithm
        rp.bleach(verbose, logger, work_out_path, acceptor_bound, donor_bound, fitter, h5_save, tiff_save, frange)
