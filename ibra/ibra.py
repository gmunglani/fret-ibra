#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
FRET-IBRA is used to create fully processed ratiometric images from input donor and acceptor intensity TIFF images
"""

import os, sys, getopt
import parameter_extraction
import gui

__version__='0.4.0'

def usage():
    print((""))
    print(("Program: FRET-IBRA (FRET-Image Background-subtracted Ratiometric Analysis)"))
    print(("Version: {}".format(__version__)))
    print((""))
    print(("Config Usage:  ./ibra.py -c <config file> [Options]"))
    print(("GUI Usage:  ./ibra.py -g")) 
    print((""))
    print(("Options: -t   Output TIFF stack"))
    print(("         -v   Print progress output (verbose)"))
    print(("         -s   Save as HDF5 file"))
    print(("         -a   Save background subtraction animation (only background module)"))
    print(("         -e   Use all output options"))
    print(("         -h   Print usage"))
    print((""))

def main():
    # Print usage file
    if '-h' in sys.argv[1:]:
        usage()
        sys.exit()

    # Initialize flags
    tiff_save = False
    verbose = False
    h5_save = False
    anim_save = False

    # Check if config file is available
    if ("-c" not in sys.argv[1:] and "-g" not in sys.argv[1:]):
        usage()
        raise IOError("Config file not provided")
    elif "-c" in sys.argv[1:]:
        options, remainder = getopt.getopt(sys.argv[1:], 'c:tvsaeh')
        for opt, arg in options:
            if opt in ('-c'):
                cfname = arg

            if opt in ('-t'):
                tiff_save = True

            if opt in ('-v'):
                verbose = True

            if opt in ('-s'):
                h5_save = True

            if opt in ('-a'):
                anim_save = True

            if opt in ('-e'):
                tiff_save = True
                verbose = True
                h5_save = True
                anim_save = True

        parameter_extraction.main_extract(cfname,tiff_save,verbose,h5_save,anim_save)

    elif "-g" in sys.argv[1:]:
        gui.main_gui()

if __name__ == "__main__":
    main()
