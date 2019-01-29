# Fluorescence Resonance Energy Transfer - Image Background-subtracted Ratiometric Analysis (fret-ibra)

fret-ibra is used to process fluorescence resonance energy transfer (FRET) intensity data to produce ratiometric images for further analysis. The package contains modules for the background subtraction (using a novel algorithm based on DBSCAN clustering), image registration, and bleach correction of the donor and acceptor channels. The package accepts only multi-image TIFF stacks and outputs both multi-image TIFF and HDF5 stacks. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install ibra
```

## Usage

```bash
Usage: python ibra.py -c <config file> [Options]
Options: -t   Output TIFF stack
         -v   Print progress output (verbose)
         -s   Save as HDF5 file
         -a   Save background subtraction animation (only background module)
         -e   Use all output options
         -h   Print usage
```

