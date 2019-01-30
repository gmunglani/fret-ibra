# FRET - Image Background-subtracted Ratiometric Analysis (FRET - IBRA)

FRET - IBRA is used to process fluorescence resonance energy transfer (FRET) intensity data to produce ratiometric images for further analysis. The package contains modules for the background subtraction (using a novel algorithm based on DBSCAN clustering), image registration, and bleach correction of the donor and acceptor channels. The package accepts only multi-image TIFF stacks and outputs both multi-image TIFF and HDF5 stacks. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install fret-ibra
```
Additional requirements: ffmpeg

## Usage

```bash
Usage: ibra -c <config file> [Options]
Options: -t   Output TIFF stack
         -v   Print progress output (verbose)
         -s   Save as HDF5 file
         -a   Save background subtraction animation (only background module)
         -e   Use all output options
         -h   Print usage
```

## Capa

### Acceptor channel input image
![YFP](/examples/YFP_input.png)

### Donor channel input image
![CFP](/examples/CFP_input.png)

### Ratiometric output image
![Ratio](/examples/Ratio_output.png)

For a detailed tutorial, look in [tutorial](/examples)
