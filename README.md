# FRET - Image Background-subtracted Ratiometric Analysis (FRET - IBRA)

FRET - IBRA is a toolkit to process fluorescence resonance energy transfer (FRET) intensity data to produce ratiometric images for further analysis. This toolkit contains modules for the background subtraction (using an algorithm based on tiled DBSCAN clustering), image registration, overlap correction, and bleach correction of the donor and acceptor channels. It accepts multi-image TIFF stacks as input and outputs both multi-image TIFF and HDF5 stacks for possible further analyses, along with frame-by-frame metrics to estimate quality. The background subtraction algorithm works best on images with a small number of cells visible in the frame.


## Installation
FRET - IBRA can be downloaded directly from github.
```bash
git clone https://github.com/gmunglani/fret-ibra.git
cd fret-ibra
python setup.py install
```
Additional requirements: ffmpeg

## Usage

```bash
Usage: ./ibra.py -c <config file> [Options]
Options: -t   Output TIFF stack
         -v   Print progress output (verbose)
         -s   Save as HDF5 file
         -a   Save background subtraction animation (only background module)
         -e   Use all output options
         -h   Print usage
```

## Examples

### Acceptor channel input image
![YFP](/examples/images/YFP_input.png)

### Donor channel input image
![CFP](/examples/images/CFP_input.png)

### Ratiometric output image (8-bit)
Processing includes:
* Background subtraction for both channels
* Image registration
* Overlap correction
* Bleach correction

![Ratio](/examples/images/Ratio_output.png)

A detailed explanation of the toolkit can be found here: [Tutorial](/examples/Tutorial.md)
