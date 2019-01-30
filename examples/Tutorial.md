# Tutorial

The config file (.cfg) contains variables including the path to the image stack files, the range of frames to be processed, and multiple other options for each module. 

The *Config_tutorial.cfg* file in this folder is used to demonstrate the functionality of this toolkit. First, the *input_path* and *filename* parameters needs to be set.
```txt
input_path = ./stack 
filename = Test
```

The user then has the option to run one of three modules. The *background* subtraction module should be run first.
```txt
background = 1
ratio = 0
bleach = 0
```

The range of frames to be processed is then stated with the parameter *continuous_range*. If individual frames are desired, then the *manual_frames* parameter is set with comma-separated values. 

*continuous_range* is only functional when the first frame entry in *manual_frames* is set to 0.
```txt
continuous_range = 1:6
manual_frames = 0
```

The background module parameters include the *window* (or tile) size (in pixels) to divide the frame into smaller parts and the acceptor and donor channel *eps* values for the DBSCAN clustering algorithm.

Note, that the higher the *eps* value, the more foreground pixels are included. The two channels can be processed separately, by simply setting the *eps* value to 0 for the other channel.
```txt
window = 40
acceptor_eps = 0.01
donor_eps = 0.01 
```

The background subtraction module can then be run with multiple options including an output HDF5 file (-s) (necessary for further processing), a video animation of per-frame metrics (-a) and a TIFF output file.
```bash
ibra -c Config_tutorial.cfg -a -t -s
```
