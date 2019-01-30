# Tutorial

The config file (.cfg) contains variables including the path to the image stack files, the range of frames to be processed, and multiple other options for each module. 

The *Config_tutorial.cfg* file in this folder is used to demonstrate the functionality of this toolkit. First, the input path and filename variables needs to be set.
```txt
input_path = ./stack 
filename = Test
```

The user then has the option to run one of three modules. The background subtraction module should be run first.
```txt
background = 1
ratio = 0
bleach = 0
```

The range of frames to be processed is then stated with the parameter *continuous_range*. If individually frames are desired, then the *manual_frames* parameter is set with comma-separated values. *continuous_range* is only functional when the first frame entry in *manual_frames* is set to zero.
```txt
continuous_range = 1:6
manual_frames = 0
'''



