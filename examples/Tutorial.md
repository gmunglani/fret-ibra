# Tutorial

The config file (.cfg) contains variables including the path to the image stack files, the range of frames to be processed, and multiple other options for each module. We will use the Config_tutorial.cfg file in this folder to demonstrate the functionality of this toolkit.

First, the input path and filename variables needs to be set
```txt
input_path = ./images 
filename = Test
```

The user then has the option to run one of three modules. The background subtraction module should be run first
```txt
background = 1
ratio = 0
bleach = 0
```



