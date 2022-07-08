# RobotMinds
A collection of tutorials and experiments using tensorflow 2.

## Prepare the Windows Environment

General directions are found here: https://www.tensorflow.org/install/pip

I prefer linux commands in windows, so I started by installing git-bash.

To prepare for tensorflow projects, I installed:

    1) python for windows
    2) pip for python

For both CPU and GPU environments, I needed to install:

    1) Tensorflow

Some of the projects require that zlib be accessible:

    1) downloaded and added 'zlibwapi.dll' into the local path

For the GPU rendering I needed to install:
    
    1) cuda from nvidia
    2) cudann from nvidia
    3) visual studio redistributable ( vc_redist.'arch'.exe )
