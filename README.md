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

## History ( latest first )

2022/08/03 = Modified the CycleGAN_Horse2Zebra.py code to use an AdventureTime image dataset
    instead of zebras.  The results were dissapointing.  The images retained the horse shape,
    but became broad swathes of adventure time color based on broad swathes of similar shading
    in the horse image.  It never picked up on the 'dark outline' for cartoon.
    to /horseZebra_transform/CycleGAN_Horse2AdventureTime.py

2022/08/03 = Modified 91_StyleTransfer.py to use an AdventureTime image for style
    The results were disappointing, colors and shapes were transfered, but not at a global scale.
    to /style/Dog_AdventureTime_StyleTransfer.py

2022/07/24 = added utility to reduce datasets locally

2022/07/20 = modified CNN_direct_cartoon_classify to use the image_dataset_from_directory method.
    to CNN_idfd_cartoon_classify.py

2022/07/20 = modified CNN_direct_hotdog_classify to use the kaggle Cartoon dataset.
    to CNN_direct_cartoon_classify.py

2022/07/20 = modified the Kaggle/hotdog_classify to use convolutional layers
    to CNN_direct_hotdog_classify.py

2022/07/20 = copied in hotdog classification from Kaggle.com
    to /_kaggle_tutorials.
    This was a great example of loading a dataset locally,
    then using it with a tensorflow neural network

2022/07/11 = Modified the TF/perceptron example to use the CIFAR dataset.
    to /cifar10_classify/CNN_ClassifyCifar10.py

2022/07/11 = Modified Perceptron_ClassifyCifar10 to include convolutional layers.
    to /cifar10_classify/Perceptron_ClassifyCifar10.py

2022/07/05 = Modified 73_DCGAN_HandwrittenDigits.py to not stop for displayed images,
    to add titles to windows, and to load previous learning from a checkpoint before training.
    to /mnistDigits_generate/DCGAN_HandwrittenDigits.py

2022/07/04 = Copied in a number of tutorials from Tensorflow.org
    to /_tensorflow_tutorials
