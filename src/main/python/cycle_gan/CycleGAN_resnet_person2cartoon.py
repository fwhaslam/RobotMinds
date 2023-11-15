#
#   First Dataset is humans in human settings, Second dataset if cartoons in cartoon setting.
#       c2h = cartoon to human
#       h2c = human to cartoon
#
#   This is a cyclegan similar to the tensorflow example with two differences:
#       Instead of UNet this implements with resnet ( no long skipping, just short skipping )
#       There is a shared central layer and/or training to make the central layer similar.
#
#   Goal: cartoons are a lot more than just flat shading and dark outlines.
#       Human figures are parodies of real humans with fewer fingers, distorted body shapes,
#       and exaggerated facial features.  I am trying to train a central layer that represents
#       the image as an abstract concept of an object which is replaced with cartoon versionsz.
#       I don't expect this particular model to succeed.  I expect that I may have to approach
#       the level of DALL-E to see what I want.
#
#   Implementation One: the central layer is shared by four trainable models = h2h, c2c, h2c, c2h
#              the trainable models are composed of pairs of partial models:  h2, c2, 2h, 2c
#           h2h and c2c self train to reconstruct the original image
#           h2c <-> c2h = cycle gan style training with discriminators
#
#   Implementation Two: normal cycle gan training with discriminators of h2c <-> c2h
#           ALSO central layer trained to match over model pairs:  2c/h2, 2h/c2
#

import sys
sys.path.append('..')

# common
import os
from pathlib import Path

# tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf

# local tools
from src.main.python._utilities.tf_loading_tools import *

######################################
#   Human Figures and Settings
#

root_path = os.path.expanduser( 'd:/Datasets/OxfordHumanInteractions/tv_human_interactions_videos' )

video_files=[root_path + '/' + f for f in os.listdir(root_path)]

IMG_WIDTH = 320     # using 5:4 image ratio, the videos are usually about twice this (width,height)
IMG_HEIGHT = 256


for idx, path in enumerate(video_files):
    # Gather all its frames and add a batch dimension.
    frames = load_video( os.path.join(root_path, path), 1 )
    print("shape[",path,"] = ",tf.shape(frames))

    new_size = tf.image.resize( frames[0], (IMG_HEIGHT, IMG_HEIGHT) )
    draw_this = new_size / 256.

    # print("draw_this = ",draw_this)
    print("Shape[draw_this]=",tf.shape(draw_this))

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow( draw_this )
    # plt.imshow( frames[0], cmap=plt.cm.binary)
    plt.show()
    plt.pause(3)

    # time.sleep(12)


######################################
#   Adventure Time Figures and Settings
#

root_path = os.path.expanduser( '~/_Workspace/Datasets/KaggleCartoonTiny/cartoon_classification' )

# img_height = 180
# img_width = 320
# batch_size = 32

# fred: switch to match 'horse' image size
img_height = 256
img_width = 256


advent_train_ds = tf.keras.utils.image_dataset_from_directory(
    Path( root_path ) / "TRAIN" / "adventure_time",
    label_mode = None,
    image_size=(img_height, img_width),
    interpolation=funcs.preferred_resize_method
)


advent_test_ds = tf.keras.utils.image_dataset_from_directory(
    Path( root_path ) / "Test" / "adventure_time",
    label_mode = None,
    image_size=(img_height, img_width),
    interpolation=funcs.preferred_resize_method
)

train_setB =  advent_train_ds.unbatch().prefetch( BATCH_SIZE )
print("train_setB = ", train_setB )
test_setB = advent_test_ds.unbatch().prefetch( BATCH_SIZE )
print("test_setB = ", test_setB )
