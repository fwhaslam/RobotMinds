#
#   This variation starts from pictures, then tries to recreate the image from the generated map.
#
#   Double Output = the terrain map, and an attempt to recreate the original image.
#
#   The idea is that by retaining some information from the original, we will get terrain that resembles the image.
#

import sys
sys.path.append('..')

# common
import os as os
import numpy as np
import random as rnd
from pathlib import Path
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow import image, keras

from land_and_sea_functions import *
from cycle_gan.tf_layer_tools import *

# tf.compat.v1.enable_eager_execution()

print(tf.__version__)

plt.ion()

set_terrain_type_goal( [0.7,0.3] )
set_terrain_surface_goal( 0.2 )

# # check the dataset parameters
# print("train[s]=", str(train_images.shape) )
# print("len=", str(len(train_labels)) )
# print("train labels = ", str(train_labels.shape) )
# print("test labels = ", str(test_labels.shape) )
# print("image shape = ", str(test_images.shape) )
# print("len(labels) = ", len(test_labels) )

IMAGE_CHANNELS = 3
WIDE = TALL = 16
INPUT_SHAPE = (WIDE, TALL, IMAGE_CHANNELS)
MAP_SHAPE =  (WIDE*2, TALL*2, TERRAIN_TYPE_COUNT)
IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]
EPOCHS = 5

lastFigure = None       # record the last displayed figure so it can be closed automatically

FLAVOR = 'dbout'

#######################################################################
#   Random Image Dataset

def random_image(wide,tall,channels):
    r"""Create a grid of floats in range [0-1)."""
    grid = np.empty((wide,tall,channels))
    for col in grid:
        for row in range(tall):
            for chan in range(channels):
                col[row][chan] = rnd.random()
    return grid


def load_random_features( count ):
    features = np.empty((count,) + INPUT_SHAPE)
    for index in range(count):
        features[index] = random_image(*INPUT_SHAPE)
    return features

random_features = load_random_features( 1000 )
print("RandomFeaturesShape=",tf.shape(random_features))

#######################################################################

features = random_features
template_image_set = image_to_template( features )

tf.random.shuffle( features, 12345 )
tflen = len(features)
train_segment = (int)(tflen * .8)
print("train_segment=",train_segment)

train_images = features[ 0 : train_segment ]
train_templates = template_image_set[ 0 : train_segment ]
test_images = features[ train_segment : tflen ]
test_templates = template_image_set[ train_segment : tflen ]

print("Original len=",tflen)
print("TrainImage len=",len(train_images))
print("TestImage len=",len(test_images))

########################################################################################################################
# image_sets have two outputs to match the model.
#       first output is (bs, 32,32, TERRAIN_TYPE_COUNT), second output is (bs 16,16, 3)

doubler = ImageResize(2,2)

bigger_train_templates = doubler( train_templates )
train_image_set = [bigger_train_templates,train_images]
tf.print('bigger_train_templates=',bigger_train_templates.shape)

bigger_test_templates = doubler( test_templates )
test_image_set = [bigger_test_templates,test_images]
tf.print('bigger_test_templates=',bigger_test_templates.shape)


########################################################################################################################
# create model

def trim_layer(filters,size,strides=1):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.ReLU())
    return result

def grow_layer(filters,size,strides=1):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def trim_layer_selu(filters,size,strides=1):
    return tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',activation='selu')

def grow_layer_selu(filters,size,strides=1):
    return tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',activation='selu')

def create_model_v1d( shape ):
    r"""two outputs, one validates image reconstruction"""

    x = inputs = keras.Input(shape=shape)
    # tf.print('x0=',x.shape)
    x = trim_layer_selu(64,2)(x)                 # ( bs, 16,16, 64 )
    x = trim_layer_selu(128,2)(x)                         # ( bs, 16,16, 64 )

    x = grow_layer_selu(128,2,2)(x)                         # ( bs, 32,32, 64 )
    x = grow_layer_selu(128,2,1)(x)                          # ( bs, 32,32, 128 )

    output1 = tf.keras.layers.Conv2DTranspose( TERRAIN_TYPE_COUNT, 1, activation='softmax')(x)  # (bs, 32,32, 2) # softmax for map
    tf.print('output1=',output1.shape)
    output2 = tf.keras.layers.Conv2D( IMAGE_CHANNELS, 2,2, activation='tanh')(x)                # (bs, 16,16, 3) # tanh for image
    tf.print('output2=',output2.shape)

    model = tf.keras.Model(inputs=inputs, outputs=[output1,output2], name=FLAVOR+'_model_v1' )
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
    #                loss=[ terrain_loss, tf.keras.losses.MeanSquaredError() ],
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.001), [ terrain_loss, tf.keras.losses.MeanSquaredError() ]

########################################################################################################################

model_id = 'v1'
model, optimizer, loss_function = create_model_v1d(INPUT_SHAPE)

########################################################################################################################
# create runner and drive process model

from lnz_runner import lnz_runner

ckpt_folder = 'landnsea_ckpt/double/' + model_id

lnz_runner(
    FLAVOR,
    model, loss_function, optimizer
).run(
    train_images,train_image_set,
    test_images,test_image_set,
    ckpt_folder
)
