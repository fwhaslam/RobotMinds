#
#   This variation uses random images as a starting point.
#
#   The question I want to answer is: can we start from completely random images and create acceptable ratios by rules?
#

import sys
sys.path.append('../..')

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
from _utilities.tf_tensor_tools import *
from _utilities.tf_layer_tools import *

# tf.compat.v1.enable_eager_execution()

print(tf.__version__)

plt.ion()

# set_terrain_type_goal( [0.7,0.3] )
# set_terrain_surface_goal( 0.2 )

# # check the dataset parameters
# print("train[s]=", str(train_images.shape) )
# print("len=", str(len(train_labels)) )
# print("train labels = ", str(train_labels.shape) )
# print("test labels = ", str(test_labels.shape) )
# print("image shape = ", str(test_images.shape) )
# print("len(labels) = ", len(test_labels) )

IMAGE_CHANNELS = 3
WIDE = TALL = 32

INPUT_SHAPE = (WIDE, TALL, IMAGE_CHANNELS)
MAP_SHAPE =  (WIDE, TALL, TERRAIN_TYPE_COUNT)
IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]

EPOCHS = 5  # 10
SKIP = 1       # only process one out of SKIP from available images :: 1 = keep all

FLAVOR = 'rand'

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

features = random_features  # + terrain_features
template_image_set = image_to_template( features )


tf.random.shuffle( features, 12345 )
tflen = len(features)
train_segment = (int)(tflen * .8)
print("train_segment=",train_segment)


train_images = features[ 0 : train_segment ]
train_image_set = template_image_set[ 0 : train_segment ]
test_images = features[ train_segment : tflen ]
test_image_set = template_image_set[ train_segment : tflen ]

print("Original len=",tflen)
print("TrainImage len=",len(train_images))
print("TestImage len=",len(test_images))

# inspect first image in dataset

lastFigure = None       # record the last displayed figure so it can be closed automatically

########################################################################################################################
# create model

layer_marker = 0
def make_layer_name(key):
    global layer_marker
    layer_marker += 1
    return key + '-' + str(layer_marker)

def trim_layer_lrelu(filters,size,strides=1):
    result = tf.keras.Sequential(name=make_layer_name('conv2d_norm_lrelu'))
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def grow_layer_lrelu(filters,size,strides=1):
    result = tf.keras.Sequential(name=make_layer_name('inv_conv2d_norm_lrelu'))
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.LeakyReLU())
    return result

def trim_layer_selu(filters,size,strides=1):
    return tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',activation='selu')

def grow_layer_selu(filters,size,strides=1):
    return tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',activation='selu')

########################################################################################################################

def create_model_v5( shape ):

    units = TERRAIN_TYPE_COUNT * WIDE * TALL

    x = inputs = keras.Input(shape=shape)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units,activation='LeakyReLU')(x)
    x = keras.layers.Reshape( (WIDE,TALL,TERRAIN_TYPE_COUNT) )(x)

    x = trim_layer_lrelu(128,4)(x)
    x = tf.keras.layers.Conv2D( TERRAIN_TYPE_COUNT, 1, strides=1, padding='same',activation='softmax')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v5')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.0001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.0001), terrain_loss

def create_model_v4( shape ):

    x = inputs = keras.Input(shape=shape)
    # tf.print('x0=',x.shape)
    x = trim_layer_selu(64,2,2)(x)       # ( bs, 16,16, 64 )
    # tf.print('x1=',x.shape)
    x = trim_layer_selu(192,2,2)(x)       # ( bs, 8,8, 192 )
    x = trim_layer_selu(512,2,2)(x)       # ( bs, 4,4, 512 )
    # tf.print('x2=',x.shape)

    x = grow_layer_selu(512,2,2)(x)       # ( bs, 8,8, 512 )
    x = grow_layer_selu(192,2,2)(x)       # ( bs, 16,16, 192 )
    x = grow_layer_selu(64,2,2)(x)       # ( bs, 32,32, 64 )

    # x = tf.keras.layers.Conv2DTranspose( TERRAIN_TYPE_COUNT, 1, activation='tanh')(x)   # (bs, 32,32, 2) # tanh for image
    x = tf.keras.layers.Conv2DTranspose( TERRAIN_TYPE_COUNT, 1, activation='softmax')(x)   # (bs, 32,32, 2) # softmax for map
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v4')
    model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
                   loss=terrain_loss,
                   metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.001), terrain_loss

def create_model_v3( shape ):

    x = inputs = keras.Input(shape=shape)
    x = trim_layer_selu(128,4)(x)
    x = trim_layer_selu(128,4)(x)
    x = trim_layer_selu(128,4)(x)
    x = tf.keras.layers.Conv2D( TERRAIN_TYPE_COUNT, 1, strides=1, padding='same',activation='softmax')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v3')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.00001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.00001), terrain_loss

def create_model_v2( shape ):

    x = inputs = keras.Input(shape=shape)
    x = trim_layer_lrelu(128,4)(x)
    x = trim_layer_lrelu(128,4)(x)
    x = trim_layer_lrelu(128,4)(x)
    x = tf.keras.layers.Conv2D( TERRAIN_TYPE_COUNT, 1, strides=1, padding='same',activation='softmax')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v2')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.001), terrain_loss

def create_model_v1( shape ):

    units = TERRAIN_TYPE_COUNT * WIDE * TALL

    x = inputs = keras.Input(shape=shape)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units,activation='LeakyReLU')(x)
    x = keras.layers.Dense(units,activation='LeakyReLU')(x)
    # x = keras.layers.Dense(2048,activation='LeakyReLU')(x)
    x = keras.layers.Reshape( (WIDE,TALL,TERRAIN_TYPE_COUNT) )(x)
    x = keras.layers.Activation( 'softmax' )(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v1')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.001), terrain_loss

########################################################################################################################

model_id = 'v1'
if len(sys.argv)>1:
    model_id = sys.argv[1]
print('using model_id =',model_id)


model = None
match model_id:
    case 'v1':
        model, optimizer, loss_function = create_model_v1(INPUT_SHAPE)
    case 'v2':
        model, optimizer, loss_function = create_model_v2(INPUT_SHAPE)
    case 'v3':
        model, optimizer, loss_function = create_model_v3(INPUT_SHAPE)
    case 'v4':
        model, optimizer, loss_function = create_model_v4(INPUT_SHAPE)
    case 'v5':
        model, optimizer, loss_function = create_model_v5(INPUT_SHAPE)
    case _:
        print('unknown model_id =',model_id)
        sys.exit(-1)


########################################################################################################################
# create runner and drive process model

from lnz_runner import lnz_runner

ckpt_folder = 'landnsea_ckpt/random/' + model_id

lnz_runner(
    FLAVOR,
    model, loss_function, optimizer
).run(
    train_images,train_image_set,
    test_images,test_image_set,
    ckpt_folder
)

