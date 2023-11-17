#
#   This variation uses cifar10 images as a starting point.
#
#   The idea is that more regular ( non-random ) images may be easier to map into fixed terrain ratios.
#

import sys
sys.path.append('../..')

# common
import os as os
import numpy as np
import random as rnd
from pathlib import Path
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

IMAGE_CHANNELS = 3
WIDE = TALL = 32
INPUT_SHAPE = (WIDE, TALL, IMAGE_CHANNELS)

FLAVOR = 'pics'

# MAP_SHAPE =  (WIDE, TALL, TERRAIN_TYPE_COUNT)
# IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]
# EPOCHS = 5  # 10

#######################################################################
#   cifar-10 object image dataset

# dataset = tf.keras.datasets.cifar10.load_data()
# select_images = dataset.cache().shuffle().take(1000)
# print('select_images=',select_images)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images / 255.0
# test_images = test_images / 255.0

object_features = tf.random.shuffle( train_images, 54321 )[:1000]
print("object_features=",tf.shape(object_features))

#######################################################################

features_linear = object_features # + random_features # + terrain_features
template_image_set_linear = image_to_template( features_linear )

# shuffle with same seed
tf.random.set_seed( 12345 )
features = tf.random.shuffle( features_linear )
tf.random.set_seed( 12345 )
template_image_set = tf.random.shuffle( template_image_set_linear )

tflen = len(features)
train_segment = (int)(tflen * .8)
print("train_segment=",train_segment)

# train_images = list(features)[ 0 : train_segment ]
# test_images = list(features)[ train_segment : tflen-train_segment ]
train_images = features[ 0 : train_segment ]
train_image_set = template_image_set[ 0 : train_segment ]
test_images = features[ train_segment : tflen ]
test_image_set = template_image_set[ train_segment : tflen ]

print("Original len=",tflen)
print("TrainImage len=",len(train_images))
print("TestImage len=",len(test_images))


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
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.001), terrain_loss

def create_model_v3( shape ):

    x = inputs = keras.Input(shape=shape)
    x = trim_layer_selu(128,4)(x)
    x = trim_layer_selu(128,4)(x)
    x = trim_layer_selu(128,4)(x)
    x = tf.keras.layers.Conv2D( TERRAIN_TYPE_COUNT, 1, strides=1, padding='same',activation='softmax')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v3')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.0001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model, tf.keras.optimizers.Adam(0.0001), terrain_loss

def create_model_v2( shape ):

    x = inputs = keras.Input(shape=shape)
    x = trim_layer_lrelu(128,4)(x)
    x = trim_layer_lrelu(128,4)(x)
    x = trim_layer_lrelu(128,4)(x)
    x = tf.keras.layers.Conv2D( TERRAIN_TYPE_COUNT, 1, strides=1, padding='same',activation='softmax')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v2')
    model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
                   loss=terrain_loss,
                   metrics=['accuracy'] )
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

ckpt_folder = 'landnsea_ckpt/cifar10/' + model_id

lnz_runner(
    FLAVOR,
    model, loss_function, optimizer
).run(
    train_images,train_image_set,
    test_images,test_image_set,
    ckpt_folder
)


