#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   What is different here:
#       Model is using 'resnet' style skipping, with fewer layers.
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

from _utilities.tf_cyclegan_tools import *
from tf_layer_tools import *

import cyclegan_runner as mgr

########################################################################################################################
# construct model

def downsample(filters, size, strides=2):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, strides=2):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.ReLU())
    return result


def merge_layer( x, y ):
    return tf.keras.layers.Concatenate()( [x, y] )
    # return tf.keras.layers.Add()( [x, y] )


def resnet_generator( output_channels ):
    r"""Starting generator using resnet blocks.
    Evaluation:
        Round1 = resnet blocks => created brown blocks
        Round2 = skip every 3 layers, concatenate => Creates pretty stripes, captures some shapes from original image.
        Round3 = more aggressive downsampling to save memory =>runs faster, about the same, some pontillism
    """

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # start model building
    x = inputs
    skip1 = x

    # downsample layers
    x = downsample( 64, 5, 3 )(x)    # ( 86,86,64 )
    x = downsample( 128, 5, 3 )(x)  # (bs, 29, 29, 128)

    x1 = tf.keras.layers.Conv2D( 128, 11, strides=9, padding='same')(skip1)
    skip2 = x = merge_layer( x, x1 )    # (bs, 29, 29, 128/256)

    x = downsample( 256, 5, 3 )(x)      # (bs, 10, 10, 256)
    x = downsample( 512, 5, 3 )(x)      # (bs, 4, 4, 512)

    x2 = tf.keras.layers.Conv2D( 512, 11, strides=9, padding='same')(skip2)
    skip3 = x = merge_layer( x, x2 )        # (bs, 4, 4, 512/1024)

    x = downsample( 512, 4, 2 )(x)           # (bs, 2, 2, 512)

    ###############################
    # center layers
    # x = tf.keras.layers.Flatten()(x)                    # ( bs, 2048 )
    # x = tf.keras.layers.Dense( 2048 )(x)                # ( bs, 2048 )
    # x = tf.keras.layers.Reshape( ( 2, 2, 512 ) )(x)     # (bs, 2, 2, 512)

    x = downsample( 512, 4, 2 )(x)           # (bs, 1, 1, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 2, 2, 512)

    ###############################
    # upsample layers
    x = upsample(512, 4, 2)(x)          # (bs, 4, 4, 512)

    skip4 = x = merge_layer( x, skip3 )  # (bs, 4, 4, 512/1536) - no convolutions

    x = upsample(256, 5, 3 )(x)         # (bs, 12, 12, 256)
    x = crop_layer( 1, 1, 10, 10 )(x)   # (bs, 10, 10, 256)
    x = upsample(128, 5, 3)(x)          # (bs, 30, 30, 128))

    x4 = tf.keras.layers.Conv2DTranspose( 128, 10, strides=8)(skip4)  # (bs, 34, 34, 128)
    x4 = crop_layer( 2, 2, 30, 30 )(x4)     # (bs, 30, 30, 128)
    skip5 = x = merge_layer( x, x4 )        # (bs, 30, 30, 128/256))

    x = upsample(64, 5, 3)(x)               # (bs, 90, 90, 64)
    x = crop_layer( 2, 2, 86, 86 )(x)       # (bs, 86, 86, 64)
    x = upsample(32, 5, 3)(x)               # (bs, 258, 258, 32)
    x = crop_layer( 1, 1, 256, 256 )(x)     # (bs, 256, 256, 32)))

    x5 = tf.keras.layers.Conv2DTranspose( 32, 11, strides=9)(skip5)  # (bs, 270, 270, 32)
    x5 = crop_layer( 7, 7, 256, 256 )(x5)       # (bs, 256, 256, 32)
    x = merge_layer( x, x5 )                    # (bs, 256, 256, 32/64)

    # cleanup to 3 channels
    x = tf.keras.layers.Conv2DTranspose( output_channels, 1, activation='tanh')(x)   # (bs, 256, 256, 3)

    outputs = x
    return tf.keras.Model(inputs=inputs, outputs=outputs)

########################################################################################################################
# construct datasets, instantiate cyclegan_runner
#

dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

runner = mgr.cyclegan_runner(
        train_horses, train_zebras, test_horses, test_zebras,
        generator_first=resnet_generator(), generator_second=resnet_generator(),
        epochs=50, checkpoint_root='zebra_resnet_terse_ckpt' )

runner.run()

