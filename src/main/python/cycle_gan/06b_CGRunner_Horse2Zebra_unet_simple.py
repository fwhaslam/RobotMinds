#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   What is different here:
#       Model uses 'unet' style skipping, but only skips on the top two layers.
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

import tensorflow_datasets as tfds

from _utilities.tf_cyclegan_tools import *
from _utilities.tf_layer_tools import *
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


def unet_generator( output_channels=3 ):
    r"""Generator skips from front to back, unet style."""

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # start model building
    x = inputs

    # downsample layers
    skip0 = x = downsample( 64, 4, 2 )(x)           # ( 128,128,64 )
    skip1 = x = downsample( 128, 4, 2 )(x)          # (bs, 64, 64, 128)

    x = downsample( 256, 4, 2 )(x)          # (bs, 32, 32, 256)
    x = downsample( 512, 4, 2 )(x)          # (bs, 16, 16, 512)

    x = downsample( 512, 4, 2 )(x)          # (bs, 8, 8, 512)
    x = downsample( 512, 4, 2 )(x)          # (bs, 4, 4, 512)

    ###############################
    # middle layers
    x = downsample( 512, 4, 2 )(x)          # (bs, 2, 2, 512)
    x = downsample( 512, 4, 2 )(x)          # (bs, 1, 1, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 2, 2, 512)

    ###############################
    # upsample layers

    x = upsample(512, 4, 2)(x)          # (bs, 4, 4, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 8, 8, 512))

    x = upsample(512, 4, 2)(x)              # (bs, 16, 16, 512))
    x = upsample(256, 4, 2)(x)               # (bs, 32, 32, 256)

    x = upsample(128, 4, 2)(x)               # (bs, 64, 64, 128)
    x = merge_layer( x, skip1 )

    x = upsample(64, 4, 2)(x)               # (bs, 128, 128, 64)
    x = merge_layer( x, skip0 )

    # cleanup to 3 channels
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    outputs = last(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

########################################################################################################################
# construct datasets, instantiate cyclegan_runner
#

dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

runner = mgr.cyclegan_runner(
        train_horses, train_zebras, test_horses, test_zebras,
        generator_first=unet_generator(), generator_second=unet_generator(),
        epochs=50, checkpoint_root='zebra_unet_simple_ckpt' )

runner.run()

