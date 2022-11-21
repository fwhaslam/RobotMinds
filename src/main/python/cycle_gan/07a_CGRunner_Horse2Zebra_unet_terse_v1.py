#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   What is different here:
#       Model is using 'resnet' style skipping, and the first and second generator share a central layer.
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


def downsample(filters, size, strides=2,padding='same'):
    r"""downsample for generator"""
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, strides=2, padding='same'):
    r"""upsample for generator"""
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,padding=padding))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.ReLU())
    return result

def merge_layer( x, y ):
    return tf.keras.layers.Concatenate()( [x, y] )
    # return tf.keras.layers.Add()( [x, y] )


def create_generator_v1(output_channels):
    r"""Create generator using a smaller unet style design.  4x3x2"""

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # start model building
    skip0 = x = inputs

    # downsample layers
    skip1 = x = downsample( 192, 4, 4 )(x)      # ( 64,64,128 )
    skip2 = x = downsample( 512, 3, 3 )(x)      # (bs, 22, 22, 512)
    # tf.print("Shape4=",tf.shape(x))

    ###############################
    # center layers
    x = downsample( 1024, 2, 2 )(x)             # (bs, 11, 11, 1024)
    x = upsample(512, 2, 2)(x)                 # (bs, 22, 22, 512)rf5ct
    # tf.print("Shape4=",tf.shape(x))

    ###############################
    # upsample layers

    x = merge_layer( x, skip2 )                 # (bs, 22, 22, 512+512 )
    x = upsample(512, 3, 3)(x)                  # (bs, 66, 66, 512)
    x = crop_layer( 1,1,64,64 )(x)              # (bs, 64, 64, 512)

    x = merge_layer( x, skip1 )                 # (bs, 64, 64, 512+128 )
    x = upsample(192, 4, 4)(x)                  # (bs, 256, 256, 128)

    x = merge_layer( x, skip0 )                 # (bs, 256, 256, 128+3)

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

OUTPUT_CHANNELS = 3

generator_first = create_generator_v1( OUTPUT_CHANNELS )
generator_second = create_generator_v1( OUTPUT_CHANNELS )


runner = mgr.cyclegan_runner(
        train_horses, train_zebras, test_horses, test_zebras,
        generator_first=generator_first, generator_second=generator_second,
        epochs=50, checkpoint_root='zebra_unet_terse_ckpt/v1' )

runner.run()

