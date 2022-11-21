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


def create_generator_v3(output_channels):
    r"""Create generator using a smaller unet style design.  3x3x3"""

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # start model building
    skip0 = x = inputs

    # downsample layers
    skip1 = x = downsample( 128, 3,3 )(x)      # (bs, 86,86, 192 )
    # tf.print('Shape0=',str(tf.shape(x)))
    x = tf.keras.layers.ZeroPadding2D(1)(x)     # (bs, 88,88, 192 )
    # tf.print('Shape1=',str(tf.shape(x)))
    skip2 = x = downsample( 512, 3,3 )(x)      # (bs, 30,30, 512 )
    # tf.print('Shape2=',str(tf.shape(x)))

    ###############################
    # center layers
    x = downsample( 1024, 3,3 )(x)              # (bs, 10,10, 1024)
    # tf.print('Shape3=',str(tf.shape(x)))
    x = upsample(512, 3,3, padding='valid')(x)  # (bs, 30,30, 512)
    # tf.print('Shape4=',str(tf.shape(x)))

    ###############################
    # upsample layers

    x = merge_layer( x, skip2 )                 # (bs, 30,30, 512+512 )
    # tf.print('Shape5=',str(tf.shape(x)))
    x = upsample(512, 3, 3)(x)                  # (bs, 90,90, 512)
    # tf.print('Shape6=',str(tf.shape(x)))
    x = crop_layer( 2,2, 86,86 )(x)           # (bs, 86,86, 512)
    # tf.print('Shape7=',str(tf.shape(x)))

    x = merge_layer( x, skip1 )                 # (bs, 86,86, 512+192 )
    # tf.print('Shape8=',str(tf.shape(x)))
    x = upsample(192, 3, 3)(x)                  # (bs, 258,258, 192)
    # tf.print('Shape9=',str(tf.shape(x)))
    x = crop_layer( 1,1, 256,256 )(x)           # (bs, 256,256, 192)
    # tf.print('ShapeA=',str(tf.shape(x)))

    x = merge_layer( x, skip0 )                 # (bs, 256,256, 192+3)
    # tf.print('ShapeB=',str(tf.shape(x)))

    # cleanup to 3 channels
    x = tf.keras.layers.Conv2DTranspose( output_channels, 1, activation='tanh')(x)   # (bs, 256, 256, 3)
    # tf.print('ShapeC=',str(tf.shape(x)))

    outputs = x
    return tf.keras.Model(inputs=inputs, outputs=outputs)


########################################################################################################################
# construct datasets, instantiate cyclegan_runner
#

dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

OUTPUT_CHANNELS = 3

generator_first = create_generator_v3( OUTPUT_CHANNELS )
generator_second = create_generator_v3( OUTPUT_CHANNELS )


runner = mgr.cyclegan_runner(
        train_horses, train_zebras, test_horses, test_zebras,
        generator_first=generator_first, generator_second=generator_second,
        epochs=50, checkpoint_root='zebra_unet_terse_ckpt/v3' )

runner.run()

