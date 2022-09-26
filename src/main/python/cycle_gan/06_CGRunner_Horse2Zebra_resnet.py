#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   NOTE: this fifth modification to the script extracts most of the framework to a runner class
#         this makes it easier to change framework functionality ( saving, configuration, sampling )
#
#   This is a copy of 03_CycleGAN_Horse2Zebra_functionalApi.py with the following changes:
#       Cyclegan framework extracted to cyclegan_runner
#       InstanceNorm moved to tf_layer_tools along with some shape resizing methods.
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


class InstanceNormalization(tf.keras.layers.Layer):
    r"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset




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

# class MyAreaResize(tf.keras.layers.Layer):
#
#     def __init__(self, wide, tall):
#         super(MyAreaResize, self).__init__()
#         self.wide = wide
#         self.tall = tall
#
#     def build(self,other):
#         # tf.print("MAR.OTHER=",other)
#         return
#
#     def call(self, inputs):
#         work = tf.repeat( inputs, repeats=self.wide, axis=1 )
#         return tf.repeat( work, repeats=self.tall, axis=2 )
#
# def resizing_layer( size ):
#     return MyAreaResize( size, size )
#     # return tf.keras.layers.Resizing( size, size, interpolation='nearest' )
#     # return tf.keras.layers.Resizing( 8,8, interpolation='area' )
#     # return tf.keras.layers.UpSampling2D( size=4, interpolation='area' )

def merge_layer( x, y ):
    return tf.keras.layers.Concatenate()( [x, y] )
    # return tf.keras.layers.Add()( [x, y] )

# class MyCropSize(tf.keras.layers.Layer):
#
#     def __init__(self, y, x, tall, wide):
#         super(MyCropSize, self).__init__()
#         self.y = y
#         self.x = x
#         self.tall = tall
#         self.wide = wide
#
#     def build(self,other):
#         return
#
#     def call(self, inputs):
#         return tf.image.crop_to_bounding_box( inputs, self.y, self.x, self.tall, self.wide )
#
#
# def crop_layer( y, x, tall, wide ):
#     return MyCropSize( y, x, tall, wide )


def resnet_generator():
    r"""Starting generator using resnet blocks.
    Evaluation:
        Round1 = resnet blocks => created brown blocks
        Round2 = skip every 3 layers, concatenate => Creates pretty stripes, captures some shapes from original image.
        Round3 = more aggressive downsampling to save memory =>runs faster, about the same, some pontillism
    """

    output_channels = 3

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # start model building
    x = inputs
    skip1 = x

    # downsample layers
    x = downsample( 64, 4, 2 )(x)           # ( 128,128,64 )
    x = downsample( 128, 4, 2 )(x)          # (bs, 64, 64, 128)

    x1 = tf.keras.layers.Conv2D( 128, 4, strides=4, padding='same')(skip1)
    skip2 = x = merge_layer( x, x1 )        # (bs, 32, 32, 128/256)

    x = downsample( 256, 4, 2 )(x)          # (bs, 32, 32, 256)
    x = downsample( 512, 4, 2 )(x)          # (bs, 16, 16, 512)

    x2 = tf.keras.layers.Conv2D( 512, 4, strides=4, padding='same')(skip2)
    skip3 = x = merge_layer( x, x2 )                # (bs, 4, 4, 512/1024)

    x = downsample( 512, 4, 2 )(x)          # (bs, 8, 8, 512)
    x = downsample( 512, 4, 2 )(x)          # (bs, 4, 4, 512)

    x3 = tf.keras.layers.Conv2D( 512, 4, strides=4, padding='same')(skip3)
    skip4 = x = merge_layer( x, x3 )                # (bs, 4, 4, 512/1024)


    ###############################
    # middle layers
    x = downsample( 512, 4, 2 )(x)          # (bs, 2, 2, 512)
    x = downsample( 512, 4, 2 )(x)          # (bs, 1, 1, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 2, 2, 512)

    ###############################
    # upsample layers

    x4 = tf.keras.layers.Conv2D( 512, 4, strides=2, padding='same')(skip4)
    skip5 = x = merge_layer( x, x4 )                # (bs, 2, 2, 512/1024)

    x = upsample(512, 4, 2)(x)          # (bs, 4, 4, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 8, 8, 512))

    x5 = tf.keras.layers.Conv2DTranspose( 512, 4, strides=4)(skip5)  # (bs, 8, 8, 512)
    skip6 = x = merge_layer( x, x5 )        # (bs, 8, 8, 512/1024))

    x = upsample(512, 4, 2)(x)              # (bs, 16, 16, 512))
    x = upsample(256, 4, 2)(x)               # (bs, 32, 32, 256)

    x6 = tf.keras.layers.Conv2DTranspose( 256, 4, strides=4)(skip6)  # (bs, 32, 32, 256)
    skip7 = x = merge_layer( x, x6 )        # (bs, 16, 16, 256/512))

    x = upsample(128, 4, 2)(x)               # (bs, 64, 64, 128)
    x = upsample(64, 4, 2)(x)               # (bs, 128, 128, 64)

    x7 = tf.keras.layers.Conv2DTranspose( 64, 4, strides=4)(skip7)  # (bs, 128, 128, 64)
    x = merge_layer( x, x7 )                                # (bs, 128, 128, 64/128)

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
        generator_first=resnet_generator(), generator_second=resnet_generator(),
        epochs=50, checkpoint_root='zebra_runner_ckpt' )

runner.run()

