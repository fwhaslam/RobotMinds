

import tensorflow as tf


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



class MyAreaResize(tf.keras.layers.Layer):
    r"""Replacement for ResizeArea which is messing up on GradientTape integration.
    This will increase the width and height by integral size, copying the single
    source value accross the new area.

    wide=2, tall=2
    inputs= [[[[0]
               [1]]

              [[2]
               [3]]]]
    result= [[[[0 0]
               [0 0]
               [1 1]
               [1 1]]

              [[2 2]
               [2 2]
               [3 3]
               [3 3]]]],
    """

    def __init__(self, wide, tall):
        super(MyAreaResize, self).__init__()
        self.wide = wide
        self.tall = tall

    def build(self,other):
        # tf.print("MAR.OTHER=",other)
        return

    def call(self, inputs):
        work = tf.repeat( inputs, repeats=self.wide, axis=1 )
        return tf.repeat( work, repeats=self.tall, axis=2 )

def resizing_layer( size ):
    return MyAreaResize( size, size )
    # return tf.keras.layers.Resizing( size, size, interpolation='nearest' )
    # return tf.keras.layers.Resizing( 8,8, interpolation='area' )
    # return tf.keras.layers.UpSampling2D( size=4, interpolation='area' )
