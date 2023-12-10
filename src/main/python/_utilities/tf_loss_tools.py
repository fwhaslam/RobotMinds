#
#   Utilities related to loading and saving information to the file system
#

import tensorflow as tf
import math

PI = math.pi


@tf.function
def valley(x):
    r"""Linear V shape, 0 at x=0, and 1 at x=-1|+1"""
    return tf.where( x<0, -x, x )


@tf.function
def peak(x):
    r"""Linear inverted V shape, 0 at x=0, and -1 at x=-1|+1"""
    return tf.where( x<0, x, -x )


@tf.function
def round_valley( x ):
    r"""-1 is at x=0.  1 is at x=+/-1"""
    return -tf.math.sin( (x+0.5) * PI )

@tf.function
def round_peak( x ):
    r"""1 is at x=0.  -1 is at x=+/-1"""
    return tf.math.sin( (x+0.5) * PI )
