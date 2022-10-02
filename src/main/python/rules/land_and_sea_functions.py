#
#   functions for use in land_and_sea, isolated for testing

import tensorflow as tf
import math


# number of possible output values, one value for each terrain type
TERRAIN_TYPE_COUNT = 2
TERRAIN_LOSS_OFFSET = tf.constant( tf.range(0, TERRAIN_TYPE_COUNT) )
# tf.print("offset=",TERRAIN_LOSS_OFFSET)

SURFACE_GOAL = 0.5

PI = math.pi
HALF = 0.5

@tf.function
def vee(x):
    r"""Linear V shape, 0 at x=0, and 1 at x=-1|+1"""
    return tf.where( x<0, -x, x )
    # if (x<0.5):
    #     return 1-2*x
    # else:
    #     return 2*x-1

@tf.function
def peak(x):
    r"""Linear inverted V shape, 0 at x=0, and -1 at x=-1|+1"""
    return tf.where( x<0, x, -x )
    # if (x<0.5):
    #     return 2*x
    # else:
    #     return 2*(1-x)

@tf.function
def round_vee( x ):
    r"""-1 is at x=0.  1 is at x=+/-1"""
    return -tf.math.sin( (x+HALF) * PI )

@tf.function
def round_peak( x ):
    r"""1 is at x=0.  -1 is at x=+/-1"""
    return tf.math.sin( (x+HALF) * PI )


@tf.function
def terrain_loss( y_true, y_pred ):

    r"""y_pred is a 2d tensor of soft logits indicating terrain for each tile.
    In this first draft, 0=sea and 1=land
    Expected shape is: (batch_size, wide,tall, type_count )
    Loss is based on count of tile types, which is approximated by summing soft logits.
    Loss is also based on certainty, which is defined as being close to 0 or 1, not 0.5
    """

    type_of_terrain_loss = terrain_type_loss( y_pred )

    certainty_loss = terrain_certainty_loss( y_pred )

    surface_loss = terrain_surface_loss( y_pred )

    return type_of_terrain_loss + certainty_loss + surface_loss


@tf.function
def terrain_type_loss(y_pred):

    # tf.print("y_pred=",y_pred)
    # tf.print("y_pred.shape=",y_pred.shape)

    [batch,wide,tall,deep] = y_pred.shape[0:4]
    size = tf.cast( wide * tall, tf.float32 )
    desired = size/deep

    counts = tf.reduce_sum( y_pred, axis=2)
    counts = tf.reduce_sum( counts, axis=1)
    # tf.print("counts=",counts)
    type_distance = ( counts - desired ) / desired
    # tf.print("type_distance=",type_distance)

    veed = vee(type_distance)
    # tf.print("peaked=",veed)
    result= tf.reduce_mean( veed, axis=-1 )
    # tf.print("result=",result)
    return result

@tf.function
def terrain_certainty_loss(y_pred):
    r"""determine certainty loss :: closer to 0 or 1 than 0.5 is better ---"""
    # tf.print("input=",y_pred)
    # tf.print("input.shape=",y_pred.shape)
    work = 2 * ( y_pred - 0.5 )
    # tf.print('scaled=',work)
    work = 1. + peak( work )
    # tf.print('veed=',work)
    work = tf.reduce_mean( work, axis=-1 )
    work = tf.reduce_mean( work, axis=-1 )
    work = tf.reduce_mean( work, axis=-1 )
    # tf.print('mean(mean(mean(work)=',work)
    # tf.map_fn( tf.reduce_mean, work, fn_output_signature=(1,) )
    # tf.print('reduced(work)=',work)
    return work


@tf.function
def terrain_surface_loss( y_pred ):
    r"""'Surface' is a measure of the terrain changes accross cells.
    It is a crude representation of the fractal level of the output."""
    # tf.print('start=',y_pred)

    [batch,wide,tall,deep] = y_pred.shape[0:4]
    size = tf.cast( wide * tall, tf.float32 )

    rollh = tf.roll( y_pred, 1, 2  )
    # tf.print('rollh=',rollh)
    diffh = -tf.keras.losses.cosine_similarity(y_pred, rollh, axis=-1 )
    # tf.print('diffh=',diffh)

    rollv = tf.roll( y_pred, 1, 1  )
    # tf.print('rollv=',rollv)
    diffv = -tf.keras.losses.cosine_similarity(y_pred, rollv, axis=-1 )
    # tf.print('diffv=',diffv)

    diff_avg= ( diffh + diffv ) / 2.
    # tf.print('diff_avg=',diff_avg)

    # add together a count of 'similar', so range[0:size]
    #   then reduce to value [0:1] where zero means close to goal, and 1 means far from goal
    reduce_sum = tf.reduce_sum( tf.reduce_sum( diff_avg, axis=-1 ), -1 ) / size
    # tf.print('reduce_sum=',reduce_sum)
    result = 2 * vee( reduce_sum - SURFACE_GOAL )
    # tf.print('result=',result)

    # TODO: surface similarity multiplied by type vector produces type/similarity vector
    #       and goal is expressed as vector of similarity by type
    return result

