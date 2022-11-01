#
#   functions for use in land_and_sea, isolated for testing
import sys
sys.path.append('..')

import tensorflow as tf
import math

import _utilities.tf_tensor_tools as teto


# number of possible output values, one value for each terrain type
TERRAIN_TYPE_COUNT = 2
PI = math.pi

# one color per terrain type
TERRAIN_ONE_HOT_COLOR = tf.constant( ((0,255,0),(0,0,255)) )

TERRAIN_TYPE_GOAL = tf.Variable( [0.5,0.5] )
TERRAIN_SURFACE_GOAL = tf.Variable( 0.5 )



@tf.function
def set_terrain_type_goal( ttg ):
    global TERRAIN_TYPE_GOAL
    TERRAIN_TYPE_GOAL.assign( ttg )

@tf.function
def set_terrain_surface_goal( tsg ):
    global TERRAIN_SURFACE_GOAL
    TERRAIN_SURFACE_GOAL.assign( tsg )


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

@tf.function
def round_vee( x ):
    r"""-1 is at x=0.  1 is at x=+/-1"""
    return -tf.math.sin( (x+0.5) * PI )

@tf.function
def round_peak( x ):
    r"""1 is at x=0.  -1 is at x=+/-1"""
    return tf.math.sin( (x+0.5) * PI )


@tf.function
def terrain_loss( y_true, y_pred ):

    r"""y_pred is a 2d tensor of soft logits indicating terrain for each tile.
    In this first draft, 0=sea and 1=land
    Expected shape is: (batch_size, wide,tall, type_count )
    Loss is based on count of tile types, which is approximated by summing soft logits.
    Loss is also based on certainty, which is defined as being close to 0 or 1, not 0.5
    """

    # print('y_true.shape=',y_true.shape)
    # print('y_pred.shape=',y_pred.shape)
    # if True: return [0.]

    sqdf = tf.math.squared_difference( y_true, y_pred )
    # print('sqdf=',sqdf)
    template_mse = tf.math.reduce_mean( sqdf )
    # print('template_mse=',template_mse)

    # move values close to one_hot
    sm_y_pred = teto.supersoftmax( y_pred )

    terrain_loss = terrain_type_loss( sm_y_pred )
    surface_loss = terrain_surface_loss( sm_y_pred )

    with tf.GradientTape() as t:
        t.watch( template_mse )
        t.watch( terrain_loss )
        t.watch( surface_loss )

    return template_mse + terrain_loss + surface_loss


@tf.function
def terrain_type_loss(y_pred):

    # tf.print("y_pred=",y_pred)
    # tf.print("y_pred.shape=",y_pred.shape)

    [batch,wide,tall,deep] = y_pred.shape[0:4]
    size = tf.cast( wide * tall, tf.float32 )
    # desired = size/deep

    counts = tf.reduce_sum( y_pred, axis=2)
    counts = tf.reduce_sum( counts, axis=1)
    # tf.print("counts=",counts)
    type_ratio = counts / size
    # tf.print("type_ratio=",type_ratio,'shape=',type_ratio.shape)
    type_distance = type_ratio - TERRAIN_TYPE_GOAL
    # tf.print("type_distance=",type_distance)

    veed = vee(type_distance)
    # tf.print("veed=",veed)
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
    It is a crude representation of the fractal level of the output.
    Cells are logits summing to 1.  Values are in range [0,1]
    Same values = -1; Max difference is 90 degrees = -0
    Cells cannot actually be opposite ( which would be +1 )"""
    # tf.print('start=',y_pred)

    [batch,wide,tall,deep] = y_pred.shape[0:4]
    size = tf.cast( wide * tall, tf.float32 )

    rollh = tf.roll( y_pred, 1, 2  )
    # tf.print('rollh=',rollh)
    diffh = tf.keras.losses.cosine_similarity(y_pred, rollh, axis=-1 )
    # tf.print('diffh=',diffh)

    rollv = tf.roll( y_pred, 1, 1  )
    # tf.print('rollv=',rollv)
    diffv = tf.keras.losses.cosine_similarity(y_pred, rollv, axis=-1 )
    # tf.print('diffv=',diffv)

    diff_avg= ( diffh + diffv ) / 2.
    # tf.print('diff_avg=',diff_avg)

    same_count = tf.reduce_sum( tf.reduce_sum( diff_avg, axis=-1 ), -1 ) / size
    # tf.print('same_count(negative)=',same_count)
    change_count = 1. + same_count
    # tf.print('change_sum=',change_count)
    # tf.print('TERRAIN_SURFACE_GOAL=',TERRAIN_SURFACE_GOAL)
    result = 2 * vee( change_count - TERRAIN_SURFACE_GOAL )
    # tf.print('result=',result)

    # TODO: surface similarity multiplied by type vector produces type/similarity vector
    #       and goal is expressed as vector of similarity by type
    return result

########################################################################################################################
#   Image Processing

TEMPLATE_SCALE = 32768.0
TEMPLATE_RANGE = TEMPLATE_SCALE / TERRAIN_TYPE_COUNT

def image_to_template( image ):
    r"""Create a 'template' image created from the available terrain types.
    The goal is to bring some of the original image information into the result,
        by adding a loss value based on divergence from the template.
    image = is shape (batch, WIDE, TALL, CHANNELS) of float [0,1)
    result = is shape (batch, WIDE, TALL, TERRAIN_TYPE_COUNT) of int(0/1)"""

    # tf.print( 'image=', image )

    # shift from [0,1) representation
    work = tf.cast( tf.multiply( image, TEMPLATE_SCALE ), tf.int16 )
    # tf.print( 'work=', work )

    # split into three layers
    w1,w2,w3 = tf.split( work, 3, axis=-1)
    # work = tf.slice( (0,0,0,0), (1,1,1,2), work )
    # tf.print( 'w1=', w1 )
    # tf.print( 'w2=', w2 )
    # tf.print( 'w3=', w3 )

    # combine bits for all three layers
    work = tf.bitwise.bitwise_xor( w1,w2)
    # tf.print('xor1=',work)
    work = tf.bitwise.bitwise_xor( work,w3)
    # tf.print('xor2=',work)

    # divide by range, int for terrain type, one-hot for comparison
    work = tf.cast( tf.divide( tf.cast(work,tf.float32), TEMPLATE_RANGE ), tf.int32 )
    # tf.print('count=',work)

    work = tf.one_hot( work, TERRAIN_TYPE_COUNT )
    # tf.print('one_hot=',work)

    work = tf.squeeze( work, axis=-2 )
    # tf.print('squeeze=',work)

    return work