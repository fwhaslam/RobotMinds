#
#   functions for use in land_and_sea, isolated for testing
import sys
sys.path.append('../..')

import tensorflow as tf
import math
import os

import _utilities.tf_tensor_tools as teto

# number of possible output values, one value for each terrain type
TERRAIN_TYPE_COUNT = 4      # land, sea, peak, deep
PI = math.pi

# one color per terrain type
TERRAIN_ONE_HOT_COLOR = tf.constant(
    ((64,192,0),     # land
     (0,64,255),     # sea
     (80,80,80),     # peak
     (0,32,128))     # deep
)

TERRAIN_TYPE_GOAL = tf.Variable( [0.25,0.25,0.25,0.25] )
TERRAIN_SURFACE_GOAL = tf.Variable( 0.5 )

########################################################################################################################

def set_terrain_type_goal( ttg ):
    global TERRAIN_TYPE_GOAL
    TERRAIN_TYPE_GOAL.assign( ttg )
    print("Changing [TERRAIN_TYPE_GOAL] to", ttg )
    return

def set_terrain_surface_goal( tsg ):
    global TERRAIN_SURFACE_GOAL
    TERRAIN_SURFACE_GOAL.assign( tsg )
    print("Changing [TERRAIN_SURFACE_GOAL] to", tsg )
    return

# read values from environment, if they exist
def get_constants_from_environment():
    type_goal = os.getenv( 'TERRAIN_TYPE_GOAL' )
    if type_goal:
        set_terrain_type_goal( ast.literal_eval(type_goal) )
    surface_goal = os.getenv( 'TERRAIN_SURFACE_GOAL' )
    if surface_goal:
        set_terrain_surface_goal( ast.literal_eval(surface_goal) )
    return

get_constants_from_environment()

########################################################################################################################

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

    similarity_loss = terrain_similarity_loss( sm_y_pred, 2 ) + \
                      terrain_similarity_loss( sm_y_pred, 4 ) + \
                      terrain_similarity_loss( sm_y_pred, 8 )

    with tf.GradientTape() as t:
        t.watch( template_mse )
        t.watch( terrain_loss )
        t.watch( surface_loss )
        t.watch( similarity_loss )

    return template_mse + terrain_loss + surface_loss + similarity_loss


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

@tf.function
def terrain_similarity_loss( y_pred, shift=4 ):
    r"""Loss increases when segments of the map resemble each other.
    Value goes down ( improves ) as self-similarity decreases.
    :param y_pred: shape(bs, wide,tall, logits ) is matrix of logits indicating terrain type
    :return: shape(bs) is array of loss values, one per batch element
    """
    # tf.print('y_pred=',y_pred)

    # shift by two rows, then -RMSE
    shifted = tf.roll( y_pred, shift=shift, axis=1 )
    # tf.print('shifted=',shifted)
    loss1 = tf.map_fn( tf.reduce_mean, tf.square( y_pred - shifted ) )
    # loss1 = tf.reduce_mean( tf.reduce_mean( sqer, axis=-1 ), axis=-1 )
    # tf.print('loss=',loss1)

    shifted = tf.roll( shifted, shift=shift, axis=2 )
    loss2 = tf.map_fn( tf.reduce_mean, tf.square( y_pred - shifted ) )

    shifted = tf.roll( y_pred, shift=shift, axis=2 )
    loss3 = tf.map_fn( tf.reduce_mean, tf.square( y_pred - shifted ) )

    return - ( loss1 + loss2 + loss3 )

########################################################################################################################
#   Color Processing

RGB_2_XYZ_MATRIX = tf.transpose( tf.constant(
    [[.4124564, .3575761, .1874305],
     [.2126729, .7151522, .0711750],
     [.0193339, .1191920, .9503041]], dtype=tf.float32 ))

RGB_2_YUV_MATRIX = tf.transpose( tf.constant(
    [[  .299,    .587,    .114],
     [ -.14713, -.28886,  .436],
     [  .615,   -.51499, -.10001]], dtype=tf.float32 ))

# first five dimensions for eincode
EINCODE = [
    '',
    'a,ab->b',
    'ab,bc->ac',
    'abc,cd->abd',
    'abcd,de->abce',
    'abcde,ef->abcdf',
]

# see: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
def rgb_to_xyz( source ):
    r"""Will transform RGB values in axis=-1 to XYZ values, up to 5 dimensions.
    Hmmm, this fails to linearize the sRGB values before applying the matrix."""
    # print('source=',source)
    num_dims = source.shape.rank
    # print('num_dims=',num_dims)
    return tf.einsum(EINCODE[num_dims], source, RGB_2_XYZ_MATRIX)

# see https://stackoverflow.com/questions/5392061/algorithm-to-check-similarity-of-colors
# see: https://en.wikipedia.org/wiki/YUV
def rgb_to_yuv( source ):
    r"""Will transform RGB values in axis=-1 to YUV values, up to 5 dimensions.
    This is a decent approximation of human color perception."""
    # print('source=',source)
    return tf.einsum( EINCODE[source.shape.rank], source, RGB_2_YUV_MATRIX)

# convert color grid to yuv
TERRAIN_COLOR_YUV = rgb_to_yuv( tf.cast( TERRAIN_ONE_HOT_COLOR, tf.float32 ) / 256. )


# store color templates for reuse
color_template_map = {}

def terrain_color_template( frame ):
    r"""Reshape the color/yuv matrix so that it matches the image plus one dimension."""

    key = str(frame)
    if key in color_template_map: return color_template_map[key]

    colors = TERRAIN_COLOR_YUV
    for size in frame[::-1]:

        colors = tf.expand_dims( colors, axis=0 )
        # print("colors.shape=",colors.shape)

        if (not size is None) and (size>1):
            colors = tf.repeat( colors, repeats=size, axis=0 )
        # print("colors=",colors)
        # print("colors.shape=",colors.shape)

    # print("colors=",colors)

    color_template_map[key] = colors
    return colors

########################################################################################################################
#   Image Processing


def image_to_template( image ):
    r"""Create a 'template' image created from the available terrain types.
    The goal is to bring some of the original image information into the result,
        by adding a loss value based on divergence from the template.
    The newer algorithm is to take the original color, and find the most
        similar color in the TERRAIN_ONE_HOT_COLOR array.
    image = is shape (batch, WIDE, TALL, CHANNELS) of float [0,1)
    result = is shape (batch, WIDE, TALL, TERRAIN_TYPE_COUNT) of int(0/1)"""

    # tf.print( 'image=', image )

    image = tf.cast( image, tf.float32 )
    frame = image.shape[:-1]
    # print("frame=",frame)

    work = tf.expand_dims( image, axis=-2)
    # print('more_dims=',work)
    work = tf.repeat( work, TERRAIN_TYPE_COUNT, axis=-2)
    # tf.print('works.shape=',work.shape)
    # tf.print('work.dtype=',work.dtype)
    work = rgb_to_yuv( work )
    # tf.print('work.dtype=',work.dtype)

    # array of color matrix same shape as image pixels
    colors = terrain_color_template( frame )
    # tf.print('colors.dtype=',colors.dtype)

    # use distance to select MIN for each pixel in image (eg. the closest terrain color )
    work = tf.math.squared_difference( work, colors )
    # tf.print('diff2=',work)
    work = tf.reduce_sum( work, axis=-1)
    # tf.print("sum=",work)
    work = tf.argmin( work, axis=-1)
    # tf.print("min=",work)
    work = tf.one_hot( work, TERRAIN_TYPE_COUNT, axis=-1 )
    # tf.print("one_hot=",work)

    return work