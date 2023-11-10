#
#   This variation uses random images as a starting point.
#
#   built from python/rules/v1/lnz_pics_via_runner.py
#

import sys
sys.path.append('../..')

# common
import os as os
import numpy as np
import random as rnd
from pathlib import Path
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# tensorflow
import tensorflow as tf
# according to the designers, the only supported import method is to use 'tf.component'
#       so import image, keras =>  tf.image and tf.keras
# from tensorflow import image, keras


# from land_and_sea_functions import *
# from _utilities.tf_tensor_tools import *
# from cycle_gan.tf_layer_tools import *
import _utilities.tf_tensor_tools as teto
import _utilities.tf_loading_tools as loto
from gaussian_layer import GaussianLayer


# tf.compat.v1.enable_eager_execution()

print(tf.__version__)

plt.ion()

SAMPLE_COUNT = 1000
TERRAIN_TYPE_COUNT = 2
TERRAIN_ONE_HOT_COLOR = tf.constant( ((0,255,0),(0,0,255)) )

WIDE = 10
TALL = 10

INPUT_SHAPE = ( TERRAIN_TYPE_COUNT )
IMAGE_SHAPE = ( WIDE, TALL )
IMAGE_UNITS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

OUTPUT_SHAPE = ( WIDE, TALL, TERRAIN_TYPE_COUNT )
OUTPUT_UNITS = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]
# MAP_SHAPE =  ( 1, 2)
# IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]

EPOCHS = 10
BATCH_SIZE=100

flavor = "feature2random"

#######################################################################
#   Random Image Dataset

# def random_image(wide,tall,channels):
#     r"""Create a grid of floats in range [0-1)."""
#     grid = np.empty((wide,tall,channels))
#     for col in grid:
#         for row in range(tall):
#             for chan in range(channels):
#                 col[row][chan] = rnd.random()
#     return grid
#
# def load_random_features( count ):
#     features = np.empty((count,) + INPUT_SHAPE)
#     for index in range(count):
#         features[index] = random_image(*INPUT_SHAPE)
#     return features

def load_features( count ):
    r"""
    shape = [batch,N] ( initially N=2 )
    :param count:
    :return:
    """
    shape = ( count, INPUT_SHAPE )
    tf.print("Feature SHAPE=",shape)
    features = np.empty( shape )
    for index in range(count):
        ratio = index / ( count - 1. )
        features[index][0] = ratio
        features[index][1] = 1. - ratio
    return features

#######################################################################

def feature_to_result( feature_set ):
    count = len(feature_set)    # tf.shape( feature_set )[0]
    shape = ( count, ) + OUTPUT_SHAPE
    tf.print("Output SHAPE=",shape)
    results = np.empty( shape )
    for index in range(count):
        results[index][0][0] = feature_set[index]
    return results

#######################################################################

feature_set = load_features( SAMPLE_COUNT )
print("FeatureSet.shape=",tf.shape(feature_set))

result_set = feature_to_result( feature_set )
print("ResultSet.shape=",tf.shape(feature_set))

# scramble with same seed
tf.random.shuffle(feature_set, 12345)
tf.random.shuffle(result_set, 12345)

tflen = len(feature_set)
train_segment = (int)(tflen * .8)
print("train_segment=",train_segment)


train_data = feature_set[0: train_segment]
train_result = result_set[0: train_segment]
test_data = feature_set[train_segment: tflen]
test_result = result_set[train_segment: tflen]

print("Original len=",tflen)
print("TrainImage len=", len(train_data))
print("TestImage len=", len(test_data))

# inspect first image in dataset

lastFigure = None       # record the last displayed figure so it can be closed automatically

########################################################################################################################
# create model

layer_marker = 0
def make_layer_name(key):
    global layer_marker
    layer_marker += 1
    return key + '-' + str(layer_marker)

# def trim_layer_lrelu(filters,size,strides=1):
#     result = tf.keras.Sequential(name=make_layer_name('conv2d_norm_lrelu'))
#     result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
#     result.add(InstanceNormalization())
#     result.add(tf.keras.layers.LeakyReLU())
#     return result
#
# def grow_layer_lrelu(filters,size,strides=1):
#     result = tf.keras.Sequential(name=make_layer_name('inv_conv2d_norm_lrelu'))
#     result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same'))
#     result.add(InstanceNormalization())
#     result.add(tf.keras.layers.Dropout(0.5))
#     result.add(tf.keras.layers.LeakyReLU())
#     return result
#
# def trim_layer_selu(filters,size,strides=1):
#     return tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',activation='selu')
#
# def grow_layer_selu(filters,size,strides=1):
#     return tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',activation='selu')

########################################################################################################################

@tf.function
def vee(x):
    r"""Linear V shape, 0 at x=0, and 1 at x=-1|+1"""
    return tf.where( x<0, -x, x )

@tf.function
def peak(x):
    r"""Linear inverted V shape, 0 at x=0, and -1 at x=-1|+1"""
    return tf.where( x<0, x, -x )


@tf.function
def terrain_ratio_loss( y_goal_ratios, y_guess ):

    # tf.print("y_actual=",y_actual)
    # tf.print("y_actual.shape=",y_actual.shape)

    # tf.print("y_guess=",y_guess)
    # tf.print("y_guess.shape=",y_guess.shape)

    # reduce expected to logit ratios
    # goal = y_actual[0]

    # tf.print("y_goal_ratios=",y_goal_ratios)
    # tf.print("y_goal_ratios.shape=",y_goal_ratios.shape)

    [batch,wide,tall,deep] = y_guess.shape[0:4]
    size = tf.cast( wide * tall, tf.float32 )
    # desired = size/deep

    # sum on axis 3 ( keeping batch size )
    counts = tf.reduce_sum( y_guess, axis=2  )
    counts = tf.reduce_sum( counts, axis=1 )

    # tf.print("counts=",counts)
    type_ratio = counts / size
    # tf.print("type_ratio=",type_ratio)
    # tf.print('type_ratio.shape=',type_ratio.shape)
    type_distance = type_ratio - y_goal_ratios
    # tf.print("type_distance=",type_distance)

    veed = vee(type_distance)
    # tf.print("veed=",veed)
    result = tf.reduce_mean( veed, axis=-1 )
    # tf.print("result=",result)
    return result


@tf.function
def terrain_hard_ratio_loss( y_goal_ratios, y_guess ):

    y_hard_guess = teto.supersoftmax( y_guess )
    return terrain_ratio_loss( y_goal_ratios, y_hard_guess )


@tf.function
def terrain_certainty_loss( y_guess ):
    r"""determine certainty loss :: closer to 0 or 1 than 0.5 is better ---"""
    # tf.print("input=",y_pred)
    # tf.print("input.shape=",y_pred.shape)
    work = 2 * ( y_guess - 0.5 )
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
def terrain_loss( y_actual, y_guess ):

    r"""y_actual is a 2d tensor of soft logits indicating terrain for each tile.
    In this first draft, 0=sea and 1=land
    Expected shape is: (batch_size, wide,tall, type_count )
    Loss is based on count of tile types, which is approximated by summing soft logits.
    Loss is also based on certainty, which is defined as being close to 0 or 1, not 0.5
    """

    # move values close to one_hot
    # sm_y_pred = teto.supersoftmax(y_guess)
    # terrain_loss = terrain_type_loss( y_actual, sm_y_pred )

    ratio_goals = tf.slice( y_actual, [0,0,0,0], [-1,1,1,TERRAIN_TYPE_COUNT] )

    ratio_loss = terrain_ratio_loss( ratio_goals, y_guess )
    # certainty_loss = terrain_certainty_loss( y_guess )
    # terrain_loss = terrain_type_loss( sm_y_pred )
    # surface_loss = terrain_surface_loss( sm_y_pred )

    terrain_loss = ratio_loss

    with tf.GradientTape() as t:
        # t.watch( template_mse )
        t.watch( terrain_loss )
        t.watch( ratio_loss )
        # t.watch( certainty_loss )
        # t.watch( surface_loss )

    return terrain_loss

@tf.function
def make_random( shape ):
    print( "make random shape=", shape )
    batch = shape[0]
    if (batch==None):
        # return tf.placeholder( tf.float32, shape )
        return tf.keras.Input( shape, dtype=tf.dtypes.float32)
    return tf.random.normal( shape )


def create_model_v1( shape ):

    image_units = IMAGE_UNITS
    decode_units = 8

    # single input value, [0,1]
    x = inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(x)

    # decode from single value to array
    x = tf.keras.layers.Dense( decode_units, activation='ReLU', name='decode1')(x)
    x = tf.keras.layers.Dense( decode_units, activation='ReLU', name='decode2')(x)
    tf.print("x.shape=",x.shape)

    # build array of random fixed values ( to be replaced with random )
    # y = keras.layers.Lambda( lambda x: 0., output_shape=image_units )( inputs )
    y = inputs
    y = tf.keras.layers.Dense( image_units, name='Rando1' )( y )
    y = GaussianLayer( name='Rando2' )( y )
    tf.print("y.shape=",y.shape)

    # join feature encoding to random image
    x = tf.keras.layers.Concatenate( name='FeatureAndRandom')( [x, y] )
    tf.print("x.shape=",x.shape)

    x = tf.keras.layers.Dense(image_units,activation='LeakyReLU', name='Eval1')(x)
    x = tf.keras.layers.Dense( OUTPUT_UNITS ,activation='LeakyReLU', name='Eval2')(x)

    # output to 'softmax' which means two values that determine one pixel
    x = tf.keras.layers.Reshape( OUTPUT_SHAPE )(x)
    x = tf.keras.layers.Activation( 'softmax' )(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='basic_model_v1')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model

########################################################################################################################


model = create_model_v1(INPUT_SHAPE)

ckpt_folder = 'landnsea_ckpt/v2/ratio_random'


########################################################################################################################

def to_display_text( goals, results ):

    ratios = terrain_ratio_loss( goals, results )
    hards = terrain_hard_ratio_loss( goals, results )
    sures = terrain_certainty_loss( results )

    texts = [''] * 9
    for x in range(9):
        goal = "%.3f" % goals[x][0]
        ratio = "%.3f" % teto.tensor_to_value( ratios[x] )
        hard = "%.3f" % teto.tensor_to_value( hards[x] )
        sure = "%.3f" % teto.tensor_to_value( sures[x] )
        texts[x] = "G="+goal+" - S="+sure+"\nR="+ratio+" - H="+hard

    return  texts


def to_display_image( image_values, one_hot_color ):

    onehot = image_values
    # tf.print('work/squeeze=',tf.shape(work))

    onehot = tf.round( onehot )
    # tf.print('work/round=',work)
    onehot = tf.cast(onehot, tf.int32)
    # tf.print('work/cast=',work)

    return tf.tensordot( onehot, one_hot_color, 1 )


def display_text_and_image( texts, images ):
    r"""Work is an array of images with descriptive text"""

    global lastFigure
    plt.close( lastFigure )
    title = flavor+'@'+model.name
    lastFigure = plt.figure( title, figsize=(9, 10) )

    plt.rcParams.update({
        'font.size': 8,
        'text.color': 'black'})

    for index in range(9):
        sub = plt.subplot( 3, 3, 1+index )
        sub.title.set_text( texts[index] )
        plt.axis('off')
        plt.imshow( images[index] )

    plt.show()
    plt.pause( 500 )
    return

def display_results( sample_goals ):

    # assume 9 values
    results = model( sample_goals )
    display = to_display_image( results, TERRAIN_ONE_HOT_COLOR )

    for x in range(9):
        tf.print("INDEX="+str(x))
        tf.print( results[x], summarize=-1 )

    text = to_display_text( sample_goals, results )
    display_text_and_image(text, display)


def run(model,
        train_data, train_result,
        test_data, test_result,
        ckpt_folder):

    model.summary()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = terrain_loss,
        metrics = ['accuracy'] )

    loto.try_load_weights( ckpt_folder, model )

    model.fit(
        x=train_data, y=train_result,
        epochs=EPOCHS,
        # validation_data=(test_images),
        batch_size=BATCH_SIZE )

    result = model.evaluate(x=test_data,y=test_result, verbose=2)

    tf.print('result=',result)
    # print('\ntest_loss:', test_loss)
    # print('\ntest_acc:', test_acc)

    ckpt_file = ckpt_folder + '/ckpt'
    model.save_weights( ckpt_file )

    # if not sys.exists(ckpt_folder): os.makedirs(ckpt_folder)
    # if not exists(filename): open(filename, 'a').close()


########################################################################################################################

run( model, train_data, train_result, test_data, test_result, ckpt_folder )

########################################################################################################################

work = np.empty( (9,TERRAIN_TYPE_COUNT) )
for index in range(9):
    ratio = index / 8.
    work[index][0] = ratio
    work[index][1] = 1. - ratio

sample_goals = tf.constant( work )

display_results(sample_goals)

