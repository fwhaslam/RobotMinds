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


# tf.compat.v1.enable_eager_execution()

print(tf.__version__)

plt.ion()

#######################################################################

def prepare_globals():

    global EPOCHS, BATCH_SIZE, SAMPLE_COUNT
    # EPOCHS = 5 # 50
    # BATCH_SIZE= 2 #1000
    # SAMPLE_COUNT = 8    # 1000
    EPOCHS = 50
    BATCH_SIZE= 1000
    SAMPLE_COUNT = 1000

    global TERRAIN_TYPE_COUNT, TERRAIN_ONE_HOT_COLOR
    TERRAIN_TYPE_COUNT = 2
    TERRAIN_ONE_HOT_COLOR = tf.constant( ((0,255,0),(0,0,255)) )

    global WIDE, TALL
    WIDE = 10
    TALL = 10

    global INPUT_SHAPE, IMAGE_SHAPE, IMAGE_UNITS
    INPUT_SHAPE = TERRAIN_TYPE_COUNT
    IMAGE_SHAPE = ( WIDE, TALL )
    IMAGE_UNITS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

    global OUTPUT_SHAPE, OUTPUT_UNITS
    OUTPUT_SHAPE = ( WIDE, TALL, TERRAIN_TYPE_COUNT )
    OUTPUT_UNITS = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]
    # MAP_SHAPE =  ( 1, 2)
    # IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]

    global flavor, ckpt_folder, lastFigure
    flavor = "feature2random"
    ckpt_folder = 'landnsea_ckpt/v3/ratio_random'
    lastFigure = None       # record the last displayed figure so it can be closed automatically

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
    for index in range(count):          # [ +0.0, +1.0 ]
        ratio = index / ( count - 1. )
        features[index][0] = ratio - 0.0
        features[index][1] = 1. - ratio
    return features

#######################################################################

def feature_to_result( feature ):
    # tf.print("feature shape=",tf.shape(feature))
    # tf.print("feature=",feature)
    ratio = feature       # [ +0.0, +1.0 ]
    limit = 100 * ratio
    block = np.empty( IMAGE_SHAPE )
    for dx in range( TALL ):
        for dy in range( WIDE ):
            x = dx * WIDE + dy
            block[dx][dy] = ( 1. if (x<limit) else 0. )
    return block

def feature_set_to_result( feature_set ):
    count = len(feature_set)    # tf.shape( feature_set )[0]
    shape = ( count, ) + IMAGE_SHAPE
    tf.print("Output SHAPE=",shape)
    results = np.empty( shape )
    for index in range(count):
        results[index] = feature_to_result( feature_set[index][0] )
    return results

#######################################################################

def prepare_data():
    feature_set_linear = load_features( SAMPLE_COUNT )
    print("FeatureSet.shape=",tf.shape(feature_set_linear))

    result_set_linear = feature_set_to_result( feature_set_linear )
    print("ResultSet.shape=",tf.shape(result_set_linear))

    # print("[50]=", feature_to_result( 0. ) );
    # print("[00]=", feature_to_result( -0.5 ) );
    # print("[99]=", feature_to_result( +0.5 ) );

    print("fs[000]=", feature_set_linear[0] )
    print("RS[000]=", result_set_linear[0] )
    print("fs[500]=", feature_set_linear[500] )
    print("RS[500]=", result_set_linear[500] )
    print("fs[999]=", feature_set_linear[999] )
    print("RS[999]=", result_set_linear[999] )


    # scramble with same seed
    tf.random.set_seed( 12345 )
    feature_set = tf.random.shuffle(feature_set_linear)
    tf.random.set_seed( 12345 )
    result_set = tf.random.shuffle(result_set_linear)


    print("POST SHUFFLE")
    print("fs[000]=", feature_set[0] )
    print("RS[000]=", result_set[0] )
    print("fs[500]=", feature_set[500] )
    print("RS[500]=", result_set[500] )
    print("fs[999]=", feature_set[999] )
    print("RS[999]=", result_set[999] )

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

    return train_data, train_result, test_data, test_result

########################################################################################################################
# create model

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
def make_random( shape ):
    print( "make random shape=", shape )
    batch = shape[0]
    if (batch==None):
        # return tf.placeholder( tf.float32, shape )
        return tf.keras.Input( shape, dtype=tf.dtypes.float32)
    return tf.random.normal( shape )


def create_model_v1( shape ):

    # input is a dual value [ x, 1-x ]
    x = inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(x)

    # decode from single value to array
    x = tf.keras.layers.Dense( 8, activation='selu', name='expand1')(x)
    x = tf.keras.layers.Dense( 32, activation='selu', name='expand2')(x)
    x = tf.keras.layers.Dense( 128, activation='selu', name='expand3')(x)
    x = tf.keras.layers.Dense( 512, activation='selu', name='expand4')(x)
    x = tf.keras.layers.Dense( IMAGE_UNITS, activation='selu', name='finale')(x)

    # output to 'softmax' which means two values that determine one pixel
    x = tf.keras.layers.Reshape( IMAGE_SHAPE )(x)
    x = tf.keras.layers.Activation( 'sigmoid' )(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='basic_model_v1')
    # model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
    #                loss=terrain_loss,
    #                metrics=['accuracy'] )
    return model

########################################################################################################################

def to_display_text( goals, results ):

    ratios = terrain_ratio_loss( goals, results )
    hards = terrain_hard_ratio_loss( goals, results )
    # sures = terrain_certainty_loss( results )

    texts = [''] * 9
    for x in range(9):
        goal = "%.3f" % goals[x][0]
        # ratio = "%.3f" % teto.tensor_to_value( ratios[x] )
        # hard = "%.3f" % teto.tensor_to_value( hards[x] )
        # sure = "%.3f" % teto.tensor_to_value( sures[x] )
        texts[x] = "G="+goal    # +"\nR="+ratio+" - H="+hard

    return  texts


def to_display_image( image_values, one_hot_color ):

    onehot = tf.round( image_values )
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

def simple_activation_to_one_hot( results ):

    work_shape = tf.shape( results)
    # print("DISPLAY IMAGE NEW work_shape = ",work_shape)
    if ( 4 == len( work_shape ) ):
        return results

    shape = np.append( work_shape, TERRAIN_TYPE_COUNT )
    # print("DISPLAY IMAGE NEW RATIO = ",shape)

    work = np.empty( shape )
    for b in range( shape[0] ):
        for x in range(IMAGE_UNITS):
            dx = (int)(x/WIDE)
            dy = x%WIDE
            ratio = results[b][dx][dy]
            work[b][dx][dy][0] = ratio
            work[b][dx][dy][1] = 1. - ratio

    return work


def display_results( sample_goals ):

    # assume 9 values
    results = model( sample_goals )
    results = simple_activation_to_one_hot( results )
    display = to_display_image( results, TERRAIN_ONE_HOT_COLOR )

    # for x in range(9):
    #     tf.print("INDEX="+str(x))
    #     tf.print( results[x], summarize=-1 )

    text = to_display_text( sample_goals, results )
    display_text_and_image(text, display)


def run(model,
        train_data, train_result,
        test_data, test_result,
        ckpt_folder):

    model.summary()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = 'mean_squared_error',
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
# run from command line, but not import

if __name__ == '__main__':

    prepare_globals()

    model = create_model_v1(INPUT_SHAPE)
    train_data, train_result, test_data, test_result = prepare_data()

    run( model, train_data, train_result, test_data, test_result, ckpt_folder )


    work = np.empty( (9,TERRAIN_TYPE_COUNT) )
    for index in range(9):
        ratio = index / 8.
        work[index][0] = ratio - 0.0
        work[index][1] = 1. - ratio

    sample_goals = tf.constant( work )

    display_results(sample_goals)

