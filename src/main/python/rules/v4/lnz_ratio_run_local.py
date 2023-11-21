#
#   This variation uses random images as a starting point.
#
#   built from python/rules/v1/lnz_pics_via_runner.py
#

import sys
sys.path.append('../..')

# common
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

def prepare_globals():

    global EPOCHS, BATCH_SIZE, SAMPLE_COUNT
    # EPOCHS = 5 # 50
    # BATCH_SIZE= 2 #1000
    # SAMPLE_COUNT = 8    # 1000
    EPOCHS = 50
    SAMPLE_COUNT = None     # calculated when producing feature_set
    BATCH_SIZE= 1000

    global TERRAIN_TYPE_COUNT, TERRAIN_ONE_HOT_COLOR, TT_SEA,TT_SAND,TT_GRASS,TT_HILLS
    TERRAIN_TYPE_COUNT = 4
    TT_SEA = 0
    TT_SAND = 1
    TT_GRASS = 2
    TT_HILLS = 3
    TERRAIN_ONE_HOT_COLOR = tf.constant( ( ( (0,0,255)),(255,255,0),(0,255,0),(128,128,128) ) )

    global WIDE, TALL
    WIDE = 10
    TALL = 10

    global INPUT_SHAPE, IMAGE_SHAPE, IMAGE_UNITS
    INPUT_SHAPE = ( TERRAIN_TYPE_COUNT )
    IMAGE_SHAPE = ( WIDE, TALL )
    IMAGE_UNITS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

    global OUTPUT_SHAPE, OUTPUT_UNITS
    OUTPUT_SHAPE = ( WIDE, TALL, TERRAIN_TYPE_COUNT )
    OUTPUT_UNITS = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]

    global flavor, ckpt_folder, lastFigure
    flavor = "feature2random"
    ckpt_folder = 'landnsea_ckpt/v2/ratio_random'
    lastFigure = None       # record the last displayed figure so it can be closed automatically


#######################################################################

def append_inverse( features ):
    r"""Append in 'inverted' set of values to the one dimensional input tensor."""
    inverse = 1. - np.array( features )
    return tf.concat( [features, inverse ], axis=-1 )


def input_transform( features ):
    r"""Transform to feature set applied to training, testing and example feature sets."""
    return features
    #return append_inverse( features )

#######################################################################
#   Training Dataset

def terrain_type_feature_count( depth:int, ratio_steps:int ):
    r"""
    Given a number of steps for the ratio, and a depth ( = terrain_count )
    :param depth: terrain type count
    :param ratio_steps: how many steps in the ratio ( eg. value of 0/10 -> 10/10 is 11 steps )
    :return:
    """
    if (depth==1):
        return ratio_steps
    else:
        return terrain_type_feature_count( depth-1, ratio_steps+1 ) * ratio_steps / depth


def terrain_type_ratios( fill, depth:int, ratio_steps:int, steps_held:int=0, index:int=0, tuple=() ):
    r"""

    :param fill: the array that needs filling ( batch, type_count )
    :param depth: the type_count
    :param ratio_steps: how many steps in the ratio ( eg. value of 0/10 -> 10/10 is 11 steps )
    :return:
    """

    for steps in range( (int) ( ratio_steps - steps_held ) ):
        feature = steps * 1. / ( ratio_steps - 1. )
        work = tuple + (feature,)
        if (depth==1):
            fill[ index ] = work
            index += 1
        else:
            index = terrain_type_ratios( fill, depth-1, ratio_steps, steps_held+steps, index, work )

    return (int)(index)


def load_features():
    r"""
    type_count = 4  :: so use 4th dim triangle
    batch = n * (n+1) * (n+2) * (n+3) / ( 4! ) :: fourth dimensional triangle
    where n = 11 ( because range is [0-10], which is 11 values )
    shape = [batch,Types] ( initially N=2 )
    :return:
    """
    batch = terrain_type_feature_count( depth=TERRAIN_TYPE_COUNT, ratio_steps=11 )
    shape = ( (int)(batch), TERRAIN_TYPE_COUNT )
    tf.print("Feature SHAPE=",shape)
    features = np.empty( shape )
    count = terrain_type_ratios( features, TERRAIN_TYPE_COUNT, 11 )
    assert (batch==count), "filled the exact right amount in the feature set"
    return features


#######################################################################

def feature_to_result( feature_set ):
    r"""Result set has to be same shape as output, even though we only use one tuple set at [0][0]."""
    count = len(feature_set)    # tf.shape( feature_set )[0]
    shape = ( count, ) + OUTPUT_SHAPE
    tf.print("Output SHAPE=",shape)
    results = np.empty( shape )
    for index in range(count):
        results[index][0][0] = feature_set[index]
    return results

#######################################################################

def prepare_data():

    feature_set_linear = load_features()
    SAMPLE_COUNT = len( feature_set_linear )
    print("SAMPLE_COUNT=",SAMPLE_COUNT)
    result_set_linear = feature_to_result( feature_set_linear )
    feature_set_linear = input_transform( feature_set_linear )

    # shuffle with same seed
    tf.random.set_seed( 12345 )
    feature_set = tf.random.shuffle( feature_set_linear  )
    tf.random.set_seed( 12345 )
    result_set = tf.random.shuffle( result_set_linear )

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

    y_hard_guess = teto.supersoftmax( y_guess, beta=2. )
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
def get_ratio_goals( y_goal ):
    ratio_goals = tf.slice( y_goal, [0,0,0,0], [-1,1,1,TERRAIN_TYPE_COUNT] )
    ratio_goals = tf.squeeze( ratio_goals, axis=2 )
    ratio_goals = tf.squeeze( ratio_goals, axis=1 )
    return ratio_goals


@tf.function
def terrain_loss( y_goal, y_guess ):

    r"""y_actual is a 2d tensor of soft logits indicating terrain for each tile.
    In this first draft, 0=sea and 1=land
    Expected shape is: (batch_size, wide,tall, type_count )
    Loss is based on count of tile types, which is approximated by summing soft logits.
    Loss is also based on certainty, which is defined as being close to 0 or 1, not 0.5
    """

    # reduce to first value pair in the image array
    ratio_goals = get_ratio_goals( y_goal )
    # tf.print("RATIO_GOALS=",ratio_goals)

    ratio_loss = terrain_ratio_loss( ratio_goals, y_guess )
    hard_loss = terrain_hard_ratio_loss( ratio_goals, y_guess )
    # certainty_loss = terrain_certainty_loss( y_guess )
    # terrain_loss = terrain_type_loss( sm_y_pred )
    # surface_loss = terrain_surface_loss( sm_y_pred )

    terrain_loss = ratio_loss + hard_loss

    with tf.GradientTape() as t:
        # t.watch( template_mse )
        t.watch( terrain_loss )
        t.watch( ratio_loss )
        t.watch( hard_loss )
        # t.watch( certainty_loss )
        # t.watch( surface_loss )

    # tf.print("TerrainLoss=",terrain_loss)
    return terrain_loss

# @tf.function
# def make_random( shape ):
#     print( "make random shape=", shape )
#     batch = shape[0]
#     if (batch==None):
#         # return tf.placeholder( tf.float32, shape )
#         return tf.keras.Input( shape, dtype=tf.dtypes.float32)
#     return tf.random.normal( shape )


def create_model_v1( shape ):

    image_units = IMAGE_UNITS
    decode_units = 4 * TERRAIN_TYPE_COUNT

    # input value, (batch,TERRAIN_TYPE_COUNT)
    x = inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(x)

    # decode features into detailed grid
    x = tf.keras.layers.Dense( 4 * TERRAIN_TYPE_COUNT, activation='selu', name='decode1')(x)
    x = tf.keras.layers.Dense( 16 * TERRAIN_TYPE_COUNT, activation='selu', name='decode2')(x)
    # tf.print("x.shape=",x.shape)

    # build array of fixed values ( to be replaced with random )
    y = inputs
    y = tf.keras.layers.Dense( IMAGE_UNITS, name='Rando1' )( y )
    y = GaussianLayer( name='Rando2' )( y )
    # tf.print("y.shape=",y.shape)

    # join feature encoding to random image
    x = tf.keras.layers.Concatenate( name='FeatureAndRandom')( [x, y] )

    # two processing layers
    x = tf.keras.layers.Dense( 64 * TERRAIN_TYPE_COUNT, activation='selu', name='eval1')(x)
    x = tf.keras.layers.Dense( 128 * TERRAIN_TYPE_COUNT, activation='selu', name='eval2')(x)

    # output to 'softmax' which means two values that determine one pixel
    x = tf.keras.layers.Dense( OUTPUT_UNITS ,activation='selu', name='reduce')(x)
    x = tf.keras.layers.Reshape( OUTPUT_SHAPE )(x)
    x = tf.keras.layers.Activation( 'softmax' )(x)
    outputs = x

    return tf.keras.Model(inputs=inputs, outputs=outputs,name='basic_model_v1')

########################################################################################################################

def to_display_text( goals, results ):

    goal_ratios = teto.simple_ratio( goals )
    ratios = terrain_ratio_loss( goal_ratios, results )
    hards = terrain_hard_ratio_loss( goal_ratios, results )
    # sures = terrain_certainty_loss( results )

    goals_str = tf.strings.as_string( tf.transpose( goals ), precision=0 )
    # tf.print("goal_str=",goals_str)
    goals_summary = teto.tensor_to_value( tf.strings.join( goals_str, '-' ) )
    # tf.print("goals_summary=",goals_summary)

    texts = [''] * 9
    for x in range(9):
        goal = goals_summary[x].decode()
        ratio = "%.3f" % teto.tensor_to_value( ratios[x] )
        hard = "%.3f" % teto.tensor_to_value( hards[x] )
        # sure = "%.3f" % teto.tensor_to_value( sures[x] )
        texts[x] = "G="+goal+"\nR="+ratio+" - H="+hard
    return texts


def to_display_image( image_values, one_hot_color ):

    onehot = image_values
    # tf.print('work/squeeze=',tf.shape(work))

    onehot = teto.supersoftmax( onehot )
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
    ratios = teto.simple_ratio( sample_goals )
    ratios = input_transform( ratios )
    results = model( ratios )
    display = to_display_image( results, TERRAIN_ONE_HOT_COLOR )

    # for x in range(9):
    #     tf.print("INDEX="+str(x))
    #     tf.print( results[x], summarize=-1 )

    text = to_display_text( sample_goals, results )
    display_text_and_image(text, display)


########################################################################################################################

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
# run from command line, but not from test

if __name__ == '__main__':

    prepare_globals()

    train_data, train_result, test_data, test_result = prepare_data()

    model_shape = tf.shape( train_data )[1:]
    model = create_model_v1( model_shape )

    run( model, train_data, train_result, test_data, test_result, ckpt_folder )

    index = -1
    work = np.empty( (9,TERRAIN_TYPE_COUNT) )
    work[0] = [ 1., 0., 0., 0. ]
    work[1] = [ 1., 1., 0., 0. ]
    work[2] = [ 1., 1., 1., 0. ]
    work[3] = [ 1., 1., 1., 1. ]
    work[4] = [ 1., 1., 1., 2. ]
    work[5] = [ 1., 1., 2., 4. ]
    work[6] = [ 1., 2., 4., 8. ]
    work[7] = [ 1., 0., 0., 2. ]
    work[8] = [ 1., 0., 0., 5. ]
    sample_goals = tf.constant( work )

    display_results(sample_goals)

