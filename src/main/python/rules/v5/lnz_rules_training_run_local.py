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
import _utilities.tf_loss_tools as lsto
from tf_inspiration_layer import InspirationLayer


# tf.compat.v1.enable_eager_execution()

print(tf.__version__)

plt.ion()

def prepare_globals():

    global EPOCHS, BATCH_SIZE, SAMPLE_COUNT, ADDITIONAL_FEATURES
    # EPOCHS = 5 # 50
    # BATCH_SIZE= 2 #1000
    # SAMPLE_COUNT = 8    # 1000
    ADDITIONAL_FEATURES = 2    # sea_edge, sea_sticky
    EPOCHS = 50
    SAMPLE_COUNT = 10000
    BATCH_SIZE= 1000

    global TERRAIN_TYPE_COUNT, TERRAIN_ONE_HOT_COLOR, TT_SEA,TT_SAND,TT_GRASS,TT_HILLS,TT_RIVER,TT_ROAD
    TERRAIN_TYPE_COUNT = 6
    TT_SEA = 0
    TT_SAND = 1
    TT_GRASS = 2
    TT_HILLS = 3
    TT_RIVER = 4
    TT_ROAD = 5
    TERRAIN_ONE_HOT_COLOR = tf.constant( ( (
        (0,0,255), (255,255,0), (0,255,0),
        (128,128,128) ,(96,96,255) ,(192,192,192)
    ) ) )

    global WIDE, TALL, EDGES, EDGE_COUNT, EDGE_RATIO, GRID_UNITS
    WIDE = 20
    TALL = 20
    GRID_UNITS = WIDE * TALL
    EDGES = make_edges( WIDE, TALL )
    EDGE_COUNT = ( 2*WIDE + 2*TALL - 4.)
    EDGE_RATIO = EDGE_COUNT * 1. / GRID_UNITS

    global INPUT_SHAPE, IMAGE_SHAPE, FEATURE_COUNT
    INPUT_SHAPE = ( TERRAIN_TYPE_COUNT )
    IMAGE_SHAPE = ( WIDE, TALL )
    FEATURE_COUNT = TERRAIN_TYPE_COUNT + ADDITIONAL_FEATURES

    global OUTPUT_SHAPE, OUTPUT_UNITS
    OUTPUT_SHAPE = ( WIDE, TALL, TERRAIN_TYPE_COUNT )
    OUTPUT_UNITS = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]

    global flavor, ckpt_folder, lastFigure
    flavor = "feature2random"
    ckpt_folder = 'landnsea_ckpt/v2/ratio_random'
    lastFigure = None       # record the last displayed figure so it can be closed automatically


def make_edges( wide, tall ):
    r"""Create a grid of given size, alls zeros with ones along the edge."""

    wide_row = np.zeros( ( wide, ) )
    wide_row[0] = wide_row[-1] = 1.
    tall_row = np.zeros( ( tall, ) )
    tall_row[0] = tall_row[-1] = 1.

    wide_grid = np.tile( wide_row, [tall,1] )
    tall_grid = np.transpose( np.tile( tall_row, [wide,1] ) )

    return np.clip( wide_grid + tall_grid, 0., 1. )

#######################################################################

def append_inverse( features ):
    r"""Append in 'inverted' set of values to the one dimensional input tensor."""
    inverse = 1. - np.array( features )
    return tf.concat( [features, inverse ], axis=-1 )


def input_transform( features ):
    r"""Transform for feature set which is applied to training, testing and examples."""
    return features
    #return append_inverse( features )


def feature_to_sample( features, shape ):
    r"""Transform feature set into shaped sample for training and testing.
    Feature is one dimensional tensor in the last dimension of 'features'.
    Shape must be big enough to contain all feature values.
    After the feature values are in the sample, the rest is filled with zero."""

    size = tf.reduce_prod( shape )
    # tf.print("size=",size)

    feature_shape = tf.shape( features )
    feature_count = feature_shape[-1]
    # print("feature_shape=",feature_shape)
    base_shape = feature_shape[:-1]
    # print("base_shape=",base_shape)

    feature_fill = size - feature_count
    # tf.print("feature_fill=",feature_fill)
    # fill_add = tf.constant([feature_fill,])
    fill_add = [feature_fill,]
    # tf.print("fill_add=",fill_add)
    fill_shape = tf.concat( [ base_shape, fill_add ], axis=0 )
    # tf.print("fill_shape=",fill_shape)
    fill_zeros = tf.zeros( fill_shape )

    feature_and_zeros = tf.concat( [features,fill_zeros], axis=-1)
    # tf.print("feature_and_zeros=",feature_and_zeros)
    full_shape = tf.concat( [ base_shape, shape], axis=0  )
    # tf.print("full_shape=",full_shape)

    return tf.reshape( feature_and_zeros, full_shape )


def sample_to_feature( samples, feature_size, axis ):
    r"""Transform shaped sample into a feature set for calculating loss.
    Feature is one dimensional tensor or a sequence of one dimensional tensors.
    Shape is usually 2-3 dimensional, but could be any size provided it has enough space to contain feature.
    After the feature values in the sample, the rest is filled with zero."""

    shape = tf.shape( samples )
    base_shape = shape[:axis]
    # tf.print("base_shape=",base_shape)
    sample_shape = shape[axis:]
    # tf.print("sample_shape=",sample_shape)
    sample_size = tf.reduce_prod( sample_shape )
    # tf.print("sample_size=",sample_size)

    flat_shape = tf.concat( [base_shape,[sample_size]],axis=0)
    # tf.print("flat_shape=",flat_shape)
    flat_samples = tf.reshape( samples, flat_shape )
    # tf.print("flat_samples=",flat_samples)

    # clip:  (z0,z1, ... zN),  (s0,s1, ... sN-1)
    zero_slice = tf.zeros( 1+axis, dtype=tf.int32 )
    # tf.print("zero_clip=",str(zero_slice))
    feature_shape = tf.constant([feature_size,])
    shape_slice = tf.concat ( [flat_shape[:-1],feature_shape], axis=0 )
    # tf.print("shape_clip=",str(shape_slice))

    # tf.print("SLICE=",flat_samples.slice(zero_slice,shape_slice))
    result = tf.slice( flat_samples, begin=zero_slice, size=shape_slice )
    # tf.print("result=",result)
    return result

#######################################################################
#   Training Dataset :: random sampling of valid combinations
#######################################################################

def build_examples( count, depth ):
    r"""

    :param count: number of samples
    :param depth: number of features in each sample
    """
    generator = tf.random.Generator.from_non_deterministic_state()

    terrain_shape = ( count, TERRAIN_TYPE_COUNT )
    feature_shape = ( count, depth-TERRAIN_TYPE_COUNT )
    tf.print("terrain_shape=",terrain_shape)
    tf.print("feature_shape=",feature_shape)

    # uniform for all features, including terrain ratios
    terrains = generator.uniform( shape=terrain_shape, minval=0., maxval=1. )
    features = generator.uniform( shape=feature_shape, minval=0., maxval=1. )

    # the first TERRAIN_COUNT features need to be normalized to a sum of one
    terrains = tf.linalg.normalize( terrains, axis=1)[0]
    tf.print( "terrains.shape=", tf.shape(terrains) )

    # join and return
    return tf.concat( [terrains,features], axis=1 )

#######################################################################

def feature_set_to_sample_set(feature_set):
    r"""Result set has to be same shape as output, even though we only use one tuple set at [0][0]."""
    return feature_to_sample( feature_set, OUTPUT_SHAPE )
    # count = len(feature_set)    # tf.shape( feature_set )[0]
    # shape = ( count, ) + OUTPUT_SHAPE
    # tf.print("Output SHAPE=",shape)
    # results = np.empty( shape )
    # for index in range(count):
    #     results[index][0][0] = feature_set[index]
    # return results

#######################################################################

def prepare_data():

    feature_set_linear = build_examples( SAMPLE_COUNT, FEATURE_COUNT )
    print("SAMPLE_COUNT=",SAMPLE_COUNT)
    result_set_linear = feature_set_to_sample_set(feature_set_linear)
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
def terrain_ratio_loss( y_goal_ratios, y_guess ):

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

    veed = lsto.valley(type_distance)
    # tf.print("veed=",veed)
    result = tf.reduce_mean( veed, axis=-1 )
    # tf.print("result=",result)
    return result


@tf.function
def terrain_hard_ratio_loss( y_goal_ratios, y_guess ):

    y_hard_guess = teto.supersoftmax( y_guess, beta=2. )
    return terrain_ratio_loss( y_goal_ratios, y_hard_guess )


@tf.function
def terrain_type_near_edge_loss( ttype_index, y_guess ):
    r"""determine certainty loss :: closer to 0 or 1 than 0.5 is better ---"""
    # tf.print("input=",y_pred)
    # tf.print("input.shape=",y_pred.shape)

    edge = tf.constant( EDGES, dtype=tf.float32 )
    # tf.print("EDGE=",edge)
    ttype_logit = tf.one_hot( ttype_index, TERRAIN_TYPE_COUNT )
    # tf.print("LOGIT=",ttype_logit)

    outer = tf.tensordot( edge, ttype_logit, axes=0 )
    # tf.print("OUTER=",outer)
    edge_type_logits = y_guess * outer
    # tf.print("WORK=",edge_type_logits)
    edge_type_sum = tf.reduce_sum( edge_type_logits, [-3,-2,-1] )
    # tf.print("edge_type_sum=",str(edge_type_sum))

    all_type_logits = y_guess * ttype_logit
    # tf.print("all_type_logits=",all_type_logits)
    all_type_sum = tf.reduce_sum( all_type_logits, [-3,-2,-1] )
    # tf.print("all_type_sum=",str(all_type_sum) )

    # TODO: 'ideal' comes from FEATURES[-2]
    ideal = 1.3 * EDGE_RATIO
    ratio = edge_type_sum / tf.clip_by_value( all_type_sum, 1., EDGE_COUNT )
    # tf.print("ratio=",ratio)
    ratio = tf.where( ratio>ideal, ideal, ratio )

    # if ratio goal is too high ( eg 1.0 ), then AI fills the edge ONLY, ignoring goal ratio
    # even if ratio goal is low ( eg. 0.1 ), then AI adds terrain even when goal ratio is ZERO
    # so this needs to see the goal ratios to decide on a correct loss.

    # return lsto.valley( ideal - ratio )
    return ideal - ratio


@tf.function
def terrain_type_sticky_loss( ttype_index, y_guess ):
    r"""We want terrain of a given type to 'stick' to itself,
    which means a minimum ratio of adjacent tiles with the same terrain.

    Use 'min(a,b)' for similarity value, so only (1,1) is maximum.

    Sum to count.
    """

    ttype_logit = tf.one_hot( ttype_index, TERRAIN_TYPE_COUNT )
    # tf.print("LOGIT=",ttype_logit)

    all_type_logits = y_guess * ttype_logit
    all_type_count = tf.reduce_sum( all_type_logits, [-1] )
    # tf.print("all_type_count=",all_type_count)
    # all_type_sum = tf.reduce_sum( all_type_logits, [-3,-2,-1] )
    all_type_sum = tf.reduce_sum( all_type_count, [-2,-1] )
    # tf.print("all_type_sum=",all_type_sum)

    rollh = tf.roll( all_type_count, 1, -1  )
    # tf.print('rollh=',rollh)
    # diffh = tf.keras.losses.cosine_similarity(y_guess, rollh, axis=-1 )
    diffh = tf.math.minimum( all_type_count, rollh )
    diffh = tf.reduce_sum( diffh, [-2,-1] )
    # tf.print('diffh=',diffh)

    rollv = tf.roll( all_type_count, 1, -2  )
    # tf.print('rollv=',rollv)
    # diffv = tf.keras.losses.cosine_similarity(y_guess, rollv, axis=-1 )
    diffv = tf.math.minimum( all_type_count, rollv )
    diffv = tf.reduce_sum( diffv, [-2,-1] )
    # tf.print('diffv=',diffv)

    diff_avg = ( diffh + diffv ) / 2.
    # tf.print('diff_avg=',diff_avg)

    # TODO: 'ideal' comes from FEATURES[-1]
    ideal = 0.3
    ratio = diff_avg / tf.clip_by_value( all_type_sum, 1., GRID_UNITS )
    # tf.print("ratio=",ratio)
    ratio = tf.where( ratio>ideal, ideal, ratio )
    # tf.print("ratio=",ratio)

    # Below 0.25 this feature does not have a great impact,
    # at 0.50 fill the edges more, increase of sea tiles, still many unconnected sea dots.
    # at 0.70 the corners are always sea, even when sea ratio is zero
    # at 1.00 this feature made ALL terrains sticky, and somewhat ignored ratio on SEA ( eg. no sea when ratio was low )
    #


    # return lsto.valley( ideal - ratio )
    return ideal - ratio


@tf.function
def get_ratio_goals( y_goal ):
    r"""ratio goals are in the logit at [0][0] in the goal grid."""
    return sample_to_feature( y_goal, feature_size=FEATURE_COUNT, axis=1 )
    # ratio_goals = tf.slice( y_goal, [0,0,0,0], [-1,1,1,TERRAIN_TYPE_COUNT] )
    # ratio_goals = tf.squeeze( ratio_goals, axis=2 )
    # ratio_goals = tf.squeeze( ratio_goals, axis=1 )
    # return ratio_goals


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
    sea_edge_loss = terrain_type_near_edge_loss( TT_SEA, y_guess )
    sea_sticky_loss = terrain_type_sticky_loss( TT_SEA, y_guess )

    terrain_loss = ratio_loss + hard_loss + sea_edge_loss + sea_sticky_loss

    with tf.GradientTape() as t:
        t.watch( terrain_loss )
        t.watch( ratio_loss )
        t.watch( hard_loss )
        t.watch( sea_edge_loss )
        t.watch( sea_sticky_loss )

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


########################################################################################################################

def create_model_v1( shape ):

    # input value, (batch,TERRAIN_TYPE_COUNT)
    x = inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(x)

    # decode features into detailed grid
    x = tf.keras.layers.Dense( 4 * TERRAIN_TYPE_COUNT, activation='selu', name='decode1')(x)
    x = tf.keras.layers.Dense( 16 * TERRAIN_TYPE_COUNT, activation='selu', name='decode2')(x)
    # tf.print("x.shape=",x.shape)

    # build array of fixed values ( to be replaced with random )
    y = InspirationLayer( GRID_UNITS, name='Inspire' )(inputs)
    # tf.print("y.shape=",y.shape)

    # join feature encoding to random image
    x = tf.keras.layers.Concatenate( name='FeatureAndInspire')( [x, y] )

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
    work[0] = [ 1., 0., 0., 0., 0., 0. ]
    work[1] = [ 1., 1., 0., 0., 0., 0. ]
    work[2] = [ 1., 1., 1., 0., 0., 0. ]
    work[3] = [ 1., 1., 1., 1., 0., 0. ]
    work[4] = [ 1., 1., 1., 1., 1., 0. ]
    work[5] = [ 1., 1., 1., 1., 1., 1. ]
    work[6] = [ 1., 0., 0., 2., 0., 0. ]
    work[7] = [ 1., 0., 0., 8., 0., 0. ]
    work[8] = [ 0., 0., 0., 1., 0., 0. ]
    sample_goals = tf.constant( work )

    # display three times to demonstrate randomness
    display_results(sample_goals)
    display_results(sample_goals)
    display_results(sample_goals)

