#
#   Larger Goal is to create a map suitable for use in Strategic Conquest with
#   six terrains:  sea, land, rough, neutral city, white city, black city
#
#
#   The goal of this design is to produce one-hot map of land + sea
#   goodness is measure via analysis of continuity and surface area ( fractal edge )
#
#   The input data is going to be a mix of terrain maps, classical art, and random noise
#
#   see:
#       https://www.kaggle.com/datasets/tpapp157/terrainimagepairs
#       https://www.kaggle.com/datasets/tpapp157/earth-terrain-height-and-segmentation-map-images
#       https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time
#
#   Analysis: it looks like rules for 'terrain types ratios' and 'tile certainty' have an impact,
#             but 'surface ratios' ( many islands vs few islands ) does not work at all.
#             When I focus entirely on surface ratios, the results have high certainty ( all 0s and 9s ),
#             tend to have an even balance of types, but remain close to maximum surface ratios.
#             I speculate that since surface ratios are a characteristic of the map as a whole,
#             individual tile training is bouncing back and forth over the same solutions.
#             Also, for some reason, the training is complete in the first epoch.  That may be a clue.
#
#             AFTER a little more testing, and a new ReLU only model, I think the answer is simply
#               that my surface_loss approximation can only give me a loose match.   I think if I
#               use softmax or softargmax I can get a more solid loss value.
#

import sys
sys.path.append('..')

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
from tensorflow import image, keras

from land_and_sea_functions import *
from cycle_gan.tf_layer_tools import *

# tf.compat.v1.enable_eager_execution()

print(tf.__version__)

CHECKPOINT_FILE = 'landnsea_ckpt/checkpoint'
BATCH_SIZE = 1
plt.ion()

# # check the dataset parameters
# print("train[s]=", str(train_images.shape) )
# print("len=", str(len(train_labels)) )
# print("train labels = ", str(train_labels.shape) )
# print("test labels = ", str(test_labels.shape) )
# print("image shape = ", str(test_images.shape) )
# print("len(labels) = ", len(test_labels) )

IMAGE_CHANNELS = 3
WIDE = TALL = 16
INPUT_SHAPE = (WIDE, TALL, IMAGE_CHANNELS)
MAP_SHAPE =  (WIDE*2, TALL*2, TERRAIN_TYPE_COUNT)
IMAGE_RESIZE = list(INPUT_SHAPE[:2])     #  [wide,tall]
EPOCHS = 10
SKIP = 1       # only process one out of SKIP from available images :: 1 = keep all

######################################################################

def display_image_in_grid( image ):
    r"""Display contents of a grid of float arrays."""
    plt.grid( True )
    wide,tall,chan = np.array(image).shape
    plt.axis([-0.5,wide-0.5,-0.5,tall-0.5])
    # plt.rcParams.update({'font.size': 8})

    # for (j,i),label[] in np.ndenumerate(grid):
    #     plt.text(i+0.5,j+0.5,"{:.2f}".format(label),ha='center',va='center')

    # for (j,i) in np.ndindex(*image.shape[:2]):
    #     label = image[j][i]
    #     plt.text(i+0.5,j+0.5,"{:.2f}\n{:.2f}\n{:.2f}".format(label[0],label[1],label[1]),ha='center',va='center')

    plt.imshow( image )
    plt.show()
    plt.pause(5)


def random_image(wide,tall,channels):
    r"""Create a grid of floats in range [0-1)."""
    grid = np.empty((wide,tall,channels))
    for col in grid:
        for row in range(tall):
            for chan in range(channels):
                col[row][chan] = rnd.random()
    return grid


def display_grid_as_onehot( grid ):
    r"""Display contents of a grid of one_hot values as integers."""
    plt.grid( True )

    # Compute the argmax across the rows+columns.
    image = tf.argmax(grid, axis=-1)
    # print("onehot_decoded=",image)

    wide,tall = np.array(image).shape
    # print("SHAPE=",np.array(image).shape)
    plt.axis([0,wide,0,tall])
    plt.rcParams.update({'font.size': 8})

    # for (j,i),label[] in np.ndenumerate(grid):
    #     plt.text(i+0.5,j+0.5,"{:.2f}".format(label),ha='center',va='center')

    for (j,i),label in np.ndenumerate(image):
        # print("LOOP=",j,i,label)
        plt.text(i+0.5,j+0.5,"{}".format(label),ha='center',va='center')

    plt.show()
    plt.pause(5)


# image = random_image( 16, 16, 3 )
# print("image=",image)
#
# # display_image_in_grid( image )
#
# display_grid_as_onehot( image )

#######################################################################
#   multiple datasets comprise the input
#

first_path = os.path.expanduser( 'd:/Datasets/KaggleTerrainMapTriples/triples' )
second_path = os.path.expanduser( 'D:/Datasets/KaggleFineArt/images/Leonardo_da_Vinci' )


# load images from triples


def load_features( dataset_path, pattern, skip=1 ):

    path = Path( dataset_path )
    size = len([f for f in path.rglob(pattern)])
    print("FROM ",dataset_path," loading count=", size )
    keep_size = (size+skip-1) // skip # double slash is integer division
    # folders = [f for f in os.listdir(path)]

    # pre-allocate space for all images + labels
    features = np.empty((keep_size,) + INPUT_SHAPE)       # ( count, wide, tall, color_channels)
    # labels = np.empty( (keep_size) )

    idx = 0
    for img_path in path.rglob(pattern):
        if (idx%SKIP)==0:
            index = idx // skip
            img = keras.utils.load_img( img_path )
            features[index] = image.resize( img, IMAGE_RESIZE ) / 255.0
            # labels[index] = folders.index( img_path.parent.name )
        idx += 1

    return features

# terrain_features = load_features( first_path, '*_t.png', SKIP )
# print("TerrainFeaturesShape=",tf.shape(terrain_features))

#######################################################################
#   Random Image Dataset

def load_random_features( count ):
    features = np.empty((count,) + INPUT_SHAPE)
    for index in range(count):
        features[index] = random_image(*INPUT_SHAPE)
    return features

random_features = load_random_features( 1000 )
print("RandomFeaturesShape=",tf.shape(random_features))

#######################################################################

features = random_features  # + terrain_features

tf.random.shuffle( features, 12345 )
tflen = len(features)
train_segment = (int)(tflen * .8)
print("train_segment=",train_segment)
# train_images = list(features)[ 0 : train_segment ]
# test_images = list(features)[ train_segment : tflen-train_segment ]
train_images = features[ 0 : train_segment ]
test_images = features[ train_segment : tflen ]

print("Original len=",tflen)
print("TrainImage len=",len(train_images))
print("TestImage len=",len(test_images))

# inspect first image in dataset

lastFigure = None       # record the last displayed figure so it can be closed automatically

########################################################################################################################
# create model

def trim_layer(filters,size,strides=1):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.ReLU())
    return result

def grow_layer(filters,size,strides=1):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def trim_layer_selu(filters,size,strides=1):
    return tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',activation='selu')

def grow_layer_selu(filters,size,strides=1):
    return tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',activation='selu')

def create_model_v4( shape ):
    r"""two outputs, one validates image reconstruction"""

    x = inputs = keras.Input(shape=shape)
    # tf.print('x0=',x.shape)
    x = trim_layer_selu(64,2)(x)                 # ( bs, 16,16, 64 )
    x = trim_layer_selu(128,2)(x)                         # ( bs, 16,16, 64 )

    x = grow_layer_selu(128,2,2)(x)                         # ( bs, 32,32, 64 )
    x = grow_layer_selu(128,2,1)(x)                          # ( bs, 32,32, 128 )

    output1 = tf.keras.layers.Conv2DTranspose( TERRAIN_TYPE_COUNT, 1, activation='softmax')(x)  # (bs, 32,32, 2) # softmax for map
    tf.print('output1=',output1.shape)
    output2 = tf.keras.layers.Conv2D( IMAGE_CHANNELS, 2,2, activation='tanh')(x)                # (bs, 16,16, 3) # tanh for image
    tf.print('output2=',output2.shape)

    model = tf.keras.Model(inputs=inputs, outputs=[output1,output2] )
    model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
                   loss=[ terrain_loss, tf.keras.losses.MeanSquaredError() ],
                   metrics=['accuracy'] )
    return model

model = create_model_v4(INPUT_SHAPE)

model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
# model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
#                loss=terrain_loss,
#                metrics=['accuracy'] )

########################################################################################################################
# train model

# tflt.try_load_weights( CHECKPOINT_FILE, model )


# print( "TRAIN SHAPE=",tf.shape(train_images) )
# print( "Variables.Module=",tf.Module.trainable_variables )
# print( "Variables.model=",model.trainable_variables )

doubler = ImageResize(2,2)

bigger_train_images = doubler( train_images )
train_image_set = [bigger_train_images,train_images]
tf.print('bigger_train_images=',bigger_train_images.shape)

bigger_test_images = doubler( test_images )
test_image_set = [bigger_test_images,test_images]
tf.print('bigger_test_images=',bigger_test_images.shape)

model.fit( x=train_images, y=train_image_set,
           epochs=EPOCHS,
           validation_data=(test_images,test_image_set),
           batch_size=BATCH_SIZE )

# following does NOT work
# model.fit( x= (train_images,train_images),
#            epochs=EPOCHS,
#            validation_data=(test_images,test_images),
#            batch_size=8 )
# tf.print('test image shape=',tf.shape(test_image_set))
# test_loss, test_acc = model.evaluate( x=test_images, y=test_image_set,  verbose=2 )
result = model.evaluate( x=test_images, y=test_image_set,  verbose=2 )

tf.print('result=',result)
# print('\ntest_loss:', test_loss)
# print('\ntest_acc:', test_acc)

########################################################################################################################


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

model.save_weights( CHECKPOINT_FILE )


########################################################################################################################
#   Display Stuff

def to_map_image( image_values, one_hot_color ):

    onehot = image_values
    # tf.print('work/squeeze=',tf.shape(work))

    onehot = tf.round( onehot )
    # tf.print('work/round=',work)
    onehot = tf.cast(onehot, tf.int32)
    # tf.print('work/cast=',work)

    return tf.tensordot( onehot, one_hot_color, 1 )


def draw_image_and_values(work):

    values = tf.squeeze(work)

    # two colors
    one_hot_color = tf.constant( ((0,255,0),(0,0,255)) )
    display = to_map_image( values, one_hot_color )

    global lastFigure
    plt.close( lastFigure )
    lastFigure = plt.figure()
    plt.imshow( display )

    # tf.print("values=",values.shape)
    if (WIDE<=32):
        for (j,i,t),value in np.ndenumerate(values):
            # tf.print('shape=',value.shape)
            if (t==0):
                label = int( 10 * value )
                plt.text(i,j,"{}".format(label),ha='center',va='center')

    plt.show()
    plt.pause( 500 )

def as_numpy_array( some_tensor ):
    if (tf.executing_eagerly()): return some_tensor.numpy()
    return some_tensor.eval()

def display_absolute_terrain_counts( work ):
    # print("Work=",work)

    print('type_loss=', terrain_type_loss( work ) )
    print('certainty_loss=', terrain_certainty_loss( work ) )
    print('surface_loss=', terrain_surface_loss( work ) )

    # transform into final map
    work = as_numpy_array( tf.argmax( tf.round( work ), axis=-1 ) )
    print("final work=",work)

    # count terrain types
    ttype = [0] * TERRAIN_TYPE_COUNT
    for i in range(0, WIDE):
        for j in range(0, TALL):
            index = work[0][i][j]
            ttype[ index ] = ttype[ index ] + 1

    print('Terrain Counts=',ttype,'sum=',np.sum(ttype) )

    # count transitions
    wide1 = WIDE-1
    tall1 = TALL-1
    ttrans = [0] * TERRAIN_TYPE_COUNT
    for i in range(0, WIDE):
        for j in range(0, TALL):
            index = work[0][i][j]

            count = 0
            if index != work[0][(i+wide1)%WIDE][j]: count = 1+count
            if index != work[0][i][(j+tall1)%TALL]: count = 1+count
            if index != work[0][(i+1)%WIDE][j]: count = 1+count
            if index != work[0][i][(j+1)%TALL]: count = 1+count

            ttrans[ index ] = ttrans[ index ] + count

    print('Terrain Transitions=',ttrans,'sum=',np.sum(ttrans) )


# pick and display one solution
image = test_images[0:1]
result = model( image )
[work,pic] = result

# tf.print('work.shape=',work.shape)
# tf.print('pic.shape=',pic.shape)
# if True: exit(0)

plt.rcParams['text.color'] = 'white'
plt.rcParams['font.size'] = '8'

draw_image_and_values( work )
display_absolute_terrain_counts( work )


