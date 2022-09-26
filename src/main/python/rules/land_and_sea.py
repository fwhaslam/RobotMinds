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

import sys
sys.path.append('..')

# common
import os as os
import numpy as np
import random as rnd
from pathlib import Path
# import matplotlib as mpl
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf
from tensorflow import image, keras
from tensorflow.keras import datasets, layers, models

# local helpers
# import _utilities.tf_cyclegan_tools as cgan
# import _utilities.tf_loading_tools as loader
from land_and_sea_functions import *

tf.compat.v1.enable_eager_execution()

print(tf.__version__)

CHECKPOINT_FILE = 'checkpoint/landnsea_ckpt'
plt.ion()

# # check the dataset parameters
# print("train[s]=", str(train_images.shape) )
# print("len=", str(len(train_labels)) )
# print("train labels = ", str(train_labels.shape) )
# print("test labels = ", str(test_labels.shape) )
# print("image shape = ", str(test_images.shape) )
# print("len(labels) = ", len(test_labels) )


IMAGE_SIZE = ( 32, 32, 3 )
IMAGE_RESIZE = list(IMAGE_SIZE[:2])     #  [wide,tall]
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
    features = np.empty( (keep_size,)+IMAGE_SIZE )       # ( count, wide, tall, color_channels)
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
    features = np.empty( (count,) + IMAGE_SIZE )
    for index in range(count):
        features[index] = random_image( *IMAGE_SIZE )
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

def display_first_image():
    global lastFigure
    lastFigure = plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    plt.pause(10)

# display first 25 images
def display_first_25():
    global lastFigure
    plt.close( lastFigure )
    lastFigure = plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # plt.xlabel(class_names[train_labels[i]])
    plt.show()
    plt.pause(10)

# display_first_image()
# display_first_25()

########################################################################################################################
# create model

from cycle_gan.tf_layer_tools import *

def trim_layer(filters,size,strides=1):
    return tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',activation='selu')
    # result = tf.keras.Sequential()
    # result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
    # result.add(InstanceNormalization())
    # result.add(tf.keras.layers.LeakyReLU())
    # return result

def grow_layer(filters,size,strides=1):
    return tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',activation='selu')
    # result = tf.keras.Sequential()
    # result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same'))
    # result.add(InstanceNormalization())
    # result.add(tf.keras.layers.LeakyReLU())
    # return result

def create_model_v2( shape ):

    x = inputs = keras.Input(shape=shape)
    # tf.print('x0=',x.shape)
    x = trim_layer(64,2,2)(x)       # ( bs, 16,16, 64 )
    # tf.print('x1=',x.shape)
    x = trim_layer(256,2,2)(x)       # ( bs, 8,8, 256 )
    x = trim_layer(1024,2,2)(x)       # ( bs, 4,4, 1024 )
    # tf.print('x2=',x.shape)

    x = grow_layer(1024,2,2)(x)       # ( bs, 8,8, 1024 )
    x = grow_layer(256,2,2)(x)       # ( bs, 16,16, 256 )
    x = grow_layer(64,2,2)(x)       # ( bs, 32,32, 64 )

    # x = tf.keras.layers.Conv2DTranspose( TERRAIN_TYPE_COUNT, 1, activation='tanh')(x)   # (bs, 32,32, 2) # tanh for image
    x = tf.keras.layers.Conv2DTranspose( TERRAIN_TYPE_COUNT, 1, activation='softmax')(x)   # (bs, 32,32, 2) # softmax for map
    outputs = x

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_model_v1( shape ):

    x = inputs = keras.Input(shape=shape)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(2048,activation='LeakyReLU')(x)
    x = keras.layers.Dense(2048,activation='LeakyReLU')(x)
    # x = keras.layers.Dense(2048,activation='LeakyReLU')(x)
    x = keras.layers.Reshape( (32,32,2) )(x)
    x = keras.layers.Activation( 'softmax' )(x)
    outputs = x

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = create_model_v2( IMAGE_SIZE )

def get_terrain_loss():
    return terrain_loss

model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
model.compile( optimizer=tf.keras.optimizers.Adam(0.001),
               loss=terrain_loss,
               metrics=['accuracy'] )

########################################################################################################################
# train model

# tflt.try_load_weights( CHECKPOINT_FILE, model )


# print( "TRAIN SHAPE=",tf.shape(train_images) )
# print( "Variables.Module=",tf.Module.trainable_variables )
# print( "Variables.model=",model.trainable_variables )

model.fit( x=train_images, y=train_images,
           epochs=EPOCHS,
           validation_data=(test_images,test_images),
           batch_size=8 )

# following does NOT work
# model.fit( x= (train_images,train_images),
#            epochs=EPOCHS,
#            validation_data=(test_images,test_images),
#            batch_size=8 )
tf.print('test image shape=',tf.shape(test_images))
test_loss, test_acc = model.evaluate( test_images,  verbose=2 )

print('\nTest accuracy:', test_acc)

########################################################################################################################
# make predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# print("pred="+ str(predictions[0]) )

np.argmax(predictions[0])

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

    work = image_values
    # tf.print('work/squeeze=',tf.shape(work))

    work = tf.round( work )
    # tf.print('work/round=',work)
    work = tf.cast(work, tf.int32)
    # tf.print('work/cast=',work)

    one_hot_color = tf.constant( ((0,255,0),(0,0,255)) )
    return tf.tensordot( work, one_hot_color, 1 )

    # work = tf.argmax( work, axis=-1 )
    # tf.print('work/argmax=',work)
    # work = tf.squeeze(work)
    # tf.print('work/sqeeze=',work)
    # def pick_to_color( pick ):
    #     tf.print('PICK=',pick)
    #     if (pick): return (0,0,255)
    #     return (0,255,0)
    # return tf.map_fn( pick_to_color, work, fn_output_signature=tf.TensorSpec( (3,), tf.int32 ) )

def draw_image_display_values():
    image = test_images[0:1]
    # print("image=",image)

    # two colors
    one_hot_color = tf.constant( ((0,255,0),(0,0,255)) )
    work = model( image )
    values = tf.squeeze(work)
    display = to_map_image( values, one_hot_color )

    print('type_loss=', terrain_type_loss( work ) )
    print('certainty_loss=', terrain_certainty_loss( work ) )
    print('surface_loss=', terrain_surface_loss( work ) )

    global lastFigure
    plt.close( lastFigure )
    lastFigure = plt.figure()
    plt.imshow( display )

    # tf.print("values=",values.shape)
    for (j,i,t),value in np.ndenumerate(values):
        # tf.print('shape=',value.shape)
        if (t==0):
            label = int( 10 * value )
            plt.text(i,j,"{}".format(label),ha='center',va='center')

    plt.show()
    plt.pause( 500 )


plt.rcParams['text.color'] = 'white'
plt.rcParams['font.size'] = '8'
draw_image_display_values()

# # verify predictions
# plt.close( lastFigure )
# i = 0
# lastFigure = plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
# plt.pause( 10 )
#
# plt.close( lastFigure );
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
# plt.pause( 10 )
#
# # Plot the first X test images, their predicted labels, and the true labels.
# # Color correct predictions in blue and incorrect predictions in red.
# plt.close( lastFigure )
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()
# plt.pause( 30 )
