#
#   Dataset from:
#       https://www.kaggle.com/datasets/volkandl/cartoon-classification
#
#   Source code from:
#       local source CNN_direct_hotdog_classify.py
#
#   This is a copy of the code with the following changes:
#       Using Custom Loss Functions: my_custom_logits_loss(), my_custom_logits_loss_v2()
#       switch to Kaggle Cartoon dataset
#       change IMAGE_SIZE, used IMAGE_SIZE in place of static values
#       added SKIP to reduce dataset to 1/32 in size ( skipping to every 32nd image )
#       fixed model + loss function to handle multiple categories
#       reduced model size to fit my local memory
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#       Tensorflow 2.9.1 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import os
import numpy as np
import tensorflow as tf
from tensorflow import image, keras
from pathlib import Path

# TODO: modify path to match local user-space
root_path = os.path.expanduser( 'd:/Datasets/KaggleCartoonReduced/cartoon_classification' )

# load feature + labels
IMAGE_SIZE = ( 180, 320, 3 )
IMAGE_RESIZE = list(IMAGE_SIZE[:2])     #  [wide,tall]
EPOCHS = 10
# SKIP = 32       # only process one out of SKIP from available images
SKIP = 1       # only process one out of SKIP from available images

#
# load all JPGs from folder tree, determine the name of top-level folders as 'class' names
#
def load_features_and_labels( type ):

    path = Path( root_path + "/" + type )
    size = len([f for f in path.rglob('*.jpg')])
    keep_size = (size+SKIP-1) // SKIP
    folders = [f for f in os.listdir(path)]

    # pre-allocate space for all images + labels
    features = np.empty( (keep_size,)+IMAGE_SIZE )       # (size, 180, 320, 3)
    labels = np.empty( (keep_size,) )

    idx = 0
    for img_path in path.rglob('*.jpg'):
        if (idx%SKIP)==0:
            index = idx // SKIP
            img = keras.utils.load_img( img_path )
            features[index] = image.resize( img, IMAGE_RESIZE ) / 255.0
            labels[index] = folders.index( img_path.parent.name )
        idx += 1

    return folders, features, labels

#
# load all image/labels based on type
#
train_folders, train_features, train_labels = load_features_and_labels( 'train' )
test_folders, test_features, test_labels = load_features_and_labels( 'test' )
assert np.array_equiv( train_folders, test_folders )

classification_count = len(train_folders)

import matplotlib.pyplot as plt
plt.ion()   # do not wait for windows to close

#
# display 10x10 grid of images for evaluation
#
def display_image_grid( title, features ):
    fig = plt.figure( figsize=(10, 10) )
    fig.suptitle( title, fontsize=16)
    for index in range(100):
        plt.subplot( 10, 10, 1+index )
        plt.axis('off')
        plt.imshow( features[index] )
    plt.show()
    plt.pause( 10 )


#
#   select 100 images evenly through dataset
#
skip_images = len(train_features) // 100
display_image_grid( 'Sample Images', train_features[::skip_images] )

#
#   Model is optimized for memory size over accuracy.
#   CNNs used for rapid dropoff of image size
#   this usually surpasses 95% by epoch 10
#
def create_model():
    model = keras.models.Sequential()
    model.add(keras.Input(shape=IMAGE_SIZE))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu' ))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu' ))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(96, (2, 2), activation='relu' ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(classification_count, activation='softmax'))
    model.add(keras.layers.Dense(classification_count))
    return model

model = create_model()
model.summary()

#
# Custom Loss Function :: without any measure of accuracy or use of GradientTape
#
# Simplify to a single method, batch is the first dimension (=n)
#
# inputs:   indices nx1 = [[a],[b],..[z]] = [batch_size,1] ( ?, for sparse categorical this sould be [batch_size,]
#           outputs = nx10 = [ [a0,b0,...j0], [a1,b1,...j1], ... [an,bn,...jn] ] = [batch_size,10]
#
# returns:  values selected from outputs by index [ c0, a1, .. hn ] = [batch_size,]
#
# NOTE: We use the y_true as an index into the y_pred to get the categorical floating value.
#       Index creates a sparse matrix with one and zeros (one_hot) to isolate a
#       single value from y_pred, and we then sum the single value with all the zeros
#       to get an array of single value results.
#       Subtract from one so that the ideal value appraoches zero.
#
# PERFORMANCE: this sucks, but with a low value for Adam(0.0001), it makes progress.
#
def my_custom_logits_loss( y_true, y_pred ):
    select = tf.squeeze( tf.one_hot( tf.cast(y_true,dtype=tf.int32), classification_count ), 1 )
    results = 1 - tf.reduce_sum( y_pred * select, 1 )
    # tf.print("** y_true=",y_true)
    # tf.print("** y_pred=",y_pred)
    # tf.print( "results=",results )
    return results

#
#   This converts the results to mean square error, but still doew not use GradientTape
#
#   according to docs y_true and the output should be one dimension smaller than y_pred
#
#       y_true is smaller because it becomes logits ( one_hot ) adding a dimension
#       output is smaller by one dimension for reasons that are unclear
#
def my_custom_logits_loss_V2( y_true, y_pred ):
    # tf.print("y_true=",y_true)
    # tf.print("y_pred=",y_pred)
    y_true = tf.squeeze( tf.one_hot( tf.cast(y_true,dtype=tf.int32), classification_count ), 1 )
    error = y_true - y_pred
    results2 = tf.reduce_mean( tf.square(error), axis=-1 )   # mean square error
    return results2



#
# model which consumes the last 'categorical' layer for 10 values
#
model.compile(optimizer='adam', loss=my_custom_logits_loss_V2, metrics=['sparse_categorical_accuracy'] )

#
#   train for training, test for validation
#   this usually hits 95% by epoch 10, although I have seen it get stuck around 50%
#
model.fit(
    train_features, train_labels,
    batch_size=16,
    epochs=EPOCHS,
    validation_data=(test_features, test_labels))


print(model.history.history['loss'])
print(model.history.history['sparse_categorical_accuracy'])
print(model.history.history['val_loss'])
print(model.history.history['val_sparse_categorical_accuracy'])

#
#   display evoluation of metrics
#
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])


plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(model.history.history['sparse_categorical_accuracy'])
plt.plot(model.history.history['val_sparse_categorical_accuracy'])

plt.show()
plt.pause( 300 )
