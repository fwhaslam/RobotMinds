#
#   Tensorflow Basic Image Classification.
#
#   This Perceptraion classifies images from a cartoon dataset:
#       https://www.kaggle.com/datasets/kanakmittal/anime-and-cartoon-image-classification
#       https://dl.acm.org/doi/abs/10.1145/3284398.3284403
#
#   NOTE: this uses the ImageDataGenerator/flow_from_directory() method to create the dataset.
#
#   This is a copy of CNN_ClassifyCifar10.py.py with the following changes:
#       IDG = using ImageDataGenerator.flow_from_directory()
#
#   Code tested with:
#

import sys
sys.path.append('..')

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# local code
import _utilities.tf_loading_tools as loader

#
#   Constants
#

print(tf.__version__)
plt.ion()

IMAGE_SIZE = ( 180, 320, 3 )
IMAGE_SHAPE = IMAGE_SIZE[:2]    # ( 180, 320 )

print("SIZE=",IMAGE_SIZE )
print("SHAPE=",IMAGE_SHAPE )

img_height = 180
img_width = 320
batch_size = 8

root_path = os.path.expanduser( 'd:/Datasets/KaggleCartoon/cartoon_classification' )
train_path = root_path + "/TRAIN"
test_path = root_path + "/TEST"

#
#   create TRAIN and TEST datasets using ImageDataGenerator/flow_from_directory
#

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    Path(train_path),
    target_size=IMAGE_SHAPE,
    batch_size=batch_size,
    class_mode='categorical')
print("train_generator=",train_generator)
# print("train_generator.shape=",train_generator.shape)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    Path(test_path),
    target_size=IMAGE_SHAPE,
    batch_size=batch_size,
    class_mode='categorical')

print("test_generator=",test_generator)



class_names = ['adventure_time', 'catdog', 'Familyguy', 'Gumball', 'pokemon',
               'smurfs', 'southpark', 'spongebob', 'tom_and_jerry', 'Tsubasa']
CHECKPOINT_FILE = 'cnn_cartoon_ckpt'

print("class_names.len = ", len(class_names) )
print()


# create and train model
def create_model( ds_shape, batch_size ):
    model = models.Sequential()
    # model.add( layers.Rescaling(1./255) )
    model.add( layers.Input( ds_shape, batch_size ) )
    model.add( layers.Conv2D(32, (5, 5), activation='relu') )
    model.add( layers.Flatten() )
    model.add( layers.Dense(64, activation='relu') )
    model.add( layers.Dense(10))
    # model.add( layers.Reshape( (10) ) )
    # model.add( layers.CategoryEncoding () )
    return model

model = create_model( IMAGE_SIZE, batch_size )
print()

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])
loader.try_load_weights( CHECKPOINT_FILE, model )

#
#   Lets Do Some Training!
#
print("TRAIN GENERATOR=",train_generator)

model.fit_generator( train_generator, validation_data=test_generator, epochs=50 )

test_loss, test_acc = model.evaluate( test_generator, verbose=2)
# model.fit(train_images, train_labels, epochs=50)
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print("\npred="+ str(predictions[0]) )

np.argmax(predictions[0])
print("\ntlab="+ str(test_labels[0]) )

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

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
plt.pause( 30 )

