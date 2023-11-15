#
#   Tensorflow Basic Image Classification.
#
#   This Perceptraion classifies images from a cartoon dataset:
#       https://www.kaggle.com/datasets/kanakmittal/anime-and-cartoon-image-classification
#       https://dl.acm.org/doi/abs/10.1145/3284398.3284403
#
#   Source code from:
#       local source CNN_ClassifyCifar10.py
#
#   This is a copy of the code with the following changes:
#       GBB = GeneratorBasedBuilder see _datasets/KaggleCartoonDataset.py
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models

from _datasets import KaggleCartoonDataset

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import _utilities.tf_loading_tools as loader


print(tf.__version__)
plt.ion()


# (train_images, train_label_array), (test_images, test_label_array) = datasets.cifar10.load_data()
# # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
# train_labels = train_label_array[:,0]
# test_labels = test_label_array[:,0]


# mnist_builder = tfds.builder("KaggleCartoonDataset")
# mnist_info = mnist_builder.info
# mnist_builder.download_and_prepare()
# datasets = mnist_builder.as_dataset()
# (train_images, train_label_array), (test_images, test_label_array) = datasets.load_data()
#
# tf.keras.datasets.fashion_mnist.load
#
# if (True):
#     exit()

ds_preview, ds_info = tfds.load('KaggleCartoonDataset',with_info=True)
ds_train, ds_test = ds_preview["train"], ds_preview["test"]
assert isinstance(ds_train, tf.data.Dataset)

ds_train = ds_train.batch(32).prefetch(1)
ds_test = ds_test.batch(32).prefetch(1)

ds_shape = ds_info.features['image'].shape
print("DS_INFO=",ds_info)
print("DS_INFO/shape=",ds_shape)
print("DS_INFO/shape2=", ds_shape[:2] )
print()

# make it small for development
ds_train = ds_train.take(3)
# ds_view = ds_train.take(1)
# print("FIRST=",list(ds_view))

#
# transform images to be a consistent size
#
def alter( x, shape ):
    # print( "X=", x )
    x['image'] = tf.image.resize( x['image'], shape, preserve_aspect_ratio=True )
    return x

ds_train = ds_train.map( lambda x: alter(x, ds_shape[:2]) )
ds_test = ds_test.map( lambda x: alter(x, ds_shape[:2]) )

# def alter_record( record, shape ):
#     record.x['image'] = tf.image.resize( record.x['image'], shape, preserve_aspect_ratio=True )
#
# for record in ds_train:
#     record.x['image'] = alter_record( record, ds_shape[:2] )
#
# for record in ds_test:
#     record.x['image'] = alter_record( record, ds_shape[:2] )

#
#   Examine shapes and info
#
print("Label Count = ", ds_info.features['label'].num_classes )
print("Train Count = ", ds_info.splits['train'].num_examples )
print("Test Count = ", ds_info.splits['test'].num_examples )
print()

print("DS_TRAIN="+str(ds_train.__dict__.keys()))
print("DS_TRAIN_KEYS="+str(ds_train._input_dataset.__dict__.keys()))
print("DS_TEST="+str(ds_test.__dict__.keys()))
print("DS_TEST_KEYS="+str(ds_test._input_dataset.__dict__.keys()))
print()

#
#   Reorganize to match train/labels paradigm similar to tfds.load_datq
#
train_images = ds_train.map( lambda x: x['image'] )
train_labels = ds_train.map( lambda x: x['label'] )
# train_images, train_labels = tuple(zip(*ds_train))
print("TRAIN_IMAGES.len=",len(train_images))
print("TRAIN_LABELS.len=",len(train_labels))
print("TRAIN_IMAGES=",train_images)
print("TRAIN_LABELS=",train_labels)
print()

test_images = ds_test.map( lambda x: x['image'] )
test_labels = ds_test.map( lambda x: x['label'] )
# test_images, test_labels = tuple(zip(*ds_test))
print("test_images.len=",len(test_images))
print("test_labels.len=",len(test_labels))
print()

class_names = ['adventure_time', 'catdog', 'Familyguy', 'Gumball', 'pokemon',
       'smurfs', 'southpark', 'spongebob', 'tom_and_jerry', 'Tsubasa']

CHECKPOINT_FILE = 'cnn_cartoon_ckpt'


#
#
# check the dataset parameters
# print("train_images.shape=", str(train_images.shape) )
# print("train_labels.shape = ", str(train_labels.shape) )
# print("test_image.shape = ", str(test_images.shape) )
# print("test_labels.shape = ", str(test_labels.shape) )
print("class_names.len = ", len(class_names) )
print()

# scale image values
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# def normalize_img(image, label):
#     """Normalizes images: `uint8` -> `float32`."""
#     return tf.cast(image, tf.float32) / 255., label
#
# # normalize
# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
#
# # normalize
# ds_test = ds_test.map(
#     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# create and train model
def create_model( ds_shape ):
    model = models.Sequential()
    # model.add( layers.Rescaling(1./255) )
    model.add(layers.Input(ds_shape))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))
    # model.add( layers.CategoryEncoding( 10 ) )
    return model

model = create_model( ds_shape )
print()

model.summary()
model.compile(optimizer='adam',
              # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
loader.try_load_weights( CHECKPOINT_FILE, model )

#
#   Lets Do Some Training!
#
model.fit( x=train_images, validation_data=test_images, epochs=50, verbose=2 )
# model.fit( x=train_images, y=train_labels, epochs=50 )
test_loss, test_acc = model.evaluate( test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# make predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
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

