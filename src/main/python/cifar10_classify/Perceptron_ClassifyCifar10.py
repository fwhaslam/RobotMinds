#
#   Tensorflow Basic Image Classification.
#
#   This Perceptraion classifies images from the CIFAR-10 dataset.
#       https://www.cs.toronto.edu/~kriz/cifar.html
#       https://www.tensorflow.org/datasets/catalog/cifar10
#       https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cifar.py
#
#   This is a copy of 11_BasicImageClassification.py with the following changes:
#       Fashion dataset is replaced with CIFAR-10 dataset
#       Processing does not pause for displayed images.
#       Old images are closed as new images are opened.
#       Data shape is 32x32x3 not 28x28
#       Added layer to perceptron, taking epoch 50 performance from ~50% to ~60%
#
#   Code tested with:
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import _utilities.tf_loading_tools as loader


print(tf.__version__)

(train_images, train_label_array), (test_images, test_label_array) = datasets.cifar10.load_data()
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels = train_label_array[:,0]
test_labels = test_label_array[:,0]

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

CHECKPOINT_FILE = 'perceptron_cifar10_ckpt'
plt.ion()

# check the dataset parameters
print("train[s]=", str(train_images.shape) )
print("len=", str(len(train_labels)) )
print("train labels = ", str(train_labels.shape) )
print("test labels = ", str(test_labels.shape) )
print("image shape = ", str(test_images.shape) )
print("len(labels) = ", len(test_labels) )

# inspect first image in dataset
lastFigure = plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.pause(10)

# scale image values
train_images = train_images / 255.0
test_images = test_images / 255.0

# display first 25 images
plt.close( lastFigure )
lastFigure = plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
plt.pause(10)

# create and train model
# this model is over 60% at epoch 50 and over 70% at epoch 100
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])

model = create_model()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

loader.try_load_weights( CHECKPOINT_FILE, model )

model.fit(train_images, train_labels, epochs=50)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# make predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print("pred="+ str(predictions[0]) )

np.argmax(predictions[0])

print("tlab="+ str(test_labels[0]) )

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

# verify predictions
plt.close( lastFigure )
i = 0
lastFigure = plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
plt.pause( 10 )

plt.close( lastFigure );
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
plt.pause( 10 )

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
plt.close( lastFigure )
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

