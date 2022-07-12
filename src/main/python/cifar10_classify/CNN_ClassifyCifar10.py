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
#

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shared_tf_utilities as stfu

print(tf.__version__)

(train_images, train_label_array), (test_images, test_label_array) = datasets.cifar10.load_data()
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels = train_label_array[:,0]
test_labels = test_label_array[:,0]

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

CHECKPOINT_FILE = 'cnn_cifar10_ckpt'
plt.ion()

# check the dataset parameters
print("train_images.shape=", str(train_images.shape) )
print("train_labels.shape = ", str(train_labels.shape) )
print("test_image.shape = ", str(test_images.shape) )
print("test_labels.shape = ", str(test_labels.shape) )
print("class_names.len = ", len(class_names) )
print()

# scale image values
train_images = train_images / 255.0
test_images = test_images / 255.0

# create and train model
# this has accuracy 95% at epoch 12 / 98% at epoch 25
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))
    return model

model = create_model()
print()

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
stfu.try_load_weights( CHECKPOINT_FILE, model )

model.fit(train_images, train_labels, epochs=50)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
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

