
#
#   Kaggle example from:
#       Code: https://www.kaggle.com/code/aasthanarang23/cartoonclassification
#       Dataset: https://www.kaggle.com/datasets/volkandl/cartoon-classification
#
#   Cartoon Image Classifier using a Pre-Trained Model
#       Also shows how to use ImageDataGenerator
#
#   This code is identical to what is presented on the source page except for:
#       some comments are added for clarity
#       location of dataset has changed
#       Model was altered around layer -6 to fix a shape error
#
#   Code tested with:
#       Tensorflow 2.9.1 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow as tf
# import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model, load_model
from pathlib import Path


img_height=180
img_width=320

# TODO: change this path to a correct local path
# root_path = os.path.expanduser( '~/_Workspace/Datasets/KaggleCartoon/cartoon_classification' )
root_path = Path( os.path.expanduser( 'd:/Datasets/KaggleCartoon/cartoon_classification' ) )

train_path= root_path / "TRAIN"
test_path= root_path / "TEST"
# train_path="../input/cartoon-classification/cartoon_classification/TRAIN"
# test_path="../input/cartoon-classification/cartoon_classification/TEST"

batch_size = 8
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

#
#   It looks like the intent here was to output a convolutional layer with 10 values
#   This may have broken if the 'MobileNet' model implementation changed over time.
#
#   Fred:   Switching from layer -6 to layer -5, and adding a 'Flatten' layer
#           The original model appears to have 91 layers ( geeez )

# x = mobile.layers[-6].output
# output = Dense(units=10, activation='softmax')(x)
layerM5 = mobile.layers[-5].output
flattenM5 = Flatten()(layerM5)        # inserting a extra 'flatten' layer
output = Dense(units=10, activation='softmax')(flattenM5)

# pre-trained model using 'imagenet'
model = Model(inputs=mobile.input, outputs=output)

# the first N-23 layers are NOT trainable, they keep their weights
# only the last 23 layers will train, and five of those are skipped ( layerM5->output )
for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

#
#   Wrapper function to investigate the shape of the inputs and outputs
#
#   input:  y_true.shape = [batch_size, one_hot_array]
#           y_pred.shape = [batch_size, array_of_weights]
#           both inputs are the same shape ...
#   output: result.shape = scalar ( default behavior for CCE is 'scalar = sum / batch_size' )
#
def my_custom_loss(y_true,y_pred):
    tf.print("y_true=",y_true)
    tf.print("y_pred=",y_pred)
    result = tf.keras.losses.CategoricalCrossentropy()(y_true,y_pred)
    tf.print("result=",result)
    return result



model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(learning_rate=0.001), loss=my_custom_loss, metrics=['accuracy'])


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

checkpoint_filepath = './'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#
#   This model is often 90% accurate in the first epoch due to pre-training ( not checkpoints )
#
hist = model.fit( x=train_generator,
               steps_per_epoch=1500,
               validation_data=test_generator,
               validation_steps=220,
               epochs=8,
               verbose=1,
               callbacks=[model_checkpoint_callback] )

model.load_weights(checkpoint_filepath)

fig1 = plt.gcf()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()