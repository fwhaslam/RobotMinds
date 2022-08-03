#
#   Dataset from:
#       https://www.kaggle.com/datasets/volkandl/cartoon-classification
#
#   Source code from:
#       local source CNN_direct_hotdog_classify.py
#
#   This is a copy of the code with the following changes:
#       switch to Kaggle Cartoon dataset
#       change IMAGE_SIZE, used IMAGE_SIZE in place of static values
#       added SKIP to reduce dataset to 1/32 in size ( skipping to every 32nd image )
#       fixed model + loss function to handle multiple categories
#       reduced model size to fit my local memory
#
#   Code tested with:
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import os
import numpy as np # linear algebra
from tensorflow import image, keras
from pathlib import Path

# TODO: modify path to match local user-space
root_path = os.path.expanduser( '~/_Workspace/Datasets/KaggleCartoon/cartoon_classification' )

# load feature + labels
IMAGE_SIZE = ( 180, 320, 3 )
IMAGE_RESIZE = list(IMAGE_SIZE[:2])     #  [180,320]
EPOCHS = 10
SKIP = 32       # only process one out of SKIP from available images

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
    labels = np.empty( (keep_size) )

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
    model.add(keras.layers.Dense(10))  # shaped to work with SparseCategoricalCrossentropy
    return model

model = create_model()
model.summary()

#
# model which consumes the last 'categorical' layer for 10 values
#
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

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
print(model.history.history['accuracy'])
print(model.history.history['val_loss'])
print(model.history.history['val_accuracy'])

#
#   display evoluation of metrics
#
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])


plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])

plt.show()
plt.pause( 300 )
