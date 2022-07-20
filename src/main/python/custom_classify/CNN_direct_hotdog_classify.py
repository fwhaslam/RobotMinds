#
#   Dataset from:
#       https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog
#
#   Source code from:
#       https://www.kaggle.com/code/gabrielcarvalho11/(page-missing)
#       NOTE: apparently this page was removed
#
#   This is an exact copy of the code with the following changes:
#       code does not wait for image display to be closed
#       showing final plots
#       revised model
#       loading 'test' data instead of splitting 'train' data
#       load image using keras instead of cv2
#       refactored image displays as a method
#
#   Code tested with:
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import os
import numpy as np # linear algebra
from tensorflow import image, keras
from pathlib import Path

# TODO: modify path to match local user-space
root_path = os.path.expanduser( '~/_Workspace/Datasets/KaggleHotdog/hotdog_classify' )

# load feature + labels
IMAGE_SIZE = ( 200, 200, 3 )
IMAGE_RESIZE = list(IMAGE_SIZE[:2])     #  [200,200]
EPOCHS = 10

#
# load all JPGs from folder tree, determine the name of top-level folders as 'class' names
#
def load_features_and_labels( type ):
    path = Path( root_path + "/" + type )
    size = len([f for f in path.rglob('*.jpg')])
    folders = [f for f in os.listdir(path)]

    # pre-allocate space for all images + labels
    features = np.empty( (size,)+IMAGE_SIZE )       # (size, 200, 200, 3)
    labels = np.empty( (size) )

    idx = 0
    for img_path in path.rglob('*.jpg'):
        img = keras.utils.load_img( img_path )
        features[idx] = image.resize( img, IMAGE_RESIZE ) / 255.0
        labels[idx] = folders.index( img_path.parent.name )
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
#   NOTE: images are hotdogs in first half, not-hotdogs in second half
#
half_size = int( len(train_features) / 2 )
display_image_grid( 'Hotdogs!', train_features[:half_size] )
display_image_grid( 'NOT Hotdogs!', train_features[half_size:])

#
#   Model is optimized for training speed over accuracy.
#   CNNs used for rapid dropoff of image size
#   final dense layers are small enough to run on my machine
#
def create_model():
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(200, 200, 3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu' ))
    model.add(keras.layers.MaxPooling2D(3))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu' ))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(160, (2, 2), activation='relu' ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(80, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid')) # for binary data
    return model

model = create_model()
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#
#   train for training, test for validation
#   this usually hits 95% by epoch 10, although I have seen it get stuck around 50%
#
model.fit(train_features, train_labels, epochs=EPOCHS, validation_data=(test_features, test_labels))


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
