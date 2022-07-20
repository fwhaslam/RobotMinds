#
#   Dataset from:
#       https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog
#
#   Source code from:
#       https://www.kaggle.com/code/gabrielcarvalho11/hotdog-nothotdog-100-accuracy
#       NOTE: apparently this page was removed
#
#   This is an exact copy of the code with the following changes:
#       model.fit() switched from feature/labels to x/Y as author probably intended
#       code does not wait for image display to be closed
#       showing final plots
#
#   Code tested with:
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#
import numpy as np # linear algebra
import os
import cv2
from tensorflow import image

root_path = os.path.expanduser( '~/_Workspace/Datasets/KaggleHotdog/hotdog_classify/train' )
HOTDOGS_PATH = root_path + "/hotdog"
NOT_HOTDOGS_PATH = root_path + "/nothotdog"

# HOTDOGS_PATH = "../input/hotdog-nothotdog/hotdog-nothotdog/hotdog-nothotdog/train/hotdog"
# NOT_HOTDOGS_PATH = "../input/hotdog-nothotdog/hotdog-nothotdog/hotdog-nothotdog/train/nothotdog"

SIZE_IMAGES = 200, 200
features = np.empty((2*2121, 200, 200, 3))
labels = np.empty((2*2121))
for idx, img_name in enumerate(os.listdir(HOTDOGS_PATH)):
    img = cv2.imread(HOTDOGS_PATH + '/' + img_name)
    features[idx] = image.resize_with_pad(img, 200, 200)/255.0
    labels[idx] = 1

for idx, img_name in enumerate(os.listdir(NOT_HOTDOGS_PATH)):
    img = cv2.imread(NOT_HOTDOGS_PATH + '/' + img_name)
    features[2121+idx] = image.resize_with_pad(img, 200, 200)/255.0
    labels[2121+idx] = 0


import matplotlib.pyplot as plt
plt.ion()

plt.figure(figsize=(10, 10))

i = 0
plot_dim = 10

for x in features:
    plt.subplot(plot_dim, plot_dim, 1 + i)
    plt.axis('off')
    plt.imshow(x)
    i+=1
    if(i == plot_dim*plot_dim):
        break
plt.show()
plt.pause( 10 )


from sklearn.utils import shuffle

features, labels = shuffle(features, labels)

len_images = features.shape[0]

test_x, test_Y = features[:int(len_images*0.3)], labels[:int(len_images*0.3)]
x, Y = features[int(len_images*0.3):], labels[int(len_images*0.3):]

print(features.shape)
print(labels.shape)

print(test_x.shape)
print(test_Y.shape)

print(x.shape)
print(Y.shape)


from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.Input(shape=(200, 200, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = (200, 200, 3)))
model.add(keras.layers.MaxPooling2D(2, padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# was erroneous before, training included the test set
model.fit(x, Y, epochs=10, validation_data=(test_x, test_Y))
# model.fit(features, labels, epochs=10, validation_data=(test_x, test_Y))



print(model.history.history['loss'])
print(model.history.history['accuracy'])
print(model.history.history['val_loss'])
print(model.history.history['val_accuracy'])


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
