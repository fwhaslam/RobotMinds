#
#   Image classification with embedded label.
#   The goal is to repeatedly develop an embedding,
#       then analyze the diverse results to see if there are common relations.
#   My theory is that digits with similar appearance will develop similar embeddings.
#           eg.  0 resembles 8, 6 resembles 6, 7 resembles 1, 3 resembles 2
#
#   This is a copy of 11_BasicImageClassification.py
#
#   This code has the following changes:
#       MNIST fashion set replaces with MNIST digits set.
#       The images + labels are passed in as pairs, with about half the pairs being invalid.
#       The model must judge which pairs are valid.
#       The labels are passed in via embedding.
#
#   Code tested with:
#

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

print(tf.__version__)


########################################################################################################################
#   Prepare images, labels, and truth sets

def mix_data_set( labels ):
    r"""Labels are either kept, paired with True; or replaced, paired with False
    Return one tensor with revised labels, and one tensor with truth values."""

    values = np.empty( labels.shape, dtype=int )
    truths = np.empty( labels.shape )

    for index in range( len(labels) ):
        is_true = rnd.randint(0,1)
        truths[index] = is_true
        if is_true:
            values[index] = labels[index]
        if not is_true:
            value = rnd.randint(0,8)
            if value>=labels[index]: value += 1     # must match old value
            values[index] = value

    return values, truths

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# training data
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

train_values, train_truths = mix_data_set( train_labels )
# train_values_hot = tf.one_hot( train_values, 10, axis=-1 )
print('train_labels=',train_labels)

train_input_set = [train_images,train_values]
train_eval_set = train_truths

# test data
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

test_values, test_truths = mix_data_set( test_labels )
# test_values_hot = tf.one_hot( test_values, 10, axis=-1 )

test_input_set = [test_images,test_values]
test_eval_set = test_truths

# description of labels
class_names = ['zero','one','two','three','four','five','six','seven','eight','nine']
truth_name = ['false','true']

########################################################################################################################

# display first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]]+'/'+class_names[train_values[i]]+' '+str(train_truths[i]) )
plt.show()

########################################################################################################################

def create_model( shape ):

    x = input1 = tf.keras.Input(shape=shape)    # (bs, 28,28)
    x = tf.keras.layers.Flatten()(x)        # ( bs, 28x28 )

    x2 = input2 = tf.keras.Input(shape=(1,) )
    x2 = tf.keras.layers.Embedding( 10, 4, name='digit_embedding' )( x2 )   # ( bs, 1,4 )
    x2 = tf.keras.layers.Flatten()(x2)              # ( bs, 4 )

    x = tf.keras.layers.Concatenate()( [x,x2] )
    x = tf.keras.layers.Dense(64, activation='selu')(x)
    # x = tf.keras.layers.Dense(256, activation='selu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs = x

    return tf.keras.Model(inputs=[input1,input2], outputs=outputs,name='model_v1')

model = create_model( (28,28) )

layer = model.get_layer( 'digit_embedding' )
print("\nEmbeddingBefore=",layer.weights)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit( x=train_input_set, y=train_eval_set, epochs=10)

test_loss, test_acc = model.evaluate( test_input_set, test_eval_set, verbose=2 )
print('\nTest accuracy:', test_acc)

########################################################################################################################
# Investigate embedding

layer = model.get_layer( 'digit_embedding' )
print("\nEmbeddingAfter=",layer.weights)

########################################################################################################################
#
# # make predictions
# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)
# print("pred="+ str(predictions[0]) )
# np.argmax(predictions[0])
# print("tlab="+ str(test_labels[0]) )
#
#
# def plot_image(i, predictions_array, true_label, img):
#     true_label, img = true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100*np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
# def plot_value_array(i, predictions_array, true_label):
#     true_label = true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#
# # verify predictions
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()
#
# # Plot the first X test images, their predicted labels, and the true labels.
# # Color correct predictions in blue and incorrect predictions in red.
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
#
