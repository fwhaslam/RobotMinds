#
#   Tensorflow Beginner Tutorial from
#       https://www.tensorflow.org/tutorials/quickstart/beginner
#
#   This is a Perceptron ( simple linear NN ), analyzing the MNIST Handwritten Digit dataset.
#
#   This code is identical to what is presented on the tutorial page except for:
#       some comments are added for clarity
#
#   Code tested with:
#       Tensorflow 2.8.0/cpuOnly  ( complains about Cudart64_110.dll, but functions )
#

import tensorflow as tf
print("TensorFlow version:", tf.__version__ )

mnist = tf.keras.datasets.mnist

# dataset storage path is relative to ~/.keras/datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
#predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])

