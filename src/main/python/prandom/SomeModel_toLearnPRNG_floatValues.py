#
#   Source:
#       _other_tutorials/mlm_RNN_withAttention_toFibonacci.py
#
#   Recurrent Neural Network with Attention to learn Random Number Generator.
#
#   This is an exact copy of the code with the following changes:
#       Data samples replaced with simple PseudoRandomGenerator
#
#   Code tested with:
#

import sys
sys.path.append("..")

import tensorflow as tf
import prandom.pseudo_random_tools as psrt

# Set up parameters
TIME_STEPS = 3
HIDDEN_UNITS = 3
OUTPUT_UNITS = 1
EPOCHS = 30

VERSION = 2

# note, this has a cycle of 887 values in the range 0-1
pseudo_random_generator = psrt.SimplePseudoRandomGenerator(0, 29, 887)

( data, labels ) = psrt.generate_examples( 887, pseudo_random_generator, TIME_STEPS, norm=1. )

# data.shape=(bs,time_steps,1), labels.shape= (bs,1)
trainX = data[0:400]
trainY = labels[0:400]
print('trainX.shape=',tf.shape(trainX))
print('trainY.shape=',tf.shape(trainY))

testX = data[600:700]
testY = labels[600:700]
print('testX.shape=',tf.shape(testX))
print('testY.shape=',tf.shape(testY))

########################################################################################################################

model_function = [
    lambda: create_model_v1,
    lambda: create_model_v2,
    lambda: create_model_v3
]

# Create a traditional RNN network
def create_model_v1():
    x = inputs = tf.keras.layers.Input( (TIME_STEPS,1) )
    x = tf.keras.layers.SimpleRNN( HIDDEN_UNITS, activation='tanh')(x)
    outputs = x = tf.keras.layers.Dense( units=OUTPUT_UNITS, activation='tanh' )(x)
    return tf.keras.Model( inputs=inputs, outputs=outputs )

def create_model_v2():
    x = inputs = tf.keras.layers.Input( (TIME_STEPS,1) )
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense( 1024, activation='LeakyReLU' )(x)
    x = tf.keras.layers.Dense( 256, activation='LeakyReLU' )(x)
    x = tf.keras.layers.Dense( 16, activation='sigmoid')(x)
    outputs = x = psrt.BitsToNumber(16) (x)
    return tf.keras.Model( inputs=inputs, outputs=outputs )

def create_model_v3():
    r"""Use one dimensional CNN, which means that each bit pair can learn addition ... hmm, carryover ?"""
    x = inputs = tf.keras.layers.Input( (TIME_STEPS,1) )
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense( 1024, activation='LeakyReLU' )(x)
    x = tf.keras.layers.Dense( 256, activation='LeakyReLU' )(x)
    x = tf.keras.layers.Dense( 16, activation='sigmoid')(x)
    outputs = x = psrt.BitsToNumber(16) (x)
    return tf.keras.Model( inputs=inputs, outputs=outputs )

########################################################################################################################

model = create_model_v2()
model.summary()

# Train the network
optimizer = tf.keras.optimizers.Nadam() # 0.0001 )
# optimizer = tf.keras.optimizers.SGD()
model.compile(loss='mse', optimizer=optimizer)
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=1, verbose=2)


# Evalute model
train_mse = model.evaluate(trainX, trainY)
test_mse = model.evaluate(testX, testY)

# Print error
print("Train set MSE = ", train_mse)
print("Test set MSE = ", test_mse)
