#
#   Common Tools to simplify some resnet tasks in tensorflow and python
#

import tensorflow as tf

#
#   Simple skip block.  Cannot change image_size nor channel_size.
#   ? Use when input and output shapes are the same
#
def identity_block( input, filter_count, kernel_size=3, stride=1 ):
    # copy tensor to chaining variable x
    x = input
    # Layer 1
    x = tf.keras.layers.Conv2D(filter_count, kernel_size=kernel_size, stride=stride, padding = 'same')( x )
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter_count, kernel_size=kernel_size, stride=stride, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, input])
    output = tf.keras.layers.Activation('relu')(x)
    return output

def identity_block_layer( input, filter_count, name='identity' ):
    output = identity_block( input, filter_count )
    return tf.keras.Model( input, output, name )

def id_block( filter_count ):
    def id_block_full( input ):
        return identity_block( input, filter_count )
    return id_block_full

def id_layer( input, filter_count, name='identity' ):
    return identity_block_layer( input, filter_count, name )

#
#   Convolutionary skip block.
#   filter_count = defines new channel_size, can replace old channel_size
#   strides = can divide the image_size ( usually =2 which will give half size )
#
def projection_block( input, filter_count, kernel_size=3, strides=1  ):
    # copy tensor to chaining variable x
    x = input
    # Layer 1
    x = tf.keras.layers.Conv2D(filters=filter_count, kernel_size=kernel_size, padding = 'same', strides=strides)( x )
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter_count, kernel_size, strides=1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_res = tf.keras.layers.Conv2D(filter_count, (1,1), strides=strides)(input)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_res])
    output = tf.keras.layers.Activation('relu')(x)
    return output

def projection_block_layer( input, filter_count, kernel_size=3, strides=1, name='proj-block' ):
    output = projection_block( input, filter_count, kernel_size, strides )
    return tf.keras.Model( input, output, name )

def proj_block( filter_count, kernel_size=3, strides=1 ):
    def proj_block_full( input ):
        return projection_block( input, filter_count, kernel_size, strides )
    return proj_block_full

def proj_layer( input, filter_count, kernel_size=3, strides=1, name='proj-block' ):
    return projection_block( input, filter_count, kernel_size, strides, name )

#
#   Alternative from: https://towardsdatascience.com/creating-deeper-bottleneck-resnet-from-scratch-using-tensorflow-93e11ff7eb02
#Conv-BatchNorm-ReLU block
#
# def conv_batchnorm_relu( x, filter_count, kernel_size=3, strides=1):
#
#     x = tf.keras.layers.Conv2D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.ReLU()(x)
#
#     return x
#
# #Identity block
#
# def identity_block(tensor, filter_count):
#
#     x = conv_batchnorm_relu(tensor, filter_count=filter_count, kernel_size=1, strides=1)
#     x = conv_batchnorm_relu(x, filter_count=filter_count, kernel_size=3, strides=1)
#     x = tf.keras.layers.Conv2D(filters=4*filter_count, kernel_size=1, strides=1)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#
#     x = tf.keras.layers.Add()([tensor,x])    #skip connection
#     x = tf.keras.layers.ReLU()(x)
#
#     return x
#
# #Projection block
#
# def projection_block(tensor, filter_count, strides):
#
#     #left stream
#     x = conv_batchnorm_relu(tensor, filter_count=filter_count, kernel_size=1, strides=strides)
#     x = conv_batchnorm_relu(x, filter_count=filter_count, kernel_size=3, strides=1)
#     x = tf.keras.layers.Conv2D(filters=4*filter_count, kernel_size=1, strides=1)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#
#     #right stream
#     shortcut = tf.keras.layers.Conv2D(filters=4*filter_count, kernel_size=1, strides=strides)(tensor)
#     shortcut = tf.keras.layers.BatchNormalization()(shortcut)
#
#     x = tf.keras.layers.Add()([shortcut,x])    #skip connection
#     x = tf.keras.layers.ReLU()(x)
#
#     return x
#
#
# #Resnet block
#
# def resnet_block(x, filters, reps, strides):
#
#     x = projection_block(x, filters, strides)
#     for _ in range(reps-1):
#         x = identity_block(x,filters)
#
#     return x


#
#   Implementation of bottleneck block from:
#       https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c
#
# def bottleneck_block(X, f, filters, stage, block):
#     """
#     Implementation of the identity block
#
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#
#     Returns:
#     X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
#     """
#
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value. You'll need this later to add back to the main path.
#     X_shortcut = X
#
#     # First component of main path
#     X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)
#
#
#     # Second component of main path
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)
#
#     # Third component of main path
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
#
#
#     return X
