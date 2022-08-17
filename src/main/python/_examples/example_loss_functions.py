#
#   The documentation on loss function input + output is confusing
#
#   I write this to learn how to create/convert arrays of integer + float
#   into the shape expected by typical loss functions.
#
#   The implementation of the MeanSquaredError loss function says that the
#   output will be one dimension smaller than the input.  Implicit in this
#   is that the output will have one result per batch element.  That 'one
#   result' can be multi-dimensional.
#
#   I am still not sure why the result is one dimension smaller.  I assume
#   it has to do with back-propogation techniques.
#

import tensorflow as tf
from keras import backend as K

print("Testing tf.unravel_index for suitability in logit analysis")

# 5x10 array with 0-9 in each row
target1 = tf.expand_dims( tf.range(0, 10), axis=0 )
print("=",target1)
target2 = tf.repeat( target1, 5, axis=0 )
print("target2=",target2)

# add x10 to each row
add_it = tf.range( 0, 50, delta=10 )
add_it = tf.expand_dims( add_it, axis=1 )
print("add_it=",add_it)

# target3 is .00-,49 over 5 rows of 10
target3 = tf.cast( target2 + add_it, dtype=tf.float32 ) / 100.
print("target3=",target3)


# first index pattern = simple list
indices1 = tf.constant( [2,6,4,0,6] )
print("indices1=",indices1)

# each element in its own row
# indices2 = tf.constant( [[2], [6], [3], [0], [6]] )
indices2 = tf.expand_dims( indices1, axis=1 )
print("indices2=",indices2)


# find value from target using row index ( gather_nd )
# select = tf.one_hot( indices1, 10 )
select = tf.squeeze( tf.one_hot( indices2, 10 ), 1 )
print("select=",select)
# results = tf.gather_nd( target3, indices1 )
results = tf.cast(target3,dtype=tf.float32) * select
print("results=",results)

sums = tf.reduce_sum( results, 1 )
print("sums=",sums)



#
# Simplify to a single method, batch is the first dimension [n,:]
#
# inputs:   y_true nx1 = [[a],[b],..[z]]
#           y_pred = nx10 = [ [a0,b0,...j0], [a1,b1,...j1], ... [an,bn,...jn] ]
#           y_true is [5,1]
#           y_pred is [5,10]
#
# returns:  values selected from outputs by index [ c0, a1, .. hn ]
#           one value per sample
#

def my_test_lost( y_true, y_pred ):
    new_y_true = tf.squeeze( tf.one_hot( tf.cast(y_true,dtype=tf.int32), 10 ), 1 )
    print("\ny_true.shape", tf.shape( new_y_true.shape )  )
    print("new_y_true.shape", tf.shape( new_y_true.shape )  )
    print("y_pred.shape", tf.shape( new_y_true.shape )  )
    return 1 - tf.reduce_sum( y_pred * new_y_true, 1 )

def my_test_lost_with_mean_square( y_true, y_pred ):
    new_y_true = tf.squeeze( tf.one_hot( tf.cast(y_true,dtype=tf.int32), 10 ), 1 )
    print("\ny_true.shape", tf.shape( new_y_true.shape )  )
    print("new_y_true.shape", tf.shape( new_y_true.shape )  )
    print("y_pred.shape", tf.shape( new_y_true.shape )  )
    return tf.map_fn( tf.reduce_mean, tf.math.squared_difference(new_y_true, y_pred) )
    # return tf.math.reduce_mean( tf.math.squared_difference(new_y_true, y_pred), axis=-1 )

def my_test_loss_with_mse(y_true, y_pred): # this is essentially the mean_square_error
    new_y_true = tf.squeeze( tf.one_hot( tf.cast(y_true,dtype=tf.int32), 10 ), 1 )
    print("\ny_true.shape", tf.shape( new_y_true.shape )  )
    print("new_y_true.shape", tf.shape( new_y_true.shape )  )
    print("y_pred.shape", tf.shape( new_y_true.shape )  )
    return tf.keras.losses.mean_squared_error(new_y_true,y_pred)    #[:,2])
    # return tf.keras.losses.mean_squared_error(new_y_true,y_pred[:,2])


test_y_true = tf.cast( tf.expand_dims( tf.constant( [2,6,4,0,6] ), 1 ), dtype=tf.float32 )

test_y_pred = tf.cast(
        tf.repeat( tf.expand_dims( tf.range(0, 10), axis=0 ), 5, axis=0 ) +
            tf.expand_dims( tf.range( 0, 50, delta=10 ), axis=1 ),
        dtype=tf.float32 ) / 100

print("test_y_true=",test_y_true)
print("test_y_pred=",test_y_pred)

print("TESTLOSS=", my_test_lost( test_y_true,test_y_pred ) )
print("MeanSquare_LOSS=", my_test_lost_with_mean_square( test_y_true,test_y_pred ) )
print("MSE_LOSS=", my_test_loss_with_mse( test_y_true,test_y_pred ) )



###################################
#
#   test loss using example code and scalar data in batches
#   see: https://stackoverflow.com/questions/63390725/should-the-custom-loss-function-in-keras-return-a-single-loss-value-for-the-batc
#   see: https://github.com/tensorflow/tensorflow/issues/42446
#   see: (fixed) https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
#   see: https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError
#
##################################

# fixed: need to retain the batch_size ... so result is at least [batch_size,]
#       first pass: I used tf.map_fn() which created one scalar per batch element
#       second pass: I copied the 'remove last axis' strategy from MeanSquaredError
def custom_mean_squared_error(y_true, y_pred):
    # y_pred = tf.convert_to_tensor(y_pred)
    # y_true = tf.cast(y_true, y_pred.dtype)
    print("\ny_true=", test_y_true )
    print("y_pred=", test_y_pred )
    sqrs = tf.square(y_true - y_pred)
    result = K.mean( sqrs, axis=-1 ) # same as tf.math.reduce_mean
    # result = tf.map_fn( K.mean, sqrs )
    print("result=",result )

    return result


test_y_true = tf.constant( [[1],[2],[3],[4],[5]] )
test_y_pred = tf.constant( [[2],[3],[4],[5],[6]] )
custom_mean_squared_error(test_y_true,test_y_pred)

test_y_true = tf.constant( [[1,1],[2,2],[3,3],[4,4],[5,5]] )
test_y_pred = tf.constant( [[2,2],[3,3],[4,4],[5,5],[6,6]] )
custom_mean_squared_error(test_y_true,test_y_pred)

test_y_true = tf.constant( [[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]],[[5,5],[5,5]]] )
test_y_pred = tf.constant( [[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]],[[5,5],[5,5]],[[6,6],[6,6]]] )
custom_mean_squared_error(test_y_true,test_y_pred)



work = tf.range(0,90)
print("work=",work)

for value in work[:-23]:
    print("value=",value)