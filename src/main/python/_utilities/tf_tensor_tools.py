import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__ )

# from tensorflow.python.ops.numpy_ops import np_config
# tf.python.ops.numpy_ops.enable_numpy_behavior()

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tensor_to_value( some_tensor:tf.Tensor ):
    r"""Extract tensor value.  This will return the value of the tensor in the same shape as the tensor."""

    if some_tensor is None: return None
    if tf.executing_eagerly(): return some_tensor.numpy()
    return some_tensor.eval()

@DeprecationWarning   # this is not ready for use
def tensor_to_string( some_tensor:tf.Tensor ):
    r"""Produce a non-truncated string representing the tensor.
    Beware, this may be VERY large.
    This is primarily used by verbose testing."""

    # return np.array_str( some_tensor.numpy() )
    # return np.array2string( some_tensor.numpy(), threshold = np.inf)
    return np.array2string( some_tensor.numpy(), threshold = 1000)

@DeprecationWarning   # this is not ready for use
def tensor_assert_string( some_tensor:tf.Tensor, precision=2 ):
    r"""Produce a non-truncated string representing the tensor.
    Beware, this may be VERY large.
    This is primarily used by verbose testing."""

    # work = tf.strings.as_string( some_tensor )
    # return tf.strings.join( work, separator=" " )

    return tf.map_fn( )

@DeprecationWarning   # this is not ready for use
def tensor_string_formatter( some_tensor:tf.Tensor ):
    r"""Appied recursively to each level of the tensor.
    At the last level it produces strings, above that it formats as array."""

    depth = tf.shape(some_tensor).size

    if (depth==0):
        return tf.strings.as_string( some_tensor ).numpy().decode("utf-8")

    elif (depth==1):
        str_list = tf.strings.as_string( some_tensor )
        str_join = [ tf.strings.join( str_list, " " ) ]
        str_add = tf.concat( [ tf.constant( ["["] ), str_join, tf.constant( ["]"] ) ], axis=0 )
        str_merge = tf.strings.join( str_add )
        return str_merge.numpy().decode("utf-8")

    else:
        line_list = tf.map_fn( lambda x: tensor_string_formatter(x), some_tensor )
        str_join = [ tf.strings.join( line_list, "\n" ) ]
        str_add = tf.concat( [ tf.constant( ["["] ), str_join, tf.constant( ["]"] ) ], axis=0 )
        str_merge = tf.strings.join( str_add )
        return str_merge.numpy().decode("utf-8")

########################################################################################################################

@tf.function
def supersoftmax( alpha:tf.Tensor, beta=64. ):
    r"""Supersoftmax implementation, applied to last dimension of tensor.
    Produces results similar to softmax, but more closely approaches zero or one.

    :param alpha: a tensor matrix containing values we want to reduce
    :param beta: exponentiation factor which causes softmax to produce values closer to zero or one.
    :return: matrix which transforms the last dimension of alpha to a supersoftmax value."""

    betalpha = beta * alpha
    # tf.print('betalpha=',betalpha)
    # extreme exponentiation creates numbers close to zero or one ( ~one for biggest entry )
    return tf.nn.softmax( betalpha )

@tf.function
def softargmax( alpha:tf.Tensor, beta=64. ):
    r"""Softargmax implementation, applied to last dimension of tensor.
    This produces results similar to argmax, but is differentiable.

    :param alpha: a tensor matrix containing values we want to reduce
    :param beta: exponentiation factor which causes softmax to produce values closer to zero or one.
    :return: matrix which reduces the last dimension of alpha to a softargmax value.

    see: https://medium.com/@nicolas.ugrinovic.k/soft-argmax-soft-argmin-and-other-soft-stuff-7f94e6120dff
    see: https://www.tutorialexample.com/understand-softargmax-an-improvement-of-argmax-tensorflow-tutorial/
    """

    # increase array values so exponentiation of softmax is more extreme
    betalpha = beta * alpha
    # tf.print('betalpha=',betalpha)
    # extreme exponentiation creates numbers close to zero or one ( ~one for biggest entry)
    smax = tf.nn.softmax( betalpha )
    # tf.print('sm=',smax)
    # size of last dimension as a range of integers [ 0, 1, ... N ]
    pos = range( alpha.shape[-1] )
    # tf.print('pos=',pos)
    # array has ~zero for most entries, and ( ~one * index ) for maximum entry
    smpos = smax * pos
    # tf.print('smpos=',smpos)
    # sum of all values in array will approximate ( ~one * index )
    softargmax = tf.reduce_sum( smpos, axis=-1)
    # tf.print('softargmax=',softargmax)
    return softargmax

@tf.function
def simple_ratio( source:tf.Tensor ):
    r"""Sum and divide on last axis.
    Simple ratio function.
    Only works for positive numbers."""
    shape = tf.shape(source)
    sums = tf.math.reduce_sum( source, axis = -1 )
    basep = tf.concat( [shape[0:-1],(1,)], axis=0 )
    sums = tf.reshape( sums, basep )
    sums  = tf.repeat( sums, shape[-1], axis = -1 )
    return source / sums