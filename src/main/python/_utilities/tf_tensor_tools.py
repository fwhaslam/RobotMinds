import tensorflow as tf
print("TensorFlow version:", tf.__version__ )

# from tensorflow.python.ops.numpy_ops import np_config
# tf.python.ops.numpy_ops.enable_numpy_behavior()

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tensor_to_value( some_tensor:tf.Tensor ):
    r"""Extract tensor value.  This will return the value of the tensor in the same shape as the tensor."""
    if some_tensor is None: return None
    if tf.executing_eagerly(): return some_tensor.numpy()
    return some_tensor.eval()

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
    Produces results similar to argmax, but is differentiable.

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

