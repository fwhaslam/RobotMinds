import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

def tensor_to_value( some_tensor:tf.Tensor ):
    r"""Extract tensor value.  This will return the value of the tensor in the same shape as the tensor."""
    if tf.executing_eagerly(): return some_tensor.numpy()
    return some_tensor.eval()