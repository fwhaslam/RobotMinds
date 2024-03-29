import tensorflow as tf
import inspect

def loss_has_gradient( loss_function, arg_count=1, output_shape=(1,) ):
    r"""Verify that a loss function does not throw a ValueError"""
    try:
        train_x = tf.ones( [1,] )
        train_y = tf.ones( [1,] )
        output_size = tf.size( tf.zeros(output_shape) ) # inefficient, but this is just a test

        def test_func(x,y):
            r"""Adjust function to arg_count."""
            fields = [y] * arg_count
            return loss_function( *fields )

        inputs = tf.keras.layers.Input(shape=(1,))
        hidden = tf.keras.layers.Dense(output_size)(inputs)
        outputs = tf.keras.layers.Reshape( output_shape )(hidden)

        tf.print('outputs.shape=',outputs.shape)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='sgd', loss=test_func)
        model.fit(train_x, train_y)
        return True
    except ValueError as err:
        if (err.args[-1].find('ValueError: No gradients provided for any variable:')<0):
            raise err     # unknown error occured, re-throw to top
        return False
