import tensorflow as tf

def loss_has_gradient( loss_function ):
    r"""Verify that a loss function does not throw a ValueError"""
    try:
        train_x = tf.constant([[1.]])
        train_y = tf.constant([[1.]])
        inputs = tf.keras.layers.Input(shape=(1,))
        model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(inputs))
        model.compile(optimizer='sgd', loss=loss_function)
        model.fit(train_x, train_y)
        return True
    except ValueError as err:
        if (err.args[-1].find('ValueError: No gradients provided for any variable:')<0):
            raise err     # unknown error occured, re-throw to top
        return False
