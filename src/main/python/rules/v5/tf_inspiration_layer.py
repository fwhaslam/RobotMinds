
import tensorflow as tf

from tensorflow.keras.layers import Layer

class InspirationLayer(Layer):
    """Create a layer of zero centered gaussian noise.
    This layer does not contain information from any other layer.
    It is a random, generative layer.

    Args:
      stddev: Float, standard deviation of the noise distribution.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor, used to determine 'batch' size for output

    Output shape:
      (batch,units)
    """

    def __init__(self, units, stddev=1., seed=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.stddev = stddev
        if (seed==None):
            self.generator = tf.random.Generator.from_non_deterministic_state()
        else:
            self.generator = tf.random.Generator.from_seed(seed)

    def call(self,inputs):
        r"""Inputs is need to determine batch size"""
        batch = tf.shape( inputs )[0]
        shape = ( batch, self.units, )
        return self.generator.normal( shape=shape, stddev=self.stddev )

    def get_config(self):
        config = {"stddev": self.stddev, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self):
        return ( self.units, )
