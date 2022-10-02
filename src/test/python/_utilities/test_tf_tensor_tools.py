import sys
sys.path.append('..')
sys.path.append('../../../main/python')

import tensorflow as tf

# testing modules
import unittest

# module under test
import _utilities.tf_tensor_tools as teto


class test_tf_tensor_tools(unittest.TestCase):

    def test__tensor_to_value__single(self):
        some_tensor = tf.constant( 4.5 )
        result = teto.tensor_to_value(some_tensor)
        self.assertEquals( '4.5', str(result) )

    def test__tensor_to_value__1dim(self):
        some_tensor = tf.constant( [4.5] )
        result = teto.tensor_to_value(some_tensor)
        self.assertEquals( '[4.5]', str(result) )

    def test__tensor_to_value__2dim(self):
        some_tensor = tf.constant( [[7.0],[4.5]] )
        result = teto.tensor_to_value(some_tensor)
        self.assertEquals( '[[7. ]\n'
                           ' [4.5]]', str(result) )

if __name__ == '__main__':
    unittest.main()