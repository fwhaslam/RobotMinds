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

########################################################################################################################

    def test__softargmax__1dim(self):
        input = tf.constant( [1.0,1.1,0.9] )
        result = teto.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor(0.99834394, shape=(), dtype=float32)', str(result) )

    def test__softargmax__2dims(self):
        input = tf.constant( [[1.0,1.1,0.9]] )
        result = teto.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor([0.99834394], shape=(1,), dtype=float32)', str(result) )

    def test__softargmax__3dims(self):
        input = tf.constant( [[[1.0,1.1,0.9]]] )
        result = teto.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor([[0.99834394]], shape=(1, 1), dtype=float32)', str(result) )

    def test__softargmax__4dims(self):
        input = tf.constant( [[[[1.0,1.1,0.9]]]] )
        result = teto.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor([[[0.99834394]]], shape=(1, 1, 1), dtype=float32)', str(result) )

    def test__softargmax__4dims_arrayed(self):
        input = tf.constant( [[[[1.0,1.1,0.9],[0.9,1.0,1.1]],[[2.0,2.1,1.9],[1.9,2.0,2.1]]]] )
        result = teto.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor(\n'
                          '[[[0.99834394 1.9983357 ]\n'
                          '  [0.99834394 1.9983357 ]]], shape=(1, 2, 2), dtype=float32)', str(result) )


if __name__ == '__main__':
    unittest.main()