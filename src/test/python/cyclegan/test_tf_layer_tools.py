import sys
sys.path.append('..')
sys.path.append('../../../main/python')
sys.path.append('../../../main/python/rules')

import support_for_testing as sft
import cycle_gan.tf_layer_tools as lato

import tensorflow as tf
import numpy as np

import unittest
import builtins
import contextlib, io

class test_land_and_sea(unittest.TestCase):

    def test__resizing_layer__4dims_arrayed(self):
        input = tf.constant( [[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]] )
        result = lato.resizing_layer(2)(input)
        tf.print('result=',result)
        self.assertEqual( """tf.Tensor(""",str(result))

    def test__ImageResize__4dims(self):
        input = tf.constant( [[[[1.0,1.1],[0.9,1.0]],[[2.0,2.1],[1.9,2.0]]]] )
        result = lato.ImageResize(2,2)(input)
        tf.print('result=',result)
        self.assertEqual( """tf.Tensor(
[[[[ 1  2  3]
   [ 1  2  3]
   [ 4  5  6]
   [ 4  5  6]]

  [[ 1  2  3]
   [ 1  2  3]
   [ 4  5  6]
   [ 4  5  6]]

  [[ 7  8  9]
   [ 7  8  9]
   [10 11 12]
   [10 11 12]]

  [[ 7  8  9]
   [ 7  8  9]
   [10 11 12]
   [10 11 12]]]], shape=(1, 4, 4, 3), dtype=int32)""", str(result) )

    ########################################################################################################################

    def test__softargmax__1dim(self):
        input = tf.constant( [1.0,1.1,0.9] )
        result = lato.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor(0.99834394, shape=(), dtype=float32)', str(result) )

    def test__softargmax__2dims(self):
        input = tf.constant( [[1.0,1.1,0.9]] )
        result = lato.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor([0.99834394], shape=(1,), dtype=float32)', str(result) )

    def test__softargmax__3dims(self):
        input = tf.constant( [[[1.0,1.1,0.9]]] )
        result = lato.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor([[0.99834394]], shape=(1, 1), dtype=float32)', str(result) )

    def test__softargmax__4dims(self):
        input = tf.constant( [[[[1.0,1.1,0.9]]]] )
        result = lato.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor([[[0.99834394]]], shape=(1, 1, 1), dtype=float32)', str(result) )

    def test__softargmax__4dims_arrayed(self):
        input = tf.constant( [[[[1.0,1.1,0.9],[0.9,1.0,1.1]],[[2.0,2.1,1.9],[1.9,2.0,2.1]]]] )
        result = lato.softargmax( input )
        print('result2=',result)
        self.assertEqual( 'tf.Tensor(\n'
                          '[[[0.99834394 1.9983357 ]\n'
                          '  [0.99834394 1.9983357 ]]], shape=(1, 2, 2), dtype=float32)', str(result) )


########################################################################################################################

if __name__ == '__main__':
    unittest.main()