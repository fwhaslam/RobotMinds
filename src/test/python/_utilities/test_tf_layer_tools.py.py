import sys
sys.path.append('..')
sys.path.append('../../../main/python')

import tensorflow as tf

# testing modules
import unittest

# module under test
import _utilities.tf_layer_tools as tflt


class test_tf_layer_tools(unittest.TestCase):


    def test__ImageResize(self):

        layer = tflt.ImageResize(3,2)

        # shape = ( 1,  2,2,  3 )
        input = tf.constant( [ [[[1,0,1],[2,0,2]],[[3,0,3],[4,0,4]]]] )

        # invocation
        result = layer( input )
        # print('result2=',result)

        # assertions
        self.assertEqual('(1, 4, 6, 3)',str(result.shape))
        self.assertEqual('<dtype: \'int32\'>',str(result.dtype))
        self.assertEqual("""tf.Tensor(
[[1 0 1]
 [1 0 1]
 [1 0 1]
 [2 0 2]
 [2 0 2]
 [2 0 2]], shape=(6, 3), dtype=int32)""",str(result[0][0]))



    def test__cross_shift(self):

        # shape = ( 1,  2,2,  3 )
        input = tf.cast( tf.constant( [ [[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]]] ), tf.float32 )

        # #invocation
        result = tflt.cross_shift( input )
        print('result=',result)

        # # assertions
        self.assertEqual("""tf.Tensor(
[[[[1. 1. 1.]
   [0. 0. 0.]
   [2. 2. 2.]]

  [[0. 0. 0.]
   [0. 0. 0.]
   [0. 0. 0.]]

  [[3. 3. 3.]
   [0. 0. 0.]
   [4. 4. 4.]]]], shape=(1, 3, 3, 3), dtype=float32)""",str(result))


if __name__ == '__main__':
    unittest.main()