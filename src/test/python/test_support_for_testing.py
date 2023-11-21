import sys
sys.path.append('../../main/python')

import tensorflow as tf

# testing modules
import unittest

# module under test
import support_for_testing as sft


class test_support_for_testing(unittest.TestCase):

    def test__loss_has_gradient__true(self):
        def loss_with_subtraction(input_x, output_x):
            return output_x - input_x
        result = sft.loss_has_gradient( loss_with_subtraction, arg_count=2 )
        self.assertTrue( result )

    def test__loss_has_gradient__false(self):
        def loss_with_argmax(input_x, output_x):
            return tf.argmax( output_x - input_x )
        result = sft.loss_has_gradient( loss_with_argmax, arg_count=2 )
        self.assertFalse( result )

    def test__loss_has_gradient__single_arg(self):
        def single_argument_loss(output_x):
            return output_x
        result = sft.loss_has_gradient( single_argument_loss )
        self.assertTrue( result )

    def test__loss_has_gradient__shape(self):
        def loss_with_subtraction(input_x, output_x):
            return output_x - input_x
        result = sft.loss_has_gradient( loss_with_subtraction, arg_count=2, output_shape=(2,2) )
        self.assertTrue( result )


    def test__object_has_shape(self):

        sft.object_has_shape( [0], tf.constant( [] ) )

        sft.object_has_shape( [3], tf.constant( [1,2,3] ) )

        sft.object_has_shape( [2,3], tf.constant( 2 * [[1,2,3]] ) )

        sft.object_has_shape( [5,2,3], tf.constant( 5 * [ 2 * [[1,2,3]] ] ) )


    def test__object_to_string(self):

        object = tf.constant( 2 * [[1,2,3]] )
        sft.object_has_shape(  [2,3], object )

        result = sft.object_to_string( object )

        self.assertEqual(
            "tf.Tensor(\n[[1 2 3]\n [1 2 3]], shape=(2, 3), dtype=int32)\n", result )


    def test__object_to_string__big(self):

        object = tf.constant( 5 * [[0,1,2,3,4,5,6,7,8,9]] )
        sft.object_has_shape(  [5,10], object )

        result = sft.object_to_string( object )

        self.assertEqual(
            "tf.Tensor(\n"
            "[[0 1 2 3 4 5 6 7 8 9]\n"
            " [0 1 2 3 4 5 6 7 8 9]\n"
            " [0 1 2 3 4 5 6 7 8 9]\n"
            " [0 1 2 3 4 5 6 7 8 9]\n"
            " [0 1 2 3 4 5 6 7 8 9]], shape=(5, 10), dtype=int32)", result )


if __name__ == '__main__':
    unittest.main()