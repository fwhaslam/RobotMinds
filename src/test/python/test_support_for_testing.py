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

if __name__ == '__main__':
    unittest.main()