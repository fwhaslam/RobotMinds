import tensorflow as tf
import unittest
from support_for_testing import *


class TestSum(unittest.TestCase):

    def test__loss_has_gradient__true(self):
        def loss_with_subtraction(input_x, output_x):
            return output_x - input_x
        result = loss_has_gradient( loss_with_subtraction )
        self.assertTrue( result )

    def test__loss_has_gradient__false(self):
        def loss_with_argmax(input_x, output_x):
            return tf.argmax( output_x - input_x )
        result = loss_has_gradient( loss_with_argmax )
        self.assertFalse( result )

    def test__loss_has_gradient__lamdba(self):
        def single_argument_loss(output_x):
            return output_x
        result = loss_has_gradient( lambda x,y: single_argument_loss(x-y) )
        self.assertTrue( result )

    def test__loss_has_gradient__shape(self):
        def loss_with_subtraction(input_x, output_x):
            return output_x - input_x
        result = loss_has_gradient( loss_with_subtraction, (2,2) )
        self.assertTrue( result )

    def test__loss_has_gradient__lamdba_shape(self):
        def single_argument_loss(output_x):
            return output_x
        result = loss_has_gradient( lambda x,y: single_argument_loss(x-y), (2,2) )
        self.assertTrue( result )

if __name__ == '__main__':
    unittest.main()