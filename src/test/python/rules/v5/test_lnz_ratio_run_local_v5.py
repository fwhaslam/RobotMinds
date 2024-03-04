import sys
sys.path.append('../..')
sys.path.append('../../../../main/python')
sys.path.append('../../../../main/python/rules/v5')

import tensorflow as tf
import numpy as np

# local modules
import support_for_testing as sft
from _utilities.tf_tensor_tools import *

# testing modules
import unittest
import builtins
import contextlib, io

# module under test
# import rules.v3.lnz_ratio_run_local as lnz
import lnz_ratio_run_local as lnz


class test_lnz_rules_training_run_local(unittest.TestCase):

    def setUp(self):
        lnz.prepare_globals()


########################################################################################################################

    def test__feature_to_sample(self):

        feature = tf.constant( [1.,2.,3.] )

        result = lnz.feature_to_sample( feature, [4,2] )

        as_str = np.array2string( result.numpy(), threshold = np.inf)
        self.assertEqual(
            "[[1. 2.]\n"
            " [3. 0.]\n"
            " [0. 0.]\n"
            " [0. 0.]]", as_str )


    def test__feature_to_sample__two_dims(self):

        feature = tf.constant( [1.,2.,3.] )
        feature = 2 * [ feature ]

        result = lnz.feature_to_sample( feature, [4,2] )

        as_str = np.array2string( result.numpy(), threshold = np.inf)
        self.assertEqual(
            """[[[1. 2.]
  [3. 0.]
  [0. 0.]
  [0. 0.]]

 [[1. 2.]
  [3. 0.]
  [0. 0.]
  [0. 0.]]]""", as_str )

    def test__sample_to_feature(self):

        sample = tf.constant( [[1.,2.],[3.,0.],[0.,0.],[0.,0.]] )

        result = lnz.sample_to_feature( sample, feature_size=3, axis=0 )

        self.assertEqual(
            "[1. 2. 3.]", str(result.numpy()) )

    def test__sample_to_feature__two_dims(self):

        sample = tf.constant( [[[1.,2.],[3.,0.],[0.,0.],[0.,0.]],
                               [[4.,5.],[6.,0.],[0.,0.],[0.,0.]]] )

        result = lnz.sample_to_feature( sample, feature_size=3, axis=1 )

        self.assertEqual(
            "[[1. 2. 3.]\n"
            " [4. 5. 6.]]", str(result.numpy()) )


########################################################################################################################

    def test__terrain_type_feature_count(self):

        self.assertEqual( 11, lnz.terrain_type_feature_count( 1, 11 ) )
        self.assertEqual( 66, lnz.terrain_type_feature_count( 2, 11 ) )
        self.assertEqual( 286, lnz.terrain_type_feature_count( 3, 11 ) )
        self.assertEqual( 1001, lnz.terrain_type_feature_count( 4, 11 ) )

        self.assertEqual( 5, lnz.terrain_type_feature_count( 1, 5 ) )
        self.assertEqual( 6, lnz.terrain_type_feature_count( 2, 3 ) )


    def test__terrain_type_ratios__one_feature_five_steps(self):

        fill = np.empty( (5,1) )
        index = lnz.terrain_type_ratios( fill, depth=1, ratio_steps=5 )
        self.assertEqual( 5, index )
        self.assertEqual( """[[0.  ]
 [0.25]
 [0.5 ]
 [0.75]
 [1.  ]]""", str(fill))


    def test__terrain_type_ratios__two_features_three_steps(self):

        fill = np.empty( (6,2) )
        index = lnz.terrain_type_ratios( fill, depth=2, ratio_steps=3 )
        self.assertEqual( 6, index )
        self.assertEqual( """[[0.  0. ]
 [0.  0.5]
 [0.  1. ]
 [0.5 0. ]
 [0.5 0.5]
 [1.  0. ]]""", str(fill))

    def test__terrain_type_ratios__four_features_eleven_steps(self):

        fill = np.empty( (10010,4) )
        index = lnz.terrain_type_ratios( fill, depth=4, ratio_steps=11 )
        self.assertEqual( 1001, index )


    def test__append_inverse(self):

        features = tf.constant( 3 * [[ 0.1, 0.2, 0.3 ]] )
        sft.object_has_shape( [3,3], features )

        result = lnz.append_inverse( features )

        self.assertEqual( "tf.Tensor([3 6], shape=(2,), dtype=int32)", str( tf.shape(result) ) )
        self.assertEqual( "tf.Tensor([0.1 0.2 0.3 0.9 0.8 0.7], shape=(6,), dtype=float32)", str(result[0]) )
        self.assertEqual( "tf.Tensor([0.1 0.2 0.3 0.9 0.8 0.7], shape=(6,), dtype=float32)", str(result[1]) )
        self.assertEqual( "tf.Tensor([0.1 0.2 0.3 0.9 0.8 0.7], shape=(6,), dtype=float32)", str(result[2]) )


    def test__make_edges(self):

        self.assertEqual(
            "[[1. 1. 1. 1.]\n"
            " [1. 0. 0. 1.]\n"
            " [1. 0. 0. 1.]\n"
            " [1. 1. 1. 1.]]", str( lnz.make_edges(4,4) )
        )

        self.assertEqual(
            "[[1. 1. 1.]\n"
            " [1. 0. 1.]\n"
            " [1. 0. 1.]\n"
            " [1. 0. 1.]\n"
            " [1. 1. 1.]]", str( lnz.make_edges(3,5) )
        )

    def test__terrain_type_near_edge_loss(self):

        lnz.EDGES = lnz.make_edges( 4, 4 )

        y_guess = 2 * [[3,0,3,0],[0,3,0,3]]
        sft.object_has_shape( [4,4], y_guess )
        # tf.print("WORK=",y_guess)
        y_guess = tf.one_hot( y_guess, lnz.TERRAIN_TYPE_COUNT )
        sft.object_has_shape( [4,4,6], y_guess )
        # tf.print("WORK=",y_guess)

        # invocation
        result = lnz.terrain_type_near_edge_loss( 0, y_guess )

        self.assertEqual(
            "tf.Tensor(0.0, shape=(), dtype=float32)", str(result)
        )

    def test__terrain_type_near_edge_loss__has_gradient(self):
        lnz.EDGES = lnz.make_edges( 4, 4 )
        self.assertTrue(sft.loss_has_gradient(lnz.terrain_type_near_edge_loss ))


    def test__terrain_type_sticky_loss(self):

        lnz.EDGES = lnz.make_edges( 4, 4 )

        y_guess = 2 * [[3,3,0,0],[0,0,3,3]]
        sft.object_has_shape( [4,4], y_guess )
        # tf.print("WORK=",y_guess)
        y_guess = tf.one_hot( y_guess, lnz.TERRAIN_TYPE_COUNT )
        sft.object_has_shape( [4,4,6], y_guess )
        # tf.print("WORK=",y_guess)

        # y_guess = 8 * [ y_guess ]
        # sft.object_has_shape( [8,4,4,6], y_guess )

        # invocation
        result = lnz.terrain_type_sticky_loss( 0, y_guess )

        self.assertEqual(
            "tf.Tensor(0.0, shape=(), dtype=float32)", str(result)
        )

    def test__terrain_type_sticky_loss__has_gradient(self):
        self.assertTrue(sft.loss_has_gradient(lnz.terrain_type_sticky_loss, arg_count=2 ))

########################################################################################################################

if __name__ == '__main__':
    unittest.main()