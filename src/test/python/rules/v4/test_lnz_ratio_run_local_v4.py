import sys
sys.path.append('../..')
sys.path.append('../../../../main/python')
sys.path.append('../../../../main/python/rules/v4')

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


class test_lnz_ratio_run_local(unittest.TestCase):

    def test__terrain_loss__min_diff(self):

        lnz.prepare_globals()

        y_goal = np.full( [2,3,3,4], 1. )
        sft.object_has_shape( [2,3,3,4], y_goal )
        y_result = np.full( [2,3,3,4], 1. )
        sft.object_has_shape( [2,3,3,4], y_result )

        # invocation
        result = lnz.terrain_loss( y_goal, y_result )
        # tf.print("RESULT=",result)
        # tf.print("RESULT=",tf.get_static_value(result))

        # assertions :: batch of 4 results
        self.assertEqual('tf.Tensor([0.75 0.75], shape=(2,), dtype=float64)',str(result))


    def test__terrain_loss__maxdiff(self):

        lnz.prepare_globals()

        y_goal = np.full( [2,3,3,4], 1. )
        sft.object_has_shape( [2,3,3,4], y_goal )
        y_result = np.full( [2,3,3,4], 0. )
        sft.object_has_shape( [2,3,3,4], y_result )

        # invocation
        result = lnz.terrain_loss( y_goal, y_result )
        # tf.print("RESULT=",result)
        # tf.print("RESULT=",tf.get_static_value(result))

        # assertions :: batch of 4 results
        self.assertEqual('tf.Tensor([1.75 1.75], shape=(2,), dtype=float64)',str(result))


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


########################################################################################################################

if __name__ == '__main__':
    unittest.main()