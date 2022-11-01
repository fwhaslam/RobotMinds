import sys
sys.path.append('../..')
sys.path.append('../../../../main/python')

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
import rules.v1.land_and_sea_functions as lnz


class test_land_and_sea(unittest.TestCase):

    def test__vee(self):
        self.assertEqual( 0., lnz.vee(0.) )
        self.assertEqual( 1., lnz.vee(1.) )
        self.assertEqual( 1., lnz.vee(-1.) )
        self.assertEqual( 0.5, lnz.vee(0.5) )
        self.assertEqual( 0.5, lnz.vee(-0.5) )

    def test__vee__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.vee ) )

    def test__peak(self):
        self.assertEqual( 0., lnz.peak(0.) )
        self.assertEqual( -1., lnz.peak(1.) )
        self.assertEqual( -1., lnz.peak(-1.) )
        self.assertEqual( -0.5, lnz.peak(0.5) )
        self.assertEqual( -0.5, lnz.peak(-0.5) )

    def test__peak__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.peak ) )


    def test__round_vee(self):
        def check_value( val ):
            input = tf.constant(val)
            result = lnz.round_vee( input )
            return tensor_to_value(result)
        with self.subTest('x=-1.0 => y=1.0'):
            self.assertAlmostEqual( 1.0, check_value( -1.0), places=6 )
        with self.subTest('x=-0.7 => y=0.588'):
            self.assertAlmostEqual( 0.58778525, check_value( -0.7), places=6 )
        with self.subTest('x=-0.3 => y=-5.88'):
            self.assertAlmostEqual( -0.58778525, check_value( -0.3), places=6 )
        with self.subTest('x=0.0 => y=-1.0'):
            self.assertAlmostEqual( -1.0, check_value( 0.0), places=6 )
        with self.subTest('x=0.3 => y=-0.588'):
            self.assertAlmostEqual( -0.58778525, check_value( 0.3), places=6 )
        with self.subTest('x=0.7 => y=0.588'):
            self.assertAlmostEqual( 0.58778525, check_value( 0.7), places=6 )
        with self.subTest('x=1.0 => y=1.0'):
            self.assertAlmostEqual( 1.0, check_value( 1.0), places=6 )

    def test__round_vee__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.round_vee ) )

    def test__round_peak(self):
        def check_value( val ):
            input = tf.constant(val)
            result = lnz.round_peak( input )
            return tensor_to_value(result)
        with self.subTest('x=-1.0 => y=-1.0'):
            self.assertAlmostEqual( -1.0, check_value( -1.0), places=6 )
        with self.subTest('x=-0.7 => y=-0.588'):
            self.assertAlmostEqual( -0.58778525, check_value( -0.7), places=6 )
        with self.subTest('x=-0.3 => y=0.588'):
            self.assertAlmostEqual( 0.58778525, check_value( -0.3), places=6 )
        with self.subTest('x=0.0 => y=1.0'):
            self.assertAlmostEqual( 1.0, check_value( 0.0), places=6 )
        with self.subTest('x=0.3 => y=0.588'):
            self.assertAlmostEqual( 0.58778525, check_value( 0.3), places=6 )
        with self.subTest('x=0.7 => y=-0.588'):
            self.assertAlmostEqual( -0.58778525, check_value( 0.7), places=6 )
        with self.subTest('x=1.0 => y=-1.0'):
            self.assertAlmostEqual( -1.0, check_value( 1.0) )

    def test__round_peak__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.round_peak ) )


########################################################################################################################

    def test__terrain_loss__perfect(self):

        # logits to [[1,0],[1,1]]
        input = [[[0.,1.],
                    [1.,0.]],
                   [[0.,1.],
                    [1.,0.]]]
        # batch_size=3, wide/tall=2x2, types=2
        inputs = tf.constant( [input,input,input]  )
        # tf.print("Input=",tf.shape(input))
        template = [[[0.,1.],
                  [1.,0.]],
                 [[0.,1.],
                  [1.,0.]]]
        # batch_size=3, wide/tall=2x2, types=2
        templates = tf.constant( [template,template,template]  )

        # invocation
        result = lnz.terrain_loss( templates, inputs )
        # tf.print("RESULT=",result)
        # tf.print("RESULT=",tf.get_static_value(result))

        # assertions
        self.assertEqual('(3,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        self.assertAlmostEqual( 0., result[0], places=6 )
        self.assertAlmostEqual( 0., result[1], places=6 )
        self.assertAlmostEqual( 0., result[2], places=6 )

    def test__terrain_loss__mostly_bad(self):
        # test terrain_loss

        # logits to [[1,1],[1,1]]
        input = [[[0.4,0.6],
                    [0.4,0.6]],
                   [[0.4,0.6],
                    [0.4,0.6]]]
        # batch_size=3, wide/tall=2x2, types=2
        inputs = tf.constant( [input,input,input]  )
        # tf.print("Input=",tf.shape(input))
        template = [[[0.,1.],
                    [0.,1.]],
                   [[0.,1.],
                    [0.,1.]]]
        templates = tf.constant( [template,template,template]  )


    # invocation
        result = lnz.terrain_loss( templates, inputs )
        # tf.print("RESULT=",result)

        # assertions
        self.assertEqual('(3,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        self.assertAlmostEqual( 1.6599972, result[0], places=6 )
        self.assertAlmostEqual( 1.6599972, result[1], places=6 )
        self.assertAlmostEqual( 1.6599972, result[2], places=6 )

    def test__terrain_loss__types_failure(self):
        # test terrain_loss

        # logits to [[1,1],[1,1]]
        input = [[[0.,1.],
                    [0.,1.]],
                   [[0.,1.],
                    [0.,1.]]]
        # batch_size=3, wide/tall=2x2, types=2
        inputs = tf.constant( [input,input,input]  )
        # tf.print("Input=",tf.shape(input))

        template = [[[0.,1.],
                  [0.,1.]],
                 [[0.,1.],
                  [0.,1.]]]
        templates = tf.constant( [template,template,template]  )

        # invocation
        result = lnz.terrain_loss( templates, inputs )
        # print("RESULT=",result)

        # assertions
        self.assertEqual('(3,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        self.assertAlmostEqual( 1.5, result[0], places=6 )
        self.assertAlmostEqual( 1.5, result[1], places=6 )
        self.assertAlmostEqual( 1.5, result[2], places=6 )

    def test__terrain_loss__has_gradient(self):
        self.assertTrue(sft.loss_has_gradient(lnz.terrain_loss, arg_count=2, output_shape=(2, 2, 2)))

########################################################################################################################
# remember: loss needs to be reduced, so the desired state is lowest

    def test__terrain_type_loss__on_target(self):

        # logits to [[1,0],[1,0]]
        element = [[[0.1,0.9],
                    [0.8,0.2]],
                   [[0.4,0.6],
                    [0.7,0.3]]]
        # batch_size=2, wide/tall=2x2, types=2
        input = tf.constant( [element,element]  )

        # invocation
        result = lnz.terrain_type_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0. 0.], shape=(2,), dtype=float32)", str(result) )

    def test__terrain_type_loss__max_off_target(self):

        # logits to [[1,0],[1,1]]
        element = [[[0.0,1.],
                    [0.0,1.]],
                   [[0.0,1.],
                    [0.0,1.]]]
        # batch_size2, wide/tall=2x2, types=2
        input = tf.constant( [element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_type_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0.5 0.5], shape=(2,), dtype=float32)", str(result) )

    def test__terrain_type_loss__half_off_target(self):

        # logits to [[1,1],[1,0]]
        element = [[[0.,1.],
                    [0.,1.]],
                   [[0.,1.],
                    [1.,0.]]]
        # batch_size2, wide/tall=2x2, types=2
        input = tf.constant( [element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_type_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0.25 0.25], shape=(2,), dtype=float32)", str(result) )


    def test__terrain_type_loss__mixed(self):

        # logits to [[?,0],[1,1]]
        element1 = [[[0.5,0.5],
                     [0.5,0.5]],
                    [[0.5,0.5],
                     [0.5,0.5]]]
        element2 = [[[0.9,0.1],
                     [0.8,0.2]],
                    [[0.7,0.5],
                     [0.6,0.4]]]
        # batch_size2, wide/tall=2x2, types=2
        input = tf.constant( [element1,element2]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_type_loss( input )
        # print("RESULT=",result)

        # assertions
        self.assertEqual('(2,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0], places=6 )
        self.assertAlmostEqual( 0.225, result[1], places=6 )

    def test__terrain_type_loss__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.terrain_type_loss, output_shape=(2,2,2) ) )

########################################################################################################################
# remember: loss needs to be reduced, so the desired state is lowest

    def test__terrain_certainty_loss__completely_certain(self):

        # logits to [[1,0],[1,0]]
        element = [[[0.0,1.0],
                    [1.0,0.0]],
                   [[0.0,1.0],
                    [1.0,0.0]]]
        # batch_size=2, wide/tall=2x2, types=2
        input = tf.constant( [element,element]  )

        # invocation
        result = lnz.terrain_certainty_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0. 0.], shape=(2,), dtype=float32)", str(result) )

    def test__terrain_certainty_loss__completely_uncertain(self):
        # test terrain_loss

        # logits to [[1,0],[1,1]]
        element = [[[0.5,0.5],
                    [0.5,0.5]],
                   [[0.5,0.5],
                    [0.5,0.5]]]
        # batch_size2, wide/tall=2x2, types=2
        input = tf.constant( [element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_certainty_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([1. 1.], shape=(2,), dtype=float32)", str(result) )

    def test__terrain_certainty_loss__half_uncertain(self):
        # test terrain_loss

        # logits to [[?,0],[1,1]]
        element = [[[0.75,0.25],
                    [0.25,0.75]],
                   [[0.25,0.75],
                    [0.75,0.25]]]
        # batch_size2, wide/tall=2x2, types=2
        input = tf.constant( [element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_certainty_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0.5 0.5], shape=(2,), dtype=float32)", str(result) )

    def test__terrain_certainty_loss__mixed(self):
        # test terrain_loss

        # logits to [[?,0],[1,1]]
        element1 = [[[0.0,1.0],
                    [1.0,0.0]],
                   [[0.0,1.0],
                    [1.0,0.0]]]
        element2 = [[[0.75,0.25],
                    [0.25,0.75]],
                   [[0.25,0.75],
                    [0.75,0.25]]]
        # batch_size2, wide/tall=2x2, types=2
        input = tf.constant( [element1,element2]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_certainty_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0.  0.5], shape=(2,), dtype=float32)", str(result) )

    def test__terrain_certainty_loss__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.terrain_certainty_loss, output_shape=(2,2,2) ) )

########################################################################################################################
# remember: loss needs to be reduced, so the desired state is lowest

    def test__terrain_surface_loss__no_surface(self):

        # logits to [[1,1],[1,1]]
        element = [[[0.0,1.0],
                    [0.0,1.0]],
                   [[0.0,1.0],
                    [0.0,1.0]]]
        # batch_size=3, wide/tall=2x2, types=2
        input = tf.constant( [element,element,element]  )

        # invocation
        result = lnz.terrain_surface_loss( input )
        # print("RESULT=",result)

        # assertions
        self.assertEqual('(3,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = tensor_to_value( result )
        self.assertAlmostEqual( 1., result[0], places=6 )
        self.assertAlmostEqual( 1., result[0], places=6 )
        self.assertAlmostEqual( 1., result[0], places=6 )

    def test__terrain_surface_loss__max_surface(self):

        # logits to [[1,0],[0,1]]
        element = [[[0.0,1.0],
                    [1.0,0.0]],
                   [[1.0,0.0],
                    [0.0,1.0]]]
        # batch_size=1, wide/tall=2x2, types=2
        input = tf.constant( [element,]  )

        # invocation
        result = lnz.terrain_surface_loss( input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( 'tf.Tensor([1.], shape=(1,), dtype=float32)', str(result) )

    def test__terrain_surface_loss__half_surface(self):

        # logits to [[0,1],[0,1]]
        element = [[[1.0,0.0],
                    [0.0,1.0]],
                   [[1.0,0.0],
                    [0.0,1.0]]]
        # batch_size=1, wide/tall=2x2, types=2
        input = tf.constant( [element,]  )

        # invocation
        result = lnz.terrain_surface_loss( input )
        # print("RESULT=",result)

        # assertions
        self.assertEqual('(1,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0], places=6 )

    def test__terrain_surface_loss__mixed(self):

        # logits to [[1,0],[1,1]]
        element = [[[0.1,0.9],
                    [1.0,0.0]],
                   [[0.2,0.8],
                    [0.3,0.7]]]
        # batch_size=1, wide/tall=2x2, types=2
        input = tf.constant( [element,]  )

        # invocation
        result = lnz.terrain_surface_loss( input )
        # print("RESULT=",result)

        # assertions
        self.assertEqual('(1,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))
        self.assertAlmostEqual( 0.241292, result[0], places=6 )


    def test__terrain_surface_loss__at_third(self):

        print("lnz.TERRAIN_SURFACE_GOAL=",lnz.TERRAIN_SURFACE_GOAL)
        old_value = tensor_to_value( lnz.TERRAIN_SURFACE_GOAL )
        lnz.set_terrain_surface_goal( 0.3 )

        try:
            # logits to [[1,0],[1,1]]
            element = [ [[0.,1.],[0.,1.],[0.,1.]],
                        [[1.,0.],[1.,0.],[1.,0.],],
                        [[1.,0.],[1.,0.],[1.,0.],] ]
            # batch_size=1, wide/tall=2x2, types=2
            input = tf.constant( [element,]  )

            # invocation
            result = lnz.terrain_surface_loss( input )
            # print("RESULT=",result)

            # assertions
            self.assertEqual('(1,)',str(result.shape))
            self.assertEqual('<dtype: \'float32\'>',str(result.dtype))
            self.assertAlmostEqual( 0.066667, result[0], places=6 )
        finally:
            lnz.set_terrain_surface_goal( old_value )
            # print("lnz.TERRAIN_SURFACE_GOAL=",lnz.TERRAIN_SURFACE_GOAL)


    def test__terrain_surface_loss__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.terrain_surface_loss, output_shape=(2,2,2) ) )

########################################################################################################################

    def test__image_to_template(self):

        # shape = (1,   2,2,    3)
        input = tf.constant( [[[[0,0,0],[0.25,0.25,0.25]],
                               [[0.5,0.5,0.5],[0.75,0.75,0.75]]]])

        # invocation
        result = lnz.image_to_template( input )

        # assertion
        # print('result=',result)
        self.assertEqual("""tf.Tensor(
[[[[1. 0.]
   [1. 0.]]

  [[0. 1.]
   [0. 1.]]]], shape=(1, 2, 2, 2), dtype=float32)""", str(result))


########################################################################################################################

if __name__ == '__main__':
    unittest.main()