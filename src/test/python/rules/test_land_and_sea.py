import sys
sys.path.append('..')
sys.path.append('../../../main/python')
sys.path.append('../../../main/python/rules')

import support_for_testing as sft
import land_and_sea_functions as lnz

# import land_and_sea as lnz
import tensorflow as tf
import numpy as np

import unittest
import builtins
import contextlib, io

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
            # input = tf.constant( (val,) )
            input = tf.constant(val)
            result = lnz.round_vee( input )
            return str(result)
        # for ix in range( 0, 11 ):
        #     vx = ix * 0.2
        #     print('{} =>'.format(vx),lnz.round_peak(vx))
        with self.subTest():
            self.assertEqual( 'tf.Tensor(1.0, shape=(), dtype=float32)', check_value( -1.0) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(0.58778524, shape=(), dtype=float32)', check_value( -0.7) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-0.58778524, shape=(), dtype=float32)', check_value( -0.3) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-1.0, shape=(), dtype=float32)', check_value( 0.0) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-0.58778524, shape=(), dtype=float32)', check_value( 0.3) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(0.58778554, shape=(), dtype=float32)', check_value( 0.7) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(1.0, shape=(), dtype=float32)', check_value( 1.0) )

    def test__round_vee__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.round_vee ) )

    def test__round_peak(self):
        def check_value( val ):
            # input = tf.constant( (val,) )
            input = tf.constant(val)
            result = lnz.round_peak( input )
            return str(result)
        # for ix in range( 0, 11 ):
        #     vx = ix * 0.2
        #     print('{} =>'.format(vx),lnz.round_peak(vx))
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-1.0, shape=(), dtype=float32)', check_value( -1.0) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-0.58778524, shape=(), dtype=float32)', check_value( -0.7) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(0.58778524, shape=(), dtype=float32)', check_value( -0.3) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(1.0, shape=(), dtype=float32)', check_value( 0.0) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(0.58778524, shape=(), dtype=float32)', check_value( 0.3) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-0.58778554, shape=(), dtype=float32)', check_value( 0.7) )
        with self.subTest():
            self.assertEqual( 'tf.Tensor(-1.0, shape=(), dtype=float32)', check_value( 1.0) )

    def test__round_peak__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.round_peak ) )


########################################################################################################################

    def test__terrain_loss__perfect(self):

        # logits to [[1,0],[1,1]]
        element = [[[0.,1.],
                    [1.,0.]],
                   [[0.,1.],
                    [1.,0.]]]
        # batch_size=3, wide/tall=2x2, types=2
        input = tf.constant( [element,element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_loss( None, input )
        # tf.print("RESULT=",result)
        # tf.print("RESULT=",tf.get_static_value(result))

        # assertions
        # self.assertTrue( np.array_equal( [0.01, 0.01, 0.01], tf.get_static_value(result) ) )
        self.assertEqual( "tf.Tensor([0. 0. 0.], shape=(3,), dtype=float32)", str(result) )

    def test__terrain_loss__mostly_bad(self):
        # test terrain_loss

        # logits to [[1,0],[1,0]]
        element = [[[0.4,0.6],
                    [0.4,0.6]],
                   [[0.4,0.6],
                    [0.4,0.6]]]
        # batch_size=3, wide/tall=2x2, types=2
        input = tf.constant( [element,element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_loss( None, input )
        # tf.print("RESULT=",result)

        # assertions
        # self.assertTrue( np.array_equal( [0,0,0], tf.get_static_value(result) ) )
        self.assertEqual( "tf.Tensor([2. 2. 2.], shape=(3,), dtype=float32)", str(result) )

    def test__terrain_loss__types_failure(self):
        # test terrain_loss

        # logits to [[?,0],[1,1]]
        element = [[[0.,1.],
                    [0.,1.]],
                   [[0.,1.],
                    [0.,1.]]]
        # batch_size=3, wide/tall=2x2, types=2
        input = tf.constant( [element,element,element]  )
        # tf.print("Input=",tf.shape(input))

        # invocation
        result = lnz.terrain_loss( None, input )
        # print("RESULT=",result)

        # assertions
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([2. 2. 2.], shape=(3,), dtype=float32)", str(result) )

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
        self.assertEqual( "tf.Tensor([1. 1.], shape=(2,), dtype=float32)", str(result) )

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
        self.assertEqual( "tf.Tensor([0.5 0.5], shape=(2,), dtype=float32)", str(result) )

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
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( "tf.Tensor([0.   0.45], shape=(2,), dtype=float32)", str(result) )

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
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( 'tf.Tensor([1. 1. 1.], shape=(3,), dtype=float32)', str(result) )

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
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( 'tf.Tensor([0.], shape=(1,), dtype=float32)', str(result) )

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
        # self.assertEqual( 0., tf.get_static_value( result ) )
        self.assertEqual( 'tf.Tensor([0.24129224], shape=(1,), dtype=float32)', str(result) )

    def test__terrain_surface_loss__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lnz.terrain_surface_loss, output_shape=(2,2,2) ) )

########################################################################################################################

if __name__ == '__main__':
    unittest.main()