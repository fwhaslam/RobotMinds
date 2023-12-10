import sys
sys.path.append('..')
sys.path.append('../../../main/python')

import tensorflow as tf

# local modules
import support_for_testing as sft
import _utilities.tf_loss_tools as lsto
import _utilities.tf_tensor_tools as teto

# testing modules
import unittest


class test_tf_loss_tools(unittest.TestCase):

    def test__valley(self):
        self.assertEqual( 0., lsto.valley(0.) )
        self.assertEqual( 1., lsto.valley(1.) )
        self.assertEqual( 1., lsto.valley(-1.) )
        self.assertEqual( 0.5, lsto.valley(0.5) )
        self.assertEqual( 0.5, lsto.valley(-0.5) )

    def test__valley__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lsto.valley ) )

    def test__peak(self):
        self.assertEqual( 0., lsto.peak(0.) )
        self.assertEqual( -1., lsto.peak(1.) )
        self.assertEqual( -1., lsto.peak(-1.) )
        self.assertEqual( -0.5, lsto.peak(0.5) )
        self.assertEqual( -0.5, lsto.peak(-0.5) )

    def test__peak__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lsto.peak ) )


    def test__round_valley(self):
        def check_value( val ):
            input = tf.constant(val)
            result = lsto.round_valley( input )
            return teto.tensor_to_value(result)
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

    def test__round_valley__has_gradient(self):
        self.assertTrue( sft.loss_has_gradient( lsto.round_valley ) )

    def test__round_peak(self):
        def check_value( val ):
            input = tf.constant(val)
            result = lsto.round_peak( input )
            return teto.tensor_to_value(result)
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
        self.assertTrue( sft.loss_has_gradient( lsto.round_peak ) )

########################################################################################################################

if __name__ == '__main__':
    unittest.main()