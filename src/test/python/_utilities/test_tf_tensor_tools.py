import sys
sys.path.append('..')
sys.path.append('../../../main/python')

import tensorflow as tf

# testing modules
import unittest

# module under test
import _utilities.tf_tensor_tools as teto
import support_for_testing as sft


class test_tf_tensor_tools(unittest.TestCase):

    def test__tensor_to_value__single(self):
        some_tensor = tf.constant( 4.5 )
        result = teto.tensor_to_value(some_tensor)
        self.assertEqual( '4.5', str(result) )

    def test__tensor_to_value__1dim(self):
        some_tensor = tf.constant( [4.5] )
        result = teto.tensor_to_value(some_tensor)
        self.assertEqual( '[4.5]', str(result) )

    def test__tensor_to_value__2dim(self):
        some_tensor = tf.constant( [[7.0],[4.5]] )
        result = teto.tensor_to_value(some_tensor)
        self.assertEqual( '[[7. ]\n'
                           ' [4.5]]', str(result) )

    def test__tensor_to_string__single_value(self):
        some_tensor = tf.constant( 4.5 )
        result = teto.tensor_to_string(some_tensor)
        self.assertEqual( '4.5', result )

    def test__tensor_to_string__one_dim(self):
        some_tensor = tf.constant( [1,2] )
        result = teto.tensor_to_string(some_tensor)
        self.assertEqual( '[1 2]', result )

    def test__tensor_to_string__two_dim(self):
        some_tensor = tf.constant( [[1,2],[3,4]] )
        result = teto.tensor_to_string(some_tensor)
        self.assertEqual( '[[1 2]\n [3 4]]', result )

    def test__tensor_assert_string__one_dim(self):
        some_tensor = tf.constant( [1,2] )
        result = teto.tensor_assert_string(some_tensor)
        self.assertEqual( '[1 2]', str(result) )

    def test__tensor_assert_string__two_dim(self):
        some_tensor = tf.constant( [[1,2],[3,4]] )
        result = teto.tensor_assert_string(some_tensor)
        self.assertEqual( '[[1 2]\n [3 4]]', str(result) )

    def test__tensor_string_formatter__zero_dim(self):
        some_tensor = tf.constant( 1 )
        result = teto.tensor_string_formatter(some_tensor)
        self.assertEqual( '1', str(result) )

    def test__tensor_string_formatter__one_dim(self):
        some_tensor = tf.constant( [1,2] )
        result = teto.tensor_string_formatter(some_tensor)
        self.assertEqual( '[1 2]', str(result) )

    def test__tensor_string_formatter__two_dim(self):
        some_tensor = tf.constant( [[1,2],[3,4]] )
        result = teto.tensor_string_formatter(some_tensor)
        self.assertEqual( '[[1 2]\n [3 4]]', str(result) )

########################################################################################################################

    def test__softargmax__1dim(self):
        input = tf.constant( [1.0,1.1,0.9] )
        result = teto.softargmax( input )
        # print('result2=',result)
        self.assertEqual( 'tf.Tensor(0.99834394, shape=(), dtype=float32)', str(result) )

    def test__softargmax__2dims(self):
        input = tf.constant( [[1.0,1.1,0.9]] )
        result = teto.softargmax( input )
        # print('result2=',result)
        self.assertEqual( 'tf.Tensor([0.99834394], shape=(1,), dtype=float32)', str(result) )

    def test__softargmax__3dims(self):
        input = tf.constant( [[[1.0,1.1,0.9]]] )
        result = teto.softargmax( input )
        # print('result2=',result)
        self.assertEqual( 'tf.Tensor([[0.99834394]], shape=(1, 1), dtype=float32)', str(result) )

    def test__softargmax__4dims(self):
        input = tf.constant( [[[[1.0,1.1,0.9]]]] )
        result = teto.softargmax( input )
        # print('result2=',result)
        self.assertEqual( 'tf.Tensor([[[0.99834394]]], shape=(1, 1, 1), dtype=float32)', str(result) )

    def test__softargmax__4dims_arrayed(self):
        input = tf.constant( [[[[1.0,1.1,0.9],[0.9,1.0,1.1]],[[2.0,2.1,1.9],[1.9,2.0,2.1]]]] )
        result = teto.softargmax( input )
        # print('result2=',result)
        self.assertEqual( 'tf.Tensor(\n'
                          '[[[0.99834394 1.9983357 ]\n'
                          '  [0.99834394 1.9983357 ]]], shape=(1, 2, 2), dtype=float32)', str(result) )


########################################################################################################################

    def test__supersoftmax__1dim(self):

        input = tf.constant( [1.0,1.1,0.9] )

        # invocation
        result = teto.supersoftmax( input )
        # print('result2=',result)

        # assertions
        self.assertEqual('(3,)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = teto.tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0], places=2 )
        self.assertAlmostEqual( 1., result[1], places=2 )
        self.assertAlmostEqual( 0., result[2], places=2 )

    def test__supersoftmax__2dims(self):

        input = tf.constant( [[1.0,1.1,0.9]] )

        # invocation
        result = teto.supersoftmax( input )
        # print('result2=',result)

        # assertions
        self.assertEqual('(1, 3)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = teto.tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0][0], places=2 )
        self.assertAlmostEqual( 1., result[0][1], places=2 )
        self.assertAlmostEqual( 0., result[0][2], places=2 )

        # self.assertEqual( 'tf.Tensor([0.99834394], shape=(1,), dtype=float32)', str(result) )

    def test__supersoftmax__3dims(self):

        input = tf.constant( [[[1.0,1.1,0.9]]] )

        # invocation
        result = teto.supersoftmax( input )
        # print('result2=',result)

        # assertions
        self.assertEqual('(1, 1, 3)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = teto.tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0][0][0], places=2 )
        self.assertAlmostEqual( 1., result[0][0][1], places=2 )
        self.assertAlmostEqual( 0., result[0][0][2], places=2 )

        # self.assertEqual( 'tf.Tensor([[0.99834394]], shape=(1, 1), dtype=float32)', str(result) )

    def test__supersoftmax__4dims(self):

        input = tf.constant( [[[[1.0,1.1,0.9]]]] )

        #invocation
        result = teto.supersoftmax( input )
        # print('result2=',result)

        # assertions
        self.assertEqual('(1, 1, 1, 3)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = teto.tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0][0][0][0], places=2 )
        self.assertAlmostEqual( 1., result[0][0][0][1], places=2 )
        self.assertAlmostEqual( 0., result[0][0][0][2], places=2 )

    def test__supersoftmax__4dims_arrayed(self):

        input = tf.constant( [[[[1.0,1.1,0.9],[0.9,1.0,1.1]],[[2.0,2.1,1.9],[1.9,2.0,2.1]]]] )

        # invocation
        result = teto.supersoftmax( input )
        # print('result2=',result)

        # assertions
        self.assertEqual('(1, 2, 2, 3)',str(result.shape))
        self.assertEqual('<dtype: \'float32\'>',str(result.dtype))

        value = teto.tensor_to_value( result )
        self.assertAlmostEqual( 0., result[0][0][0][0], places=2 )
        self.assertAlmostEqual( 1., result[0][0][0][1], places=2 )
        self.assertAlmostEqual( 0., result[0][0][0][2], places=2 )

        self.assertAlmostEqual( 0., result[0][0][1][0], places=2 )
        self.assertAlmostEqual( 0., result[0][0][1][1], places=2 )
        self.assertAlmostEqual( 1., result[0][0][1][2], places=2 )

        self.assertAlmostEqual( 0., result[0][1][0][0], places=2 )
        self.assertAlmostEqual( 1., result[0][1][0][1], places=2 )
        self.assertAlmostEqual( 0., result[0][1][0][2], places=2 )

        self.assertAlmostEqual( 0., result[0][1][1][0], places=2 )
        self.assertAlmostEqual( 0., result[0][1][1][1], places=2 )
        self.assertAlmostEqual( 1., result[0][1][1][2], places=2 )

########################################################################################################################

    def test__simple_ratio__2dims(self):

        input = 3 * [ [1.,2.,5.] ]
        sft.object_has_shape( [3,3], input )
        result = teto.simple_ratio( input )
        self.assertEqual(
            "[[0.125 0.25  0.625]\n"
            " [0.125 0.25  0.625]\n"
            " [0.125 0.25  0.625]]", teto.tensor_to_string(result) )

    def test__simple_ratio__3dims(self):

        input = 2 * [ 2 * [ [1.,0.,0.,1.] ] ]
        sft.object_has_shape( [2,2,4], input )

        # invocation
        result = teto.simple_ratio( input )

        # assertion
        self.assertEqual(
            "tf.Tensor(\n"
            "[[[0.5 0.  0.  0.5]\n"
            "  [0.5 0.  0.  0.5]]\n"
            "[73 chars]t32)", teto.tensor_to_string(result) )


########################################################################################################################

if __name__ == '__main__':
    unittest.main()