import sys
sys.path.append('..')
sys.path.append('../../../main/python')

import tensorflow as tf

# local modules
import support_for_testing as sft

# testing modules
import unittest
import builtins
import contextlib, io

# module under test
import prandom.pseudo_random_tools as psrt


########################################################################################################################


class test_pseudo_random_tools(unittest.TestCase):

    def test__is_prime(self):

        self.assertFalse( psrt.is_prime(0) )
        self.assertFalse( psrt.is_prime(1) )
        self.assertTrue( psrt.is_prime(2) )
        self.assertTrue( psrt.is_prime(3) )
        self.assertFalse( psrt.is_prime(4) )

        self.assertTrue( psrt.is_prime(5) )
        self.assertFalse( psrt.is_prime(6) )
        self.assertTrue( psrt.is_prime(7) )
        self.assertFalse( psrt.is_prime(8) )
        self.assertFalse( psrt.is_prime(9) )

        self.assertFalse( psrt.is_prime(10) )
        self.assertTrue( psrt.is_prime(11) )
        self.assertFalse( psrt.is_prime(12) )
        self.assertTrue( psrt.is_prime(13) )
        self.assertFalse( psrt.is_prime(14) )

        self.assertFalse( psrt.is_prime(81) )
        self.assertTrue( psrt.is_prime(83) )
        self.assertTrue( psrt.is_prime(127) )
        self.assertFalse( psrt.is_prime(129) )

        self.assertFalse( psrt.is_prime(437) )
        self.assertTrue( psrt.is_prime(439) )
        self.assertFalse( psrt.is_prime(1000001) )
        self.assertTrue( psrt.is_prime(1000003) )

        return

    def test__PRNG(self):

        mult = 29
        modu = 887

        # invocation
        result = psrt.SimplePseudoRandomGenerator(0, mult, modu)

        # assertion
        self.assertEqual( modu, result.count_set_size() )

        return

    def test__generate_examples__default(self):

        prng = psrt.SimplePseudoRandomGenerator(0, 29, 887)

        # invocation
        result = psrt.generate_examples( 10, prng, 3 )
        # print('examples[0].shape=',tf.shape(examples[0]))
        # print('examples[1].shape=',tf.shape(examples[1]))
        # print('examples[0]=',examples[0])
        # print('examples[1]=',examples[1])

        #assertions
        data = result[0]
        labels = result[1]

        # data[1] is last values from data[0] with label[0] appended
        self.assertEqual( '[[0.50632911]\n [0.98849252]\n [0.40161105]]', str(data[0]) )
        self.assertEqual( '[0.76294591]', str(labels[0]) )
        self.assertEqual( '[[0.98849252]\n [0.40161105]\n [0.76294591]]', str(data[1]) )
        self.assertEqual( '[0.00690449]', str(labels[1]) )
        self.assertEqual( '[[0.0552359 ]\n [0.25316456]\n [0.18527043]]', str(data[9]) )
        self.assertEqual( '[0.84810127]', str(labels[9]) )

        return

    def test__generate_examples__normalized(self):

        # hmmm, normalize to max value? how ...

        prng = psrt.SimplePseudoRandomGenerator(0, 29, 887)

        # invocation
        result = psrt.generate_examples( 10, prng, 3, norm=prng.max() )

        #assertions
        data = result[0]
        labels = result[1]

        # data[1] is last values from data[0] with label[0] appended
        self.assertEqual( '[[0.49605411]\n [0.96843292]\n [0.3934611 ]]', str(data[0]) )
        self.assertEqual( '[0.74746336]', str(labels[0]) )
        self.assertEqual( '[[0.96843292]\n [0.3934611 ]\n [0.74746336]]', str(data[1]) )
        self.assertEqual( '[0.00676437]', str(labels[1]) )
        self.assertEqual( '[[0.05411499]\n [0.24802706]\n [0.18151071]]', str(data[9]) )
        self.assertEqual( '[0.83089064]', str(labels[9]) )

        return

    def test__generate_examples__notNormalized(self):

        numgen = psrt.SimplePseudoRandomGenerator(0, 29, 887)

        # invocation
        result = psrt.generate_examples( 10, numgen, 3, norm=1. )

        #assertions
        data = result[0]
        labels = result[1]

        # data[1] is last values from data[0] with label[0] appended
        self.assertEqual( '[[440.]\n [859.]\n [349.]]', str(data[0]) )
        self.assertEqual( '[663.]', str(labels[0]) )
        self.assertEqual( '[[859.]\n [349.]\n [663.]]', str(data[1]) )
        self.assertEqual( '[6.]', str(labels[1]) )
        self.assertEqual( '[[ 48.]\n [220.]\n [161.]]', str(data[9]) )
        self.assertEqual( '[737.]', str(labels[9]) )

        return

    def test__meta__findSomePrimesForPRNG(self):

        # 13/23 => 23 values
        # 17/23 => 23 values
        # 7/29 => 29 values
        # 13/29 => 29 values
        # 13/557 =  557
        # 17/683 =  683
        # 29/887 =  887

        mult = 29
        for ix in range (500,1000):
            if not psrt.is_prime(ix): continue
            print( f'{mult}/{ix} = ', psrt.SimplePseudoRandomGenerator(0, mult, ix).count_set_size())

        return

    def test__Fibonacci(self):

        result = psrt.Fibonacci()

        self.assertEqual( 0, next(result) )
        self.assertEqual( 1, next(result) )
        self.assertEqual( 1, next(result) )
        self.assertEqual( 2, next(result) )
        self.assertEqual( 3, next(result) )

        self.assertEqual( 5, next(result) )
        self.assertEqual( 8, next(result) )
        self.assertEqual( 13, next(result) )
        self.assertEqual( 21, next(result) )
        self.assertEqual( 34, next(result) )

        self.assertEqual( 55, next(result) )
        self.assertEqual( 89, next(result) )
        self.assertEqual( 144, next(result) )
        self.assertEqual( 233, next(result) )
        self.assertEqual( 377, next(result) )

        self.assertEqual( 610, next(result) )
        self.assertEqual( 987, next(result) )
        self.assertEqual( 1597, next(result) )
        self.assertEqual( 2584, next(result) )
        self.assertEqual( 4181, next(result) )

    def test__generate_examples__withFibonacci(self):

        numgen = psrt.Fibonacci()

        # invocation
        result = psrt.generate_examples( 10, numgen, 2, norm=1. )

        #assertions
        data = result[0]
        labels = result[1]

        # data[1] is last values from data[0] with label[0] appended
        self.assertEqual( '[[0.]\n [1.]]', str(data[0]) )
        self.assertEqual( '[1.]', str(labels[0]) )
        self.assertEqual( '[[1.]\n [1.]]', str(data[1]) )
        self.assertEqual( '[2.]', str(labels[1]) )
        self.assertEqual( '[[34.]\n [55.]]', str(data[9]) )
        self.assertEqual( '[89.]', str(labels[9]) )

        return


    def test_BitsToNumber_loRange(self):

        # shape = ( bs=8, 3 )
        inputs = tf.cast( [[-1,-1,-1],
                           [+1,-1,-1],
                           [-1,+1,-1],
                           [+1,+1,-1],
                           [-1,-1,+1],
                           [+1,-1,+1],
                           [-1,+1,+1],
                           [+1,+1,+1]], tf.float32 )

        # invocation
        result = psrt.BitsToNumber(-3,0)

        # assertions
        output = result( inputs )

        self.assertEqual( 'tf.Tensor([0.    0.125 0.25  0.375 0.5   0.625 0.75  0.875], shape=(8,), dtype=float32)', str(output) )

    def test_BitsToNumber_hiRange(self):

        # shape = ( bs=8, 3 )
        inputs = tf.cast( [[-1,-1,-1],
                           [+1,-1,-1],
                           [-1,+1,-1],
                           [+1,+1,-1],
                           [-1,-1,+1],
                           [+1,-1,+1],
                           [-1,+1,+1],
                           [+1,+1,+1]], tf.float32 )

        # slightly OFF from one, should resolve to one anyway
        # inputs = inputs * .7

        # invocation
        result = psrt.BitsToNumber(0,3)

        # assertions
        output = result( inputs )

        self.assertEqual( 'tf.Tensor([0. 1. 2. 3. 4. 5. 6. 7.], shape=(8,), dtype=float32)', str(output) )

    def test_BitsToNumber_splitRange(self):

        # shape = ( bs=8, 3 )
        inputs = tf.cast( [[-1,-1,-1],
                           [+1,-1,-1],
                           [-1,+1,-1],
                           [+1,+1,-1],
                           [-1,-1,+1],
                           [+1,-1,+1],
                           [-1,+1,+1],
                           [+1,+1,+1]], tf.float32 )
        print('inputs.shape=',inputs.shape)

        # slightly OFF from one, should resolve to one anyway
        # inputs = inputs * .7

        # invocation
        result = psrt.BitsToNumber(-1,2)

        # assertions
        output = result( inputs )

        self.assertEqual( 'tf.Tensor([0.  0.5 1.  1.5 2.  2.5 3.  3.5], shape=(8,), dtype=float32)', str(output) )
        return

########################################################################################################################

    def test_NumberToBits_loRange(self):

        # shape = ( bs=8, 1 )
        inputs = tf.cast( [[0],[1],[2],[3],[4],[5],[6],[7]], tf.float32 ) / 8.
        print('inputs.shape=',inputs.shape)

        # invocation
        result = psrt.NumberToBits(-3,0)

        # assertions
        output = result( inputs )

        self.assertEqual( """tf.Tensor(
[[[-1. -1. -1.]]

 [[ 1. -1. -1.]]

 [[-1.  1. -1.]]

 [[ 1.  1. -1.]]

 [[-1. -1.  1.]]

 [[ 1. -1.  1.]]

 [[-1.  1.  1.]]

 [[ 1.  1.  1.]]], shape=(8, 1, 3), dtype=float32)""", str(output) )
        return

    def test_NumberToBits_hiRange(self):

        # shape = ( bs=8, 1 )
        inputs = tf.cast( [[0],[1],[2],[3],[4],[5],[6],[7]], tf.float32 )
        print('inputs.shape=',inputs.shape)

        # invocation
        result = psrt.NumberToBits(0,3)

        # assertions
        output = result( inputs )

        self.assertEqual( """tf.Tensor(
[[[-1. -1. -1.]]

 [[ 1. -1. -1.]]

 [[-1.  1. -1.]]

 [[ 1.  1. -1.]]

 [[-1. -1.  1.]]

 [[ 1. -1.  1.]]

 [[-1.  1.  1.]]

 [[ 1.  1.  1.]]], shape=(8, 1, 3), dtype=float32)""", str(output) )
        return

    def test_NumberToBits_splitRange(self):

        # shape = ( bs=8, 1 )
        inputs = tf.cast( [[0],[1],[2],[3],[4],[5],[6],[7]], tf.float32 ) / 2.
        print('inputs.shape=',inputs.shape)

        # invocation
        result = psrt.NumberToBits(-1,2)

        # assertions
        output = result( inputs )

        self.assertEqual( """tf.Tensor(
[[[-1. -1. -1.]]

 [[ 1. -1. -1.]]

 [[-1.  1. -1.]]

 [[ 1.  1. -1.]]

 [[-1. -1.  1.]]

 [[ 1. -1.  1.]]

 [[-1.  1.  1.]]

 [[ 1.  1.  1.]]], shape=(8, 1, 3), dtype=float32)""", str(output) )
        return


########################################################################################################################

if 'unittest.util' in __import__('sys').modules:
    # Show full diff in self.assertEqual.
    __import__('sys').modules['unittest.util']._MAX_LENGTH = 999999999

if __name__ == '__main__':
    unittest.main()