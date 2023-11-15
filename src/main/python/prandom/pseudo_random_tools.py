#
#   Pseudo Random Number Generators
#

import time
import math
import numpy as np
import tensorflow as tf

def is_prime( value:int ):

    if (value<2): return False
    if (value<4): return True
    if ((value%2)==0): return False

    limit = int( math.sqrt( value ) )
    for modu in range( 3, 1+limit, 2 ):
        if (value%modu)==0: return False

    return True


class SimplePseudoRandomGenerator:

    def __init__(self,seed=None,mult=None,modu=None):
        self.seed = seed if not seed is None else int( 1000. * time.time() )
        self.mult = mult if not mult is None else 15485863
        self.modu = modu if not modu is None else 2038074743
        return

    def __iter__(self):
        return self

    def __next__(self):
        self.seed = self.seed + 1
        a = self.seed * self.mult
        return (a * a * a) % self.modu

    def max(self):
        return self.modu

    def count_set_size( self ):
        r"""Used for testing."""
        found = set()
        while True:
            value = next(self)
            if value in found:
                return len(found)
            else:
                found.add( value )
        return

class Fibonacci:

    def __init__(self,first=-1,second=1):
        self.num = first,second
        return

    def __iter__(self):
        return self

    def __next__(self):
        sum = self.num[0] + self.num[1]
        self.num = self.num[1],sum
        return sum


def generate_examples( examples, numgen, time_steps, norm=0. ):
    r"""Data shape, data=(examples,time_steps,1), labels=(examples,1).
    max = 0 means use highest value to cap at range [0,1].
        = non-zero means div all digits by the 'norm'. """

    # array of (time_steps) values
    data = [ next(numgen) for ix in range(0,time_steps ) ]
    limit = max( data[0], data[1] )
    label = next(numgen)

    # build array of examples at the index
    datas = np.empty( ( examples, time_steps, 1) )
    labels = np.empty( ( examples, 1 ) )

    for ex in range(examples):
        for ix in range(time_steps):
            datas[ex][ix] = data[ix]
        labels[ex] = [ label ]
        limit = max( limit, label )
        data = data[1:] + [label]   # skip first, append next
        label = next(numgen)

    # print('norm=',norm)
    # print('limit=',limit)
    # normalize data based on 'norm' value
    if norm==0.:
        datas = datas / float(limit)
        labels = labels/ float(limit)
    elif norm!=1.:
        datas = datas / float(norm)
        labels = labels/ float(norm)

    return datas, labels


def xorshift128():
    '''xorshift
    https://ja.wikipedia.org/wiki/Xorshift
    '''

    x = 123456789
    y = 362436069
    z = 521288629
    w = 88675123

    def _random():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
        x, y, z = y, z, w
        w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        return w

    return _random


class BitsToNumber(tf.keras.layers.Layer):
    r"""Takes N channels of input, treats then as ever increasing binary values.
    The lowest value is the first element of the channel, and the highest is the last.
    Lowest value is 2 ^ lo_bit_power, highest is 2^(hi_bit_power-1).
    If you set lo=-16 and hi=0, then the value range is between zero and one.
    If you set lo=0 and hi=16, then the value range is between zero and 16k-1"""

    def __init__(self, lo_bit_power, hi_bit_power):
        super(BitsToNumber, self).__init__()
        self.lo = lo_bit_power
        self.hi = hi_bit_power
        self.bit_range = self.hi - self.lo
        if self.bit_range<1:
            raise Exception('Invalid bit range.  hi_bit_power must exceed lo_bit_power.')
        if self.bit_range>=32:
            raise Exception('Invalid bit range.  hi_bit_power cannot be more that 31 bits higher than lo_bit_power.')
        channel_values = [ 2.**ix for ix in range(lo_bit_power,hi_bit_power) ]
        print('channel_values=',channel_values)
        self.values = tf.constant( channel_values )
        return

    def build(self,other):
        tf.print('BitsToNumber.build/other=',other)
        return

    @tf.function
    def call(self, inputs):
        r"""Expects input in shape (bs, channels), with values in the [-1,+1] range."""

        # tf.print('inputs=',inputs)
        x = ( 1 + inputs ) / 2.     # scale from [-1,+1] range to [0,+1) range
        return tf.reduce_sum( x * self.values, axis=-1 )

class NumberToBits(tf.keras.layers.Layer):
    r"""Takes N channels of input, treats then as ever decreasing binary values.
    The lowest value is the first element of the channel, and the highest is the last.
    Lowest value is 2 ^ lo_bit_power, highest is 2^hi_bit_power-1.
    If you set lo=-16 and hi=0, then the value range is between zero and one.
    If you set lo=0 and hi=16, then the value range is between zero and 16k-1"""

    def __init__(self, lo_bit_power, hi_bit_power):
        super(NumberToBits, self).__init__()
        self.lo = lo_bit_power
        self.hi = hi_bit_power
        self.bit_range = self.hi - self.lo
        if self.bit_range<1:
            raise Exception('Invalid bit range.  hi_bit_power must exceed lo_bit_power.')
        if self.bit_range>=32:
            raise Exception('Invalid bit range.  hi_bit_power cannot be more that 31 bits higher than lo_bit_power.')
        return

    def build(self,other):
        tf.print('NumberToBits.build/other=',other)
        return

    @tf.function
    def call(self, inputs):
        r"""Expects input in shape (bs, channels), with values in the [-1,+1] range."""

        # tf.print('NumberToBits.inputs=',inputs)

        # shift to integer above zero
        work = tf.cast( inputs / (2.**self.lo), tf.int32 )
        # tf.print('shifted=',work)

        # as bits
        work = tf.expand_dims(work,1)
        # tf.print('expanded=',work)
        work = tf.bitwise.right_shift(work, tf.range(self.bit_range))
        # tf.print('shifted=',work)

        # create in [0,1] range
        work = tf.math.floormod(work, 2)
        # tf.print('bitted=',work)

        # shift to [-1,+1] range
        return tf.cast( work, tf.float32 ) * 2. - 1.

