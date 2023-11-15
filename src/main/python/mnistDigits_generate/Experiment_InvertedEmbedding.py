#
#   Playing with the embedding weights from 56_NeuralMachineTranslationWithAttention.py
#
#   Can we create an 'inverse' to use for end translation?
#
#   what is the determinant if we add a bunch of zero rows?  tf.linalg.det
#   can we create some kind of inverse?  tf.linalg.inv
#

import tensorflow as tf

embedding = tf.constant([[-0.8959618 ,  2.2839823 , -1.9336246 , -0.01028966],
                         [ 1.2777444 , -1.6295084 ,  0.8190948 ,  0.59062874],
                         [-1.8145654 , -0.9967288 , -0.51381916,  1.9750237 ],
                         [-0.8401218 ,  0.3790238 , -1.071506  ,  1.0719615 ],
                         [ 1.1598861 , -0.8027018 ,  2.1503801 , -1.4213352 ],
                         [ 0.20158052,  0.96800387, -0.7521279 , -0.242653  ],
                         [-1.1466305 , -2.322513  ,  1.1108656 ,  1.0629116 ],
                         [ 1.5732962 ,  1.444018  , -0.18647921, -2.0115876 ],
                         [-0.25006676, -0.43794042,  0.19088256,  0.33338353],
                         [ 0.7257275 ,  0.22434635,  0.71969056, -1.1542754 ]])
print("\nembedding=",embedding)

onehot_four = tf.one_hot( 4, 10 )  #tf.constant( [0.,0.,0.,0.,1., 0.,0.,0.,0.,0.] )
print("\nonehot_four=",onehot_four)

symbolic_four = tf.linalg.matvec( embedding, onehot_four, transpose_a=True )
print("\nsymbolic_four=", symbolic_four )


append_ones = tf.ones( (10,6) )
print("\nappend_ones=",append_ones)


original_squared = tf.concat( [embedding, append_ones], 1 )
print("\nappend 1s for square shape=",original_squared)


WIDE = 10
SIZE = WIDE*WIDE
SHAKE_RANGE = 5.e-4     # 5.e-6

work = tf.Variable( original_squared )
trycount = 0

determinant = tf.linalg.det( work )
print("\ndet=",determinant)

# NOTES: det < 5.e-23 produces reliable Identity from modified
#        SR = 5.e-6 produces determinant after about 10 tries
while tf.math.abs(determinant) < 5.e-18:    #  < 5.e-23:
    trycount += 1
    print("TRYCOUNT=",trycount)
    shake = tf.random.uniform( (WIDE,WIDE),  minval=-SHAKE_RANGE, maxval=+SHAKE_RANGE )
    # print("SHAKE=",shake)
    work = work + shake
    determinant = tf.linalg.det( work )
    print("det=",determinant)

print("\nTRYCOUNT=",trycount)


inverted = tf.linalg.inv( work )
print( "\ninverted=", inverted )

# # lets make sure we can invert
# determinant = tf.linalg.det( work )
# print("DET=",determinant )
#
# if determinant != 0.:
#     print( "inverted=", tf.linalg.inv( concatted ) )
# else:
#     addthis = tf.random.uniform( (10,10),  minval=-SHAKE_RANGE, maxval=+SHAKE_RANGE )
#     trythis = concatted + addthis
#     determinant = tf.linalg.det( trythis )
#     print("DET=",determinant )
#
#     if determinant != 0.:
#         print( "inverted=", tf.linalg.inv( trythis ) )


# can we use the inverted matrix with our original inputs?
wide_symbolic_four = tf.concat( [symbolic_four, tf.ones( (6,) ) ], axis=0 )
print("\nwide_symbolic_four=",wide_symbolic_four)

reversed = tf.linalg.matvec( inverted, wide_symbolic_four, transpose_a=True  )
print("\nreversed=",reversed)

identityO = tf.linalg.matmul( inverted, original_squared )
print("\nidentity from original=",identityO)
print("\nrounded identity from original=", tf.round(identityO * 10.)/10. )

identityS = tf.linalg.matmul( inverted, work )
print("\nidentity from modified=",identityS)
print("\nrounded identity from modified=",tf.round(identityS * 10.)/10. )

print("\nTryCount=",trycount)
