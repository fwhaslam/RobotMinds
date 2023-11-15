import tensorflow as tf
import numpy as np


embedding0 = tf.constant([[-0.8959618 ,  2.2839823 , -1.9336246 , -0.01028966],
                         [ 1.2777444 , -1.6295084 ,  0.8190948 ,  0.59062874],
                         [-1.8145654 , -0.9967288 , -0.51381916,  1.9750237 ],
                         [-0.8401218 ,  0.3790238 , -1.071506  ,  1.0719615 ],
                         [ 1.1598861 , -0.8027018 ,  2.1503801 , -1.4213352 ],
                         [ 0.20158052,  0.96800387, -0.7521279 , -0.242653  ],
                         [-1.1466305 , -2.322513  ,  1.1108656 ,  1.0629116 ],
                         [ 1.5732962 ,  1.444018  , -0.18647921, -2.0115876 ],
                         [-0.25006676, -0.43794042,  0.19088256,  0.33338353],
                         [ 0.7257275 ,  0.22434635,  0.71969056, -1.1542754 ]])

embedding1 = tf.constant( [[-1.66477358e+00, -1.64783573e+00, -2.02043366e+00,
                            -2.12636089e+00],
                           [-4.79070023e-02,  3.81615281e-01,  5.26476979e-01,
                            7.64577031e-01],
                           [-8.54591548e-01, -1.15581088e-01, -9.68311548e-01,
                            -8.42832267e-01],
                           [-2.28769749e-01, -2.58928478e-01, -8.76648724e-02,
                            3.66216265e-02],
                           [ 9.50155616e-01,  4.65923071e-01,  8.02741706e-01,
                             9.44929361e-01],
                           [-2.74308532e-01, -7.63988376e-01, -1.68071210e-01,
                            -3.26336116e-01],
                           [-1.26241350e+00, -1.08196270e+00, -9.67322111e-01,
                            -1.44277501e+00],
                           [ 1.91304588e+00,  1.66931772e+00,  1.81241477e+00,
                             1.89504707e+00],
                           [ 1.73159003e-01,  7.89087673e-04,  1.40223339e-01,
                             1.21108286e-01],
                           [ 1.32353675e+00,  1.05757892e+00,  1.24285388e+00,
                             1.25904000e+00]])

embedBefore = tf.constant([[ 0.02172827,  0.03777016,  0.03369477, -0.04685205],
                           [-0.01126382,  0.02621901,  0.03488164, -0.02823803],
                           [ 0.01398848,  0.01955033,  0.00181626, -0.0016823 ],
                           [-0.01663943,  0.01319352, -0.00257684,  0.02440593],
                           [-0.01974982,  0.0470874 ,  0.00155471,  0.0102881 ],
                           [ 0.04399853, -0.04036029,  0.04488299,  0.03565487],
                           [-0.03534895,  0.00330114, -0.03708703, -0.02748339],
                           [ 0.04682765,  0.04840603, -0.0104972 ,  0.01208546],
                           [-0.02081177,  0.04170794, -0.03213545, -0.00084712],
                           [-0.01906513,  0.04635538, -0.03299944, -0.04190104]])

embedAfter1 = tf.constant([[-1.5772935 , -1.9400489 ,  3.2575665 , -3.118669  ],
                          [ 2.4549708 ,  1.9430304 , -0.7566304 ,  3.684524  ],
                          [-1.7409592 , -0.857967  ,  2.125937  ,  1.0320624 ],
                          [ 1.9064583 , -1.1680357 ,  1.4436415 ,  1.4112303 ],
                          [-1.1937866 ,  1.1533616 , -2.2861059 , -0.72011316],
                          [ 0.71896875, -0.66748416,  0.7259194 , -0.89915204],
                          [-2.7474337 , -1.1665169 , -0.39833945, -1.916639  ],
                          [ 1.6960709 ,  2.246159  , -1.116916  ,  0.7263426 ],
                          [-0.23457795, -0.54581594, -0.96676797, -0.08872323],
                          [ 0.71255857,  1.0608187 , -1.5451485 , -0.39962217]])

embedAfter2 = tf.constant([[-2.9031868 ,  1.8127636 , -3.3649714 ,  2.8705041 ],
                           [ 2.358446  ,  2.1940012 ,  3.2273066 , -1.8994598 ],
                           [-1.9625907 ,  1.5993571 ,  1.3999735 ,  1.0675763 ],
                           [ 0.5639672 ,  2.132106  ,  0.3874994 , -0.55647796],
                           [-0.33186728, -3.1637902 , -0.23784363, -0.3663448 ],
                           [ 0.03718204,  0.78591204, -2.2162845 ,  0.80543387],
                           [-2.4862604 , -2.0642962 , -1.3947288 ,  1.4448609 ],
                           [ 1.7803938 , -0.85629964,  2.595718  , -2.186285  ],
                           [ 0.6320304 , -0.29520667, -0.36722526,  0.4017116 ],
                           [ 1.3596541 , -1.6808116 ,  0.00640297, -1.0381615 ]])

# cross similarity based on cosine similarity


test1 = tf.constant( [[1., 1.], [1., 1.], [1., 1.]] )   # should be all minus ones ( eg. all same )
test2 = tf.constant( [[1., 1.], [1., -1.], [-1., -1.]] )    # should be minus one, then zero, plus one


def similarity_grid( matrix, simplify=False ):

    # step 1: copy values vertically for first matrix
    # print('shape=',matrix.shape)
    first = tf.repeat( tf.expand_dims( matrix, 0 ), repeats=matrix.shape[0], axis=0 )
    # print('first=',first)

    # step 2: second matrix transposes vectors diagonally
    second = tf.transpose( first, perm=[1,0,2] )
    # print('second=',second)

    # step 3: perform similarity measure for cross values

    # cosine scaled to [0,1] where 1 means high similarity
    work = (1. - tf.keras.losses.cosine_similarity( first, second, axis=2) ) / 2.

    # rmse: scaled to 1 for high similarity, zero or negative for low similarity
    # work = 1. - tf.sqrt( tf.reduce_sum( tf.square(first - second), axis=-1) ) / 5.
    # print("\nsimilarity.shape=",work.shape)
    # print("similarity=",work)

    if simplify:
        # print('shape=',work.shape[0:-1])
        # work = tf.linalg.set_diag( work, tf.zeros(work.shape[0:-1]) )
        work = tf.linalg.band_part( work, -1, 0)
        work = tf.round( 10. * work) / 10.
        # print("work=",work)

    return work


result = similarity_grid( test1, True )
print("\ntest1=",result)

result = similarity_grid( test2, True )
print("\ntest2=",result)

# result = similarity_grid( embedding0, True )
# print("\nembedding0=", result )
#
# result = similarity_grid( embedding1, True )
# print("\nembedding1=", result )

result = similarity_grid( embedBefore, True )
print("\nembedBefore=", result )

result = similarity_grid( embedAfter1, True )
print("\nembedAfter1=", result )

result = similarity_grid( embedAfter2, True )
print("\nembedAfter2=", result )
