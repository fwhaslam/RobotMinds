
import tensorflow as tf
from stoplight_dataset import *

FLAVOR = 'none'

SAMPLE_COUNT = 30000

########################################################################################################################



ds = stoplight_dataset()
images,notes = ds.build( SAMPLE_COUNT )

TRAIN_COUNT = int( SAMPLE_COUNT * 0.8 )

train_images = images[:TRAIN_COUNT]
train_image_set = notes[:TRAIN_COUNT]

test_images = images[TRAIN_COUNT:]
test_image_set = notes[TRAIN_COUNT:]

########################################################################################################################

def stoplight_loss_function( y_true, y_pred ):
    r"""'y_true' will be values of (bs,OPTION_COUNT).
    'y_pred' will be one_hot of (bs,OPTION_COUNT)
    loss = - values * one_hot
    """

    value = y_true * y_pred
    value = tf.reduce_sum( value, axis=-1)

    # loss is negative value
    return -value

def create_model_v2( shape ):

    # units = WIDE * TALL * CHANNELS      # 3 * 1k

    x = inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D( 32, 1, activation='selu' )(x)               # ( bs, 48,48, 32 )
    x = tf.keras.layers.Conv2D( 128, 2, strides=2, activation='selu' )(x)   # ( bs, 24,24, 128 )
    x = tf.keras.layers.Conv2D( 256, 2, strides=2, activation='selu' )(x)   # ( bs, 12,12, 256 )
    x = tf.keras.layers.Conv2D( 512, 2, strides=2, activation='selu' )(x)   # ( bs, 6,6, 512 )
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048,activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(OPTIONS_COUNT,activation='softmax')(x)
    outputs = x

    return tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v2')


def create_model_v1( shape ):

    # units = WIDE * TALL * CHANNELS      # 3 * 1k

    x = inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3072,activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(1024,activation='LeakyReLU')(x)
    x = tf.keras.layers.Dense(OPTIONS_COUNT,activation='softmax')(x)
    outputs = x

    return tf.keras.Model(inputs=inputs, outputs=outputs,name=FLAVOR+'_model_v1')

########################################################################################################################
# create runner and drive process model

from pnz_runner import pnz_runner

model_id = 'v1'
ckpt_folder = 'pnz_runner/stoplight/' + model_id


model = create_model_v2( IMAGE_SHAPE )
loss_function = stoplight_loss_function
optimizer = tf.optimizers.Adam( 0.0001 )


pnz_runner(
    FLAVOR,
    model, loss_function, optimizer,
    epochs = 10
).run(
    train_images,train_image_set,
    test_images,test_image_set,
    ckpt_folder
)

