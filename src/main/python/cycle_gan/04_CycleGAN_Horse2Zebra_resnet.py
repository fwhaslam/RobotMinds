#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   NOTE: 04 means this was the fourth modified version of the original tutorial
#
#   This is a copy of CycleGAN_Horse2AdventureTime.py with the following changes:
#       The generators and discriminators are allocated in local methods using functional api.
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

import tensorflow as tf

import tensorflow_datasets as tfds

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

from _utilities.tf_cyclegan_tools import *

# configuration and datasets
AUTOTUNE = tf.data.AUTOTUNE
plt.interactive(True)

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

###########################################################################################################
#       Constant Parameters

EPOCHS = 100

BUFFER_SIZE = 1000
BATCH_SIZE = 1

checkpoint_path = "zebra_resnet_ckpt/train"
OUTPUT_CHANNELS = 3

LAMBDA = 10

GEN_LOSS_FACTOR = 1.0 * 0.5       # 1.0 = generator loss ?
DISC_LOSS_FACTOR = 0.5 * 0.5      # 0.5 = discrinator
CYCLE_LOSS_FACTOR = 10.0    # LAMBDA * 1.0 = full cycle
IDENT_LOSS_FACTOR = 5.0     # LAMBDA * 0.5 = real to same


train_horses = train_horses.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_zebras = train_zebras.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

# select one horse/zebra for demonstrating progress :: Display alongside generator images
sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))
horse_loop = iter(RepeatLoop(train_horses))


#####
# Workaround for Gradient bug?
#####
# @tf.RegisterGradient("ResizeArea")
# def _ResizeArea(op,grad):
#     return [0.,0.]

########################################################################################################################
#   Functional Api version of UNet generator from tensorflow_examples.models.pix2pix
#       see: https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
#

class InstanceNormalization(tf.keras.layers.Layer):
    r"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset




def downsample(filters, size, strides=2):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, strides=2):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,padding='same'))
    result.add(InstanceNormalization())
    result.add(tf.keras.layers.ReLU())
    return result

def merge_layer( x, y ):
    return tf.keras.layers.Concatenate()( [x, y] )
    # return tf.keras.layers.Add()( [x, y] )


def resnet_generator( output_channels ):
    r"""Starting generator using resnet blocks.
    Evaluation:
        Round1 = resnet blocks => created brown blocks
        Round2 = skip every 3 layers, concatenate => Creates pretty stripes, captures some shapes from original image.
        Round3 = more aggressive downsampling to save memory =>runs faster, about the same, some pontillism
    """

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # start model building
    x = inputs
    skip1 = x

    # downsample layers
    x = downsample( 64, 4, 2 )(x)           # ( 128,128,64 )
    x = downsample( 128, 4, 2 )(x)          # (bs, 64, 64, 128)

    x1 = tf.keras.layers.Conv2D( 128, 4, strides=4, padding='same')(skip1)
    skip2 = x = merge_layer( x, x1 )        # (bs, 32, 32, 128/256)

    x = downsample( 256, 4, 2 )(x)          # (bs, 32, 32, 256)
    x = downsample( 512, 4, 2 )(x)          # (bs, 16, 16, 512)

    x2 = tf.keras.layers.Conv2D( 512, 4, strides=4, padding='same')(skip2)
    skip3 = x = merge_layer( x, x2 )                # (bs, 4, 4, 512/1024)

    x = downsample( 512, 4, 2 )(x)          # (bs, 8, 8, 512)
    x = downsample( 512, 4, 2 )(x)          # (bs, 4, 4, 512)

    x3 = tf.keras.layers.Conv2D( 512, 4, strides=4, padding='same')(skip3)
    skip4 = x = merge_layer( x, x3 )                # (bs, 4, 4, 512/1024)


    ###############################
    # middle layers
    x = downsample( 512, 4, 2 )(x)          # (bs, 2, 2, 512)
    x = downsample( 512, 4, 2 )(x)          # (bs, 1, 1, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 2, 2, 512)

    ###############################
    # upsample layers

    x4 = tf.keras.layers.Conv2D( 512, 4, strides=2, padding='same')(skip4)
    skip5 = x = merge_layer( x, x4 )                # (bs, 2, 2, 512/1024)

    x = upsample(512, 4, 2)(x)          # (bs, 4, 4, 512)
    x = upsample(512, 4, 2)(x)          # (bs, 8, 8, 512))

    x5 = tf.keras.layers.Conv2DTranspose( 512, 4, strides=4)(skip5)  # (bs, 8, 8, 512)
    skip6 = x = merge_layer( x, x5 )        # (bs, 8, 8, 512/1024))

    x = upsample(512, 4, 2)(x)              # (bs, 16, 16, 512))
    x = upsample(256, 4, 2)(x)               # (bs, 32, 32, 256)

    x6 = tf.keras.layers.Conv2DTranspose( 256, 4, strides=4)(skip6)  # (bs, 32, 32, 256)
    skip7 = x = merge_layer( x, x6 )        # (bs, 16, 16, 256/512))

    x = upsample(128, 4, 2)(x)               # (bs, 64, 64, 128)
    x = upsample(64, 4, 2)(x)               # (bs, 128, 128, 64)

    x7 = tf.keras.layers.Conv2DTranspose( 64, 4, strides=4)(skip7)  # (bs, 128, 128, 64)
    x = merge_layer( x, x7 )                                # (bs, 128, 128, 64/128)

    # cleanup to 3 channels
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    outputs = last(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def discriminator():
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    Args:
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
      target: Bool, indicating whether target image is an input or not.
    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    down1 = downsample(64, 4)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


# generator_g = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
# generator_f = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_g = resnet_generator(OUTPUT_CHANNELS)
generator_f = resnet_generator(OUTPUT_CHANNELS)

discriminator_x = discriminator()
discriminator_y = discriminator()

########################################################################################################################
# Core Loss

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * DISC_LOSS_FACTOR

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated) * GEN_LOSS_FACTOR

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return loss1 * CYCLE_LOSS_FACTOR

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return loss * IDENT_LOSS_FACTOR

########################################################################################################################
# Optimizers

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

########################################################################################################################
# Checkpoints

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

########################################################################################################################
# Train

@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

# MAIN LOOP
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print ('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    next_sample = next( horse_loop )
    generate_images(generator_g, generator_f, next_sample)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

# Run the trained model on the test dataset
# for inp in test_horses.take(5):
#     generate_images(generator_g, generator_f, inp)
#
