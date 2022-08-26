#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   This is a copy of CycleGAN_Horse2AdventureTime.py with the following changes:
#       The generators and discriminators are allocated in local methods using functional api.
#
#   Code tested with:
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

from _utilities.tf_cyclegan_tools import *
import resnet.tf_resnet_tools as resnet

# configuration and datasets
AUTOTUNE = tf.data.AUTOTUNE
plt.interactive(True)

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

###########################################################################################################
#       Constant Parameters


BUFFER_SIZE = 1000
BATCH_SIZE = 1
EPOCHS = 40
checkpoint_path = "zebra_resnet_ckpt/train"

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

OUTPUT_CHANNELS = 3

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


class MyAreaResize(tf.keras.layers.Layer):

    def __init__(self, wide, tall):
        super(MyAreaResize, self).__init__()
        self.wide = wide
        self.tall = tall

    def build(self,other):
        # tf.print("MAR.OTHER=",other)
        return

    def call(self, inputs):
        work = tf.repeat( inputs, repeats=self.wide, axis=1 )
        return tf.repeat( work, repeats=self.tall, axis=2 )


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    r"""Downsamples an input.
    Conv2D => Batchnorm => LeakyRelu
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_norm: If True, adds the batchnorm layer
    Returns:
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


# def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
#     r"""Upsamples an input.
#     Conv2DTranspose => Batchnorm => Dropout => Relu
#     Args:
#       filters: number of filters
#       size: filter size
#       norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
#       apply_dropout: If True, adds the dropout layer
#     Returns:
#       Upsample Sequential Model
#     """
#
#     initializer = tf.random_normal_initializer(0., 0.02)
#
#     result = tf.keras.Sequential()
#     result.add(
#         tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
#                                         padding='same',
#                                         kernel_initializer=initializer,
#                                         use_bias=False))
#
#     if norm_type.lower() == 'batchnorm':
#         result.add(tf.keras.layers.BatchNormalization())
#     elif norm_type.lower() == 'instancenorm':
#         result.add(InstanceNormalization())
#
#     if apply_dropout:
#         result.add(tf.keras.layers.Dropout(0.5))
#
#     result.add(tf.keras.layers.ReLU())
#
#     return result

def resizing_layer( size ):
    return MyAreaResize( size, size )
    # return tf.keras.layers.Resizing( size, size, interpolation='nearest' )
    # return tf.keras.layers.Resizing( 8,8, interpolation='area' )
    # return tf.keras.layers.UpSampling2D( size=4, interpolation='area' )

def resnet_generator(output_channels, norm_type='batchnorm'):
    r"""Starting generator using resnet blocks.
    Evaluation = pretty bad.  It can emulate horizans but thats about it.
    """

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    initializer = tf.random_normal_initializer(0., 0.02)
    if (norm_type=='batchnorm'):
        resnet.set_make_norm_layer( lambda : tf.keras.layers.BatchNormalization(axis=3) )
    elif (norm_type=='instancenorm'):
        resnet.set_make_norm_layer( lambda : InstanceNormalization() )
    elif (norm_type=='layernorm'):
        resnet.set_make_norm_layer( lambda : tf.keras.layers.LayerNormalization(axis=3) )
    else:
        print("Invalid normtype=",norm_type)
        exit( -1 )

    # start model building
    x = inputs

    # downsample layers
    x = resnet.proj_block(64,kernel_size=4,strides=2)(x)    # ( 128,128,64 )
    x = tf.keras.layers.MaxPooling2D(2)(x)                  # ( 64,64,64 )
    x = resnet.proj_block(128,kernel_size=4,strides=2)(x)   # ( 32,32,128 )
    x = tf.keras.layers.MaxPooling2D(2)(x)                  # ( 16,16,128 )
    x = resnet.proj_block(256,kernel_size=4,strides=2)(x)   # ( 8,8,256 )
    x = tf.keras.layers.MaxPooling2D(2)(x)                  # ( 4,4,256 )
    x = resnet.proj_block(512,kernel_size=4,strides=2)(x)   # ( 2,2,512 )
    x_skip = x
    # x = tf.keras.layers.MaxPooling2D(2)(x)                  # ( 1,1,512 )

    # center layer is dense, then reshape for expansion
    x = tf.keras.layers.Flatten()(x)                        # ( 2048 )
    x = tf.keras.layers.Dense(2048,activation='relu')(x)    # ( 2048 )
    x = tf.keras.layers.Reshape( (2,2,512), name='reshape_center')(x)             # (2,2,512)

    # upsample layers
    x = tf.keras.layers.Add()([x,x_skip])   # skip center block
    x = resnet.proj_block(512,kernel_size=4)(x)     # ( 2,2,512 )
    x = resizing_layer( 4 )(x)                      # ( 8,8,512 )
    x = resnet.proj_block(256,kernel_size=4)(x)     # ( 8,8,256 )
    x = resizing_layer( 4 )(x)                      # ( 32,32,256 )
    x = resnet.proj_block(128,kernel_size=4)(x)     # ( 32,32,128 )
    x = resizing_layer( 4 )(x)                      # ( 128,128,128 )
    x = resnet.proj_block(64,kernel_size=4)(x)      # ( 128,128,64 )

    # cleanup to 3 channels
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')                                                  # (bs, 256, 256, 3)
    outputs = last(x)
    print("    >>>>>     Shape8=",tf.shape(outputs))

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# def unet_generator(output_channels, norm_type='batchnorm'):
#     """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
#     Args:
#       output_channels: Output channels
#       norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
#     Returns:
#       Generator model
#     """
#
#     down_stack = [
#         downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
#         downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
#         downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
#         downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
#         downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
#         downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
#         downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
#         downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
#     ]
#
#     up_stack = [
#         upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
#         upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
#         upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
#         upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
#         upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
#         upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
#         upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
#     ]
#
#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = tf.keras.layers.Conv2DTranspose(
#         output_channels, 4, strides=2,
#         padding='same', kernel_initializer=initializer,
#         activation='tanh')  # (bs, 256, 256, 3)
#
#     concat = tf.keras.layers.Concatenate()
#
#     inputs = tf.keras.layers.Input(shape=[None, None, 3])
#     x = inputs
#
#     # Downsampling through the model
#     skips = []
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)
#
#     skips = reversed(skips[:-1])
#
#     # Upsampling and establishing the skip connections
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = concat([x, skip])
#
#     x = last(x)
#
#     return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=True):
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

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


# generator_g = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
# generator_f = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_g = resnet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = resnet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = discriminator(norm_type='instancenorm', target=False)
discriminator_y = discriminator(norm_type='instancenorm', target=False)

########################################################################################################################


# demonstrate the 'real horse/zebra' analysis

def display_reality_analysis():
    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('Is a real zebra?')
    plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(122)
    plt.title('Is a real horse?')
    plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

    plt.show()
    plt.pause(1) # show on startup

display_reality_analysis()

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

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


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.close()     # close previous image
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    plt.pause(GPU_REST_SECONDS)


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
    generate_images(generator_g, next_sample)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

# Run the trained model on the test dataset
for inp in test_horses.take(5):
    generate_images(generator_g, inp)

