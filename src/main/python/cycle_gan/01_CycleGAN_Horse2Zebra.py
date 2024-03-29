#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from horse to zebra and back.
#
#   NOTE: 01 means this was the first modified version of the original tutorial
#
#   This is a copy of 75_CycleGAN_Horse2Zebra.py with the following changes:
#       Processing does not pause for displayed images.
#       Old images are closed as new images are opened
#       The focal image rotates through the dataset so we can can see progress on multiple images.
#       Some methods have been extracted to a shared function file.
#       periodic image display has: original, transformed, cycled, and same
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
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

# configuration and datasets
AUTOTUNE = tf.data.AUTOTUNE
plt.interactive(True)

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

########################################################################################################################
#       Constant Parameters

BUFFER_SIZE = 1000
BATCH_SIZE = 1
EPOCHS = 40
checkpoint_path = "zebra_ckpt/train"

# IMG_WIDTH = 256
# IMG_HEIGHT = 256
# GPU_REST_SECONDS = 20
#
# def random_crop(image):
#     cropped_image = tf.image.random_crop(
#         image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
#
#     return cropped_image
#
# # normalizing the images to [-1, 1]
# def normalize(image):
#     image = tf.cast(image, tf.float32)
#     image = (image / 127.5) - 1
#     return image
#
# def random_jitter(image):
#     # resizing to 286 x 286 x 3
#     image = tf.image.resize(image, [286, 286],
#                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#
#     # randomly cropping to 256 x 256 x 3
#     image = random_crop(image)
#
#     # random mirroring
#     image = tf.image.random_flip_left_right(image)
#
#     return image
#
# def preprocess_image_train(image, label):
#     image = random_jitter(image)
#     image = normalize(image)
#     return image
#
# def preprocess_image_test(image, label):
#     image = normalize(image)
#     return image

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

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# def display_generator_comparison( ):
#     plt.subplot(121)
#     plt.title('Horse')
#     plt.imshow(sample_horse[0] * 0.5 + 0.5)
#
#     plt.subplot(122)
#     plt.title('Horse with random jitter')
#     plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
#
#     plt.subplot(121)
#     plt.title('Zebra')
#     plt.imshow(sample_zebra[0] * 0.5 + 0.5)
#
#     plt.subplot(122)
#     plt.title('Zebra with random jitter')
#     plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)
#
#     to_zebra = generator_g(sample_horse)
#     to_horse = generator_f(sample_zebra)
#     plt.figure(figsize=(8, 8))
#     contrast = 8
#
#     imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
#     title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']
#
#     for i in range(len(imgs)):
#         plt.subplot(2, 2, i+1)
#         plt.title(title[i])
#         if i % 2 == 0:
#             plt.imshow(imgs[i][0] * 0.5 + 0.5)
#         else:
#             plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
#     plt.show()
#     plt.pause(1) # show on startup

# display_generator_comparison( gen_g=generator_g, gen_f=generator_f, samp_h=sample_horse, samp_z=sample_zebra)

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
for inp in test_horses.take(5):
    generate_images(generator_g, generator_f, inp)

