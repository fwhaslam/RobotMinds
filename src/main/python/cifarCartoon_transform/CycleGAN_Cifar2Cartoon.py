#
#   Tensorflow Cycle Generative Adversarial Network.
#
#   This Cycle GAN converts images from Cifar to Cartoon and back.
#
#   This is a copy of CycleGAN_Horse2Zebra.py with the following changes:
#       horse + zebra renamed first+second
#       first is Cifar10 dataset
#       second is a cartoon dataset
#
#   Code tested with:
#       Tensorflow 2.9.l / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#

import sys
sys.path.append('..')

# common
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output

# tensorflow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

# local tools
import _utilities.tf_cyclegan_tools as funcs
from _utilities.tf_cyclegan_tools import *


# configuration and datasets
AUTOTUNE = tf.data.AUTOTUNE
plt.interactive(True)



BUFFER_SIZE = 1000
BATCH_SIZE = 1
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
# GPU_REST_SECONDS = 20

something = datasets.cifar10.load_data()
print("SOMETHING=",something)

(train_images, train_label_array), (test_images, test_label_array) = datasets.cifar10.load_data()

train_first = tf.data.Dataset.from_tensors( train_images ).unbatch().prefetch( BATCH_SIZE )
test_first = tf.data.Dataset.from_tensors( test_images ).unbatch().prefetch( BATCH_SIZE )

######################################
#   Adventure Time path

root_path = os.path.expanduser( '~/_Workspace/Datasets/KaggleCartoonTiny/cartoon_classification' )

# img_height = 180
# img_width = 320
# batch_size = 32

# fred: switch to match 'horse' image size
img_height = 256
img_width = 256


advent_train_ds = tf.keras.utils.image_dataset_from_directory(
    Path( root_path ) / "TRAIN" / "adventure_time",
    label_mode = None,
    image_size=(img_height, img_width),
    interpolation=funcs.preferred_resize_method
)


advent_test_ds = tf.keras.utils.image_dataset_from_directory(
    Path( root_path ) / "Test" / "adventure_time",
    label_mode = None,
    image_size=(img_height, img_width),
    interpolation=funcs.preferred_resize_method
)

train_second =  advent_train_ds.unbatch().prefetch( BATCH_SIZE )
print("train_second = ", train_second )
test_second = advent_test_ds.unbatch().prefetch( BATCH_SIZE )
print("test_second = ", test_second )

######################################


train_first = train_first.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_second = train_second.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_first = test_first.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_second = test_second.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

# select one horse/zebra for demonstrating progress :: Display alongside generator images
sample_first = next(iter(train_first))
sample_second = next(iter(train_second))
first_loop = iter(RepeatLoop(train_first))

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

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

checkpoint_path = "./checkpoints/train"

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

EPOCHS = 40

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
    for image_x, image_y in tf.data.Dataset.zip((train_first, train_second)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print ('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_first) so that the progress of the model
    # is clearly visible.
    next_sample = next( first_loop )
    generate_images(generator_g, next_sample)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

# Run the trained model on the test dataset
for inp in test_first.take(5):
    generate_images(generator_g, inp)

