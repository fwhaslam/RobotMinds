#
#   This code is a modified version of:
#       93_DCGAN_HandwrittenDigits.py
#
#   This code is identical to what is presented on the tutorial page except for:
#       'pip install imageio'
#       'pip install git+https://github.com/tensorflow/docs'
#       Added SMALL_MACHINE so I could run on my laptop in a reasonable timeframe
#       using plt.ion() with plt.pause(30) so training continues when I am away.
#       using checkpoint_manager to restore both model and epoch_count ( eg. latest_epoch )
#       images and training going to single subfolder
#
#   Code tested with:
#       Tensorflow 2.10.0/cpuOnly  ( complains about Cudart64_110.dll, but it still functions )
#
import sys
sys.path.append('..')

import os
import re
import time
import glob

import tensorflow as tf
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from _utilities.tf_tensor_tools import *

plt.ion()

# model and samples directory
VERSION = 'v02'
SCRIPT_TITLE = f'MnistDigits_withIdentity/{VERSION}'

CHECKPOINT_DIR = f'./guided_dcgan_ckpt/{VERSION}'
TRAIN_FOLDER = CHECKPOINT_DIR + "/train"
IMAGE_FOLDER = CHECKPOINT_DIR + "/samples"
ANIMATION_FILE = f'guided_dcgan_animation_{VERSION}.gif'

# am I running this under via CPU on my laptop ( eg. small machine ) ?
# SMALL_MACHINE = ( tf.test.is_gpu_available() == False )   # deprecated method
GPU_LIST = tf.config.list_physical_devices('GPU')
SMALL_MACHINE = ( len(GPU_LIST) == 0 )
# TODO: check to see if nvidia cuda is intalled
#  import os / os.system( 'nvidia-smi' ) or  os.system( 'nvcc --Version )

EPOCHS = 50
BUFFER_SIZE = 60_000
BATCH_SIZE = 256
NOISE_DIM = 100

POPUP_WAIT_TIME = 1 #30

if SMALL_MACHINE:
    EPOCHS = 5
    BUFFER_SIZE = 3_000
    BATCH_SIZE = 32

print("\n\nConfig:")
print("EPOCHS=",EPOCHS)
print("BUFFER_SIZE=",BUFFER_SIZE)
print("BATCH_SIZE=",BATCH_SIZE)
print("\n\n")

DIGIT_NAME = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

########################################################################################################################

# load MNIST handwritten digits dataset
(digit_images, digit_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = digit_images.reshape(digit_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Batch and shuffle the data, reduce size if necessary ( eg. take() )
train_dataset = tf.data.Dataset.\
    from_tensor_slices( (train_images,digit_labels) ).\
    take(BUFFER_SIZE).\
    shuffle(BUFFER_SIZE).\
    batch(BATCH_SIZE)

# tf.print('dataset=',train_dataset)      # shows two tensorspecs, so both images + labels are in here

def make_generator_model():
    r"""Generator will have two inputs ( noise, digit ), and two outputs ( image, digit ).
    The 'digit' is a value from zero to nine indicating what the image should look like.
    The output 'digit' will be the same as the input 'digit'.
    """

    input0 = tf.keras.layers.Input(shape=[NOISE_DIM,])
    input1 = tf.keras.layers.Input(shape=[1])

    # appending digit 3 weight embedding to the noise 100 weights
    embed = tf.keras.layers.Embedding( 10, 3 )(input1)  # [0-9] digit embeds as 3-dim dense key ( bs, 1,3 )
    embed = tf.keras.layers.Flatten()(embed)
    x = tf.keras.layers.Concatenate()( [input0,embed] )

    # begin expanding data towards image generation
    x = tf.keras.layers.Dense(7*7*256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Reshape((7, 7, 256))(x)

    x = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    assert x.shape == (None, 14, 14, 64)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # cleanup
    x = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    output0 = x

    return tf.keras.Model(inputs=(input0,input1), outputs=(output0,input1) )

generator = make_generator_model()

# noise = tf.random.normal([1, NOISE_DIM])
# digit = tf.random.uniform( (1,), maxval=10, dtype=tf.int32 )
# generated_image = generator( (noise,digit), training=False)

########################################################################################################################

def make_discriminator_model():
    r"""Discriminator will have two inputs ( image, digit ).
    Digit is the label from the dataset saying what the image should look like ( zero to nine ).
    Output is still binary cross entropy value.
    """

    input0 = tf.keras.layers.Input(shape=[28,28,1])
    input1 = tf.keras.layers.Input(shape=[1])

    # merge embedded digit with image
    embed = tf.keras.layers.Embedding( 10, 3, input_length=1 )(input1)  # [0-9] digit embeds as 3 dimensional dense key
    embed = tf.keras.layers.Flatten()(embed)
    x = tf.keras.layers.Dense( 28*28 )( embed )
    x = tf.keras.layers.Reshape( (28,28,1) )( x )
    x = tf.keras.layers.Concatenate( axis=-1)( [input0,x] )

    # begin convolutional analysis
    x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    outputs = x

    return tf.keras.Model(inputs=(input0,input1), outputs=outputs )


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

ckpt = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(ckpt, TRAIN_FOLDER, max_to_keep=10)

def load_with_epoch( ckpt, ckpt_manager ):
    r"""if a checkpoint exists, restore the latest checkpoint."""
    checkpoint_name = ckpt_manager.latest_checkpoint
    if checkpoint_name:
        tf.print("\n\nLATEST_CHECKPOINT =",checkpoint_name)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
        return int( re.findall(r'\d+', str.split(checkpoint_name,'/')[-1] )[0] )
    else:
        return 0

# returns latest epoch, or zero if just starting
latest_epoch = load_with_epoch( ckpt, ckpt_manager )

# make sure that missing directories are created
if not os.path.exists(TRAIN_FOLDER):
    os.makedirs(TRAIN_FOLDER)
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

########################################################################################################################

sample_seed = 43271823
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
save_image_noise = tf.random.normal( [num_examples_to_generate, NOISE_DIM], seed=sample_seed )
save_image_digit = tf.random.uniform( (num_examples_to_generate,1), maxval=10, dtype=tf.int32, seed=sample_seed )

save_image_digit = tf.reshape( [val % 10 for val in range(num_examples_to_generate) ], (16,1) )
save_image_seed = (save_image_noise,save_image_digit)

# print("save_image_seed=",save_image_seed)
# print("noise.shape=",tf.shape(save_image_seed[0]))
# print("digit.shape=",tf.shape(save_image_seed[1]))


# Notice the use of `tf.function`.  This annotation causes the function to be compiled for GPU.
@tf.function
def train_step(images):

    noise = tf.random.normal( [BATCH_SIZE, NOISE_DIM] )
    digit = tf.random.uniform( [BATCH_SIZE, 1], maxval=10, dtype=tf.int32 )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image_set = generator( (noise,digit), training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_image_set, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train( dataset, steps ):

    global latest_epoch

    for step in range(steps):
        print(f"\n>>> Training step {1+step} of {steps}")
        latest_epoch = latest_epoch + 1
        start = time.time()

        for image_batch in dataset:
            # tf.print('image_batch[0].shape=',tf.shape(image_batch[0]))
            # tf.print('image_batch[1].shape=',tf.shape(image_batch[1]))
            train_step(image_batch)

        total_time = time.time()-start

        # Produce images for the GIF as you go
        generate_and_save_images( generator, latest_epoch, save_image_seed )

        ckpt_manager.save()
        print ('Time for epoch {} is {} sec'.format(latest_epoch, total_time))

    # # Generate after the final epoch
    # generate_and_save_images( generator, latest_epoch, save_image_seed )

    return


def generate_and_save_images( model, epoch, test_input ):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    global IMAGE_FOLDER
    outputs = model(test_input, training=False)

    (predictions,labels) = outputs
    # print("predictions=",predictions)
    # print("labels=",labels)

    plt.close()
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle( SCRIPT_TITLE, fontsize=16)

    for i in range(predictions.shape[0]):
        ax = plt.subplot(4, 4, i+1)
        name = "p="+DIGIT_NAME[ int(labels[i][0]) ]
        ax.set_title( name )
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    image_path = IMAGE_FOLDER+'/image_at_epoch_{:04d}.png'.format(epoch)
    plt.savefig(image_path)

    plt.show()
    plt.pause(POPUP_WAIT_TIME)
    return

# train the model for some number of epochs
train( train_dataset, EPOCHS )

########################################################################################################################

# Use imageio to create an animated gif using the images saved during training.

with imageio.get_writer(ANIMATION_FILE, mode='I') as writer:
    filenames = glob.glob(IMAGE_FOLDER + '/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(ANIMATION_FILE)
