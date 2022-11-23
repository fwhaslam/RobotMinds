#
#   This code is a modified version of:
#       01_DCGAN_MnistDigits_original.py
#
#   This code is identical to what is presented on the tutorial page except for:
#       Some percentage of the images are 'not-a-number'.
#       This is achieved by transforming the image, and editing the label from True to False.
#
#   Code tested with:
#
import sys
sys.path.append('..')

import os
import re
import time
import glob
import random

import tensorflow as tf
import imageio.v2 as imageio
import matplotlib.pyplot as plt

import _utilities.tf_layer_tools as tflt
import _utilities.tf_tensor_tools as tftt

plt.ion()

# false percent derived from command line
FALSE_PERCENT = 10
if len(sys.argv)>1:
    FALSE_PERCENT = int( sys.argv[1] )

# model and samples directory
VERSION = f'v13p{FALSE_PERCENT}'
SCRIPT_TITLE = f'MnistDigits_tighter_someFalse/{VERSION}'

CHECKPOINT_DIR = f'./guided_dcgan_ckpt/{VERSION}'
TRAIN_FOLDER = CHECKPOINT_DIR + "/train"
IMAGE_FOLDER = CHECKPOINT_DIR + "/samples"
ANIMATION_FILE = f'guided_dcgan_animation_{VERSION}.gif'

# am I running this under via CPU on my laptop ( eg. small machine ) ?
# SMALL_MACHINE = ( tf.test.is_gpu_available() == False )   # deprecated method
GPU_LIST = tf.config.list_physical_devices('GPU')
SMALL_MACHINE = ( len(GPU_LIST) == 0 )
# TODO: check to see if nvidia cuda is installed
#  import os / os.system( 'nvidia-smi' ) or  os.system( 'nvcc --Version )

EPOCHS = 50
BUFFER_SIZE = 60_000
BATCH_SIZE = 256
NOISE_DIM = 100

FALSE_RATE = tf.constant( FALSE_PERCENT / 100. )
POPUP_WAIT_TIME = 1 #30

if SMALL_MACHINE:
    EPOCHS = 5
    BUFFER_SIZE = 3_000
    BATCH_SIZE = 32

print("\n\nConfig:")
print("EPOCHS=",EPOCHS)
print("BUFFER_SIZE=",BUFFER_SIZE)
print("BATCH_SIZE=",BATCH_SIZE)
print("FALSE_PERCENT=",FALSE_PERCENT)
print("\n\n")

########################################################################################################################

# transformations which can make the image 'not a number', depends on digit

# stripe = create vertical black stripe(s) covering part of the image
# flip horz = flip on the x axis
# flip vert = flip on the y axis
# rotate 90 = rotate 90 degrees clockwise
# rotate 180 = rotate 180 degrees clockwise
# rotate 270 = rotate 270 degrees clockwise

TRANSFORM_DESCRIPTION = ['Stripe','flipHorz','flipVert','rotate+90','rotate+180','rotate+270']
OPTION_COUNT = len(TRANSFORM_DESCRIPTION)

DIGIT_TRANSFORMS = [
    [ True, False, False, False, False, False],       # zero
    [ True, False, False, True, False, True],         # one
    [ True, False, False, True, False, True],         # two
    [ True, True, False, True, True, True],           # three
    [ True, True, True, True, True, True],            # four

    [ True, False, False, True, False, True],         # five
    [ True, True, True, True, False, True],           # six
    [ True, True, True, True, True, True],            # seven
    [ True, False, False, True, False, True],         # eight
    [ True, True, True, True, False, True],           # nine
]

DIGIT_VALID_TRANSFORMS = [ [-1] * ( 1 + OPTION_COUNT ) for index in range(10) ]

for digit in range(0,10):
    index = 0
    for action in range(0,OPTION_COUNT):
        if DIGIT_TRANSFORMS[digit][action]:
            index = index + 1
            DIGIT_VALID_TRANSFORMS[digit][index] = action
    DIGIT_VALID_TRANSFORMS[digit][0] = index

DIGIT_VALID_TRANSFORMS = tf.constant( DIGIT_VALID_TRANSFORMS )
# print('DIGIT_VALID_TRANSFORMS=',DIGIT_VALID_TRANSFORMS)


BLACK_STRIPE = tf.concat( [tf.ones( (11,) ), tf.zeros( (6,) ), tf.ones( (11) )], axis=0 )
# print('BLACK_STRIPE=',BLACK_STRIPE)

@tf.function
def add_stripes(image):
    r"""Starts and ends with (28, 28, 1).  Multiply matrix by vector to set some columns to zero."""

    work = tf.squeeze( image, axis=-1)
    # tf.print('\n\nwork no stripe=',work,summarize=-1)
    work = work * BLACK_STRIPE
    # tf.print('work with stripe=',work,summarize=-1)
    work = tf.expand_dims( work, axis=-1 )
    # tf.print('work 2=',work,summarize=-1)

    return work

@tf.function
def make_some_images_false( image, digit ):
    r"""For some percentage of records,
    apply a transform which makes the image a 'false' number."""

    if tf.random.uniform( () ) >= FALSE_RATE:
        return (image,True)

    # tf.print('\ndigit=',digit)
    options = DIGIT_VALID_TRANSFORMS[ digit ]
    # tf.print('options=',options)
    action  = options[ 1 + tf.random.uniform(  (), maxval=options[0], dtype=tf.int32 ) ]
    # tf.print('action=',action)

    tf.switch_case( action,{
        0: ( lambda: add_stripes( image ) ),
        1: ( lambda: tf.image.flip_left_right( image ) ),
        2: ( lambda: tf.image.flip_up_down( image ) ),
        3: ( lambda: tf.image.rot90( image ) ),
        4: ( lambda: tf.image.rot90( image, k=2 ) ),
        5: ( lambda: tf.image.rot90( image, k=3 ) )
    } )

    return (image,False)


########################################################################################################################

# load MNIST handwritten digits dataset
(digit_images, digit_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = digit_images.reshape(digit_images.shape[0], 28, 28, 1).astype('float32')

# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
@tf.function
def normalize_image( image, label ):
    image = (image - 127.5) / 127.5
    return image, label

image_digit_set = ( train_images, digit_labels )

# Batch and shuffle the data, reduce size if necessary ( eg. take() )
#       Digits are included for mapping, but become bools after mapping
#       YES!  We shuffle twice, once before transform, once after.
#       NOTE: normalize AFTER applying image falsify
train_dataset = tf.data.Dataset.from_tensor_slices( image_digit_set ).\
    take(BUFFER_SIZE). \
    map( make_some_images_false ). \
    map( normalize_image ).\
    shuffle(BUFFER_SIZE).\
    batch(BATCH_SIZE)


def make_generator_model():

    inputs = x = tf.keras.layers.Input(shape=[NOISE_DIM,])

    x = tf.keras.layers.Dense( 5*5*256, use_bias=False, activation='selu')(x)
    x = tf.keras.layers.Reshape((5, 5, 256))(x)
    assert x.shape == (None, 5,5, 256)  # NOTE: None is the batch size

    x = tf.keras.layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False, activation='selu')(x)
    assert x.shape == (None, 5,5, 128)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False, activation='selu')(x)
    assert x.shape == (None, 10,10, 64)
    x = tflt.crop_layer( 1,1,8,8 ) (x)
    assert x.shape == (None, 8,8, 64)

    x = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same', use_bias=False, activation='selu')(x)
    assert x.shape == (None, 16,16, 32)
    x = tflt.crop_layer( 1,1,14,14 ) (x)
    assert x.shape == (None, 14,14, 32)

    outputs = x = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding='same', use_bias=False, activation='tanh')(x)
    assert x.shape == (None, 28,28, 1)

    return tf.keras.Model(inputs=inputs, outputs=outputs )

generator = make_generator_model()

noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)

########################################################################################################################

def make_discriminator_model():

    inputs = x = tf.keras.layers.Input(shape=[28,28,1])
    assert x.shape == (None, 28, 28, 1)

    x = tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='selu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    assert x.shape == (None, 14, 14, 32)

    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='selu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    assert x.shape == (None, 7, 7, 64)

    x = tf.keras.layers.Conv2D(128, 2, strides=2, padding='same', activation='selu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    assert x.shape == (None, 4,4, 128)

    x = tf.keras.layers.Conv2D(256, 2, strides=2, padding='same', activation='selu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    assert x.shape == (None, 2,2, 256)

    x = tf.keras.layers.Flatten()(x)
    outputs = x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs )


def discriminator_loss(real_output, real_labels, fake_output):
    real_loss = cross_entropy( real_labels, real_output)
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
save_image_seed = tf.random.normal( [num_examples_to_generate, NOISE_DIM], seed=sample_seed )

discriminator.compile()
generator.compile()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step( train_dataset ):

    ( images, labels ) = train_dataset
    # TODO: add labels to training/fit

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, labels, fake_output)

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
            train_step(image_batch)

        total_time = time.time()-start

        # Produce images for the GIF as you go
        generate_and_save_images( generator, latest_epoch, save_image_seed )

        ckpt_manager.save()
        print ('Time for epoch {} is {} sec'.format(latest_epoch, total_time))

    # # Generate after the final epoch
    # generate_and_save_images( generator, latest_epoch, seed )

    return


def generate_and_save_images( model, epoch, test_input ):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    # display.clear_output(wait=True)

    global IMAGE_FOLDER
    predictions = model(test_input, training=False)

    plt.close()
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle( SCRIPT_TITLE, fontsize=16)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    image_path = IMAGE_FOLDER + '/image_at_epoch_{:04d}.png'.format(epoch)
    plt.savefig(image_path)

    plt.show()
    plt.pause(POPUP_WAIT_TIME)
    return

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
