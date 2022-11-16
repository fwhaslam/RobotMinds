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

import os
import re
import time
import glob

import tensorflow as tf
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tensorflow.keras import layers

plt.ion()

# model and samples directory
checkpoint_dir = './guided_dcgan_ckpt/v01'
train_folder = checkpoint_dir+"/train"
image_folder = checkpoint_dir+"/samples"

# am I running this under via CPU on my laptop ( eg. small machine ) ?
# SMALL_MACHINE = ( tf.test.is_gpu_available() == False )   # deprecated method
GPU_LIST = tf.config.list_physical_devices('GPU')
SMALL_MACHINE = ( len(GPU_LIST) == 0 )
# TODO: check to see if nvidia cuda is intalled
#  import os / os.system( 'nvidia-smi' ) or  os.system( 'nvcc --Version )

EPOCHS = 50
BUFFER_SIZE = 60_000
BATCH_SIZE = 256

if SMALL_MACHINE:
    EPOCHS = 5
    BUFFER_SIZE = 3_000
    BATCH_SIZE = 32

print("\n\nConfig:")
print("EPOCHS=",EPOCHS)
print("BUFFER_SIZE=",BUFFER_SIZE)
print("BATCH_SIZE=",BATCH_SIZE)
print("\n\n")

########################################################################################################################

# load MNIST handwritten digits dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


# Batch and shuffle the data, reduce size if necessary ( eg. take() )
train_dataset = tf.data.Dataset.\
    from_tensor_slices(train_images).\
    take(BUFFER_SIZE).\
    shuffle(BUFFER_SIZE).\
    batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

########################################################################################################################

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

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

ckpt_manager = tf.train.CheckpointManager( ckpt, train_folder, max_to_keep=10 )

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
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

########################################################################################################################

noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

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
            train_step(image_batch)

        total_time = time.time()-start

        # Produce images for the GIF as you go
        generate_and_save_images( generator, latest_epoch, seed )

        ckpt_manager.save()
        print ('Time for epoch {} is {} sec'.format(latest_epoch, total_time))

    # Generate after the final epoch
    generate_and_save_images( generator, latest_epoch, seed )


def generate_and_save_images( model, epoch, test_input ):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    # display.clear_output(wait=True)

    global image_folder
    predictions = model(test_input, training=False)

    plt.close()
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    image_path = image_folder+'/image_at_epoch_{:04d}.png'.format(epoch)
    plt.savefig(image_path)
    plt.show()
    plt.pause(30)

train( train_dataset, EPOCHS )

########################################################################################################################

# Use imageio to create an animated gif using the images saved during training.
anim_file = 'guided_dcgan_animation.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(image_folder+'/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
