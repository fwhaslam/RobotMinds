#
#   Tensorflow Cycling Generative Adversarial Network.
#
#   This CycleGAN framework will provide the following:
#       training, checkpoints, and sampling
#
#   It needs the following:
#       generator models, datasets
#
#   Future improvements:
#       configuration control
#       sampling at different epochs
#
#   Code tested with:
#       Tensorflow 2.10.0 / Cuda 11.7 / CudaNN 8.4 / VC_Redist 2019+
#
#   NOTE: all this nonsense with 'class/self' explains why module is that natural unit of organization
#

import sys
sys.path.append('..')

import tensorflow as tf
import tensorflow_datasets as tfds

import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

import re
import os
from os.path import *
import configparser
from collections.abc import Callable


from _utilities.tf_cyclegan_tools import *
from tf_layer_tools import *


class cyclegan_runner:

    def __init__(self,
                 train_first,train_second,test_first,test_second,
                 generator_first,generator_second,
                 epochs,checkpoint_root):
        self.train_first = train_first
        self.train_second = train_second
        self.test_first = test_first
        self.test_second = test_second
        self.checkpoint_root = checkpoint_root
        self.generator_g = generator_first
        self.generator_f = generator_second
        self.EPOCHS = epochs
        if len(sys.argv)>2: self.EPOCHS = int( sys.argv[2] )

########################################################################################################################
#   Functional Api version of UNet generator from tensorflow_examples.models.pix2pix
#       see: https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
#

    def downsample(self,filters, size, apply_norm=True):
        """Downsamples an input.
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
            result.add(InstanceNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result


    def discriminator(self):
        r"""PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
        Args:
          norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
          target: Bool, indicating whether target image is an input or not.
        Returns:
          Discriminator model
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        x = inp

        down1 = self.downsample( 64, 4, False )(x)  # (bs, 128, 128, 64)
        down2 = self.downsample( 128, 4 )(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample( 256, 4) (down2)  # (bs, 32, 32, 256)

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

########################################################################################################################

    def discriminator_loss(self,real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * self.DISC_LOSS_FACTOR

    def generator_loss(self,generated):
        return self.loss_obj(tf.ones_like(generated), generated) * self.GEN_LOSS_FACTOR

    def calc_cycle_loss(self,real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return loss1 * self.CYCLE_LOSS_FACTOR

    def identity_loss(self,real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return loss * self.IDENT_LOSS_FACTOR


########################################################################################################################
# Train

    @tf.function
    def train_step(self,real_x, real_y):
        r"""persistent is set to True because the tape is used more than
        once to calculate the gradients."""
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

###########################################################################################################

    def store_config( self, folder, epoch_count ):
        config = configparser.RawConfigParser()
        config.add_section('cyclegan')
        config.set( 'cyclegan', 'epoch', '{}'.format(epoch_count) )

        path = folder+'/configfile.ini'
        if not exists(path): open(path, 'a').close()

        fp = open( folder+'/configfile.ini', 'w' )
        config.write( fp )
        fp.close()

    def store_samples( self, folder, image_array, gen_first, gen_second, epoch_count ):
        test_set = image_array.cache().take(10).repeat()
        for (index,base_image) in zip( range(10), test_set ):
            prediction = gen_first(base_image)
            cycled = gen_second(prediction)
            same = gen_second(base_image)

            # axis=0 is vertical, axis=1 is horizontal
            base_and_predict = tf.concat( [tf.squeeze(base_image),tf.squeeze(prediction)], axis=1 )
            cycled_and_same = tf.concat( [tf.squeeze(cycled),tf.squeeze(same)], axis=1 )
            all_images = tf.concat( [base_and_predict,cycled_and_same],axis=0)
            tf.keras.utils.save_img( folder + "/img{}a_{}.png".format(index,epoch_count), tf.squeeze( all_images ))

    def perform_all_saves( self, epoch_count, ckpt_manager, ckpt, delta_secs ):
        r"""Store simple metrics, including current epoch
        convert and store first 10 training images ( original, altered, cycled, same )"""

        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch_count, ckpt_save_path))
        print ('Time taken for epoch {} is {} sec\n'.format(epoch_count,delta_secs))
        if ((epoch_count%50)!=0):
            return

        # periodic checkpoint save, image samples
        folder = self.checkpoint_root + '/e{}'.format( epoch_count )
        ckpt.write( folder+'/ckpt-{}'.format( epoch_count) ) # %50 backup of checkpoint

        samples_folder = folder + '/samples'
        if not exists(samples_folder): os.makedirs(samples_folder)
        self.store_config( samples_folder, epoch_count )
        self.store_samples( samples_folder, self.test_first, self.generator_f, self.generator_f, epoch_count )


    def load_with_epoch( self, ckpt, ckpt_manager ):
        r"""if a checkpoint exists, restore the latest checkpoint."""
        checkpoint_name = ckpt_manager.latest_checkpoint
        if checkpoint_name:
            tf.print("\n\nLATEST_CHECKPOINT =",checkpoint_name)
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
            return int( re.findall(r'\d+', str.split(checkpoint_name,'/')[-1] )[0] )
        else:
            return 0

    def display_config(self):
        print('\n\n')
        print('config: checkpoint_root =',self.checkpoint_root)
        print('config: EPOCHS=',self.EPOCHS)
        print('config: GEN_LOSS_FACTOR=',self.GEN_LOSS_FACTOR)
        print('config: DISC_LOSS_FACTOR=',self.DISC_LOSS_FACTOR)
        print('config: CYCLE_LOSS_FACTOR=',self.CYCLE_LOSS_FACTOR)
        print('config: IDENT_LOSS_FACTOR=',self.IDENT_LOSS_FACTOR)
        print('\n\n')

########################################################################################################################
#       Runner Functionality

    def run(self, base_loss=10.):

        plt.interactive(True)

        # Constant Parameters
        BUFFER_SIZE = 1000
        BATCH_SIZE = 1
        # OUTPUT_CHANNELS = 3

        self.GEN_LOSS_FACTOR = 1.0       # 1.0 = generator loss ?
        self.DISC_LOSS_FACTOR = 0.5      # 0.5 = discriminator
        self.CYCLE_LOSS_FACTOR = base_loss * 1.0    # LAMBDA * 1.0 = full cycle
        self.IDENT_LOSS_FACTOR = base_loss * 0.5    # LAMBDA * 0.5 = real to same

        self.display_config()

        self.train_first = self.train_first.cache().\
            map(preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE).\
            shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        self.train_second = self.train_second.cache().\
            map(preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE).\
            shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        # no shuffle, we want the same samples at the front end every time
        self.test_first = self.test_first.\
            map(preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(BATCH_SIZE)
        self.test_second = self.test_second.\
            map(preprocess_image_test, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(BATCH_SIZE)


        # select images for demonstrating progress :: Display alongside generator images
        # sample_first = next(iter(self.train_first))
        # sample_second = next(iter(self.train_second))
        first_loop = iter(RepeatLoop(self.train_first))

########################################################################################################################

        self.discriminator_x = self.discriminator()
        self.discriminator_y = self.discriminator()
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

########################################################################################################################
# Checkpoints

        ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                   generator_f=self.generator_f,
                                   discriminator_x=self.discriminator_x,
                                   discriminator_y=self.discriminator_y,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer)

        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            self.checkpoint_root +'/train',
            max_to_keep=10 )

        # if a checkpoint exists, restore the latest checkpoint.
        latest_epoch = self.load_with_epoch( ckpt, ckpt_manager )
        print('latest_epoch=',latest_epoch)

########################################################################################################################
# MAIN LOOP

        for epoch_step in range(self.EPOCHS):
            latest_epoch += 1
            print('\nStarting Epoch =',latest_epoch)

            start = time.time()

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((self.train_first, self.train_second)):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print ('.', end='')
                n += 1
            clear_output(wait=True)
            # Using a consistent image (sample_first) so that the progress of the model
            # is clearly visible.
            next_sample = next( first_loop )
            generate_images(self.generator_g, self.generator_f, next_sample)

            self.perform_all_saves( latest_epoch, ckpt_manager, ckpt, time.time()-start )

        print('\n\nEND OF PROCESS latest_epoch =',latest_epoch,'\n\n')
