import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

IMG_WIDTH = 256
IMG_HEIGHT = 256
GPU_REST_SECONDS = 20

preferred_resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
preferred_antialias = True

class RepeatLoop:
    r"""This wraps an iterator and repeats the content forever."""
    def __init__(self,looper):
        self.looper = looper

    def __iter__(self):
        self.loop = iter(self.looper)
        return self

    def __next__(self):
        try:
            return next(self.loop)
        except StopIteration:
            self.loop = iter(self.looper)
            return next(self.loop)


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


def normalize(image):
    r"""normalizing the images to range [-1, 1]"""
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    r"""Expand and randomly crop and mirror image to produce image variations."""
    image = tf.image.resize(image, [286, 286], method=preferred_resize_method, antialias=preferred_antialias )
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_image_train(image, label='can ignore'):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label='can ignore'):
    image = normalize(image)
    return image


def display_generator_comparison( gen_g, gen_f, samp_h, samp_z):
    r"""Show some sample images."""
    plt.subplot(121)
    plt.title('Horse')
    plt.imshow(samp_h[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Horse with random jitter')
    plt.imshow(random_jitter(samp_h[0]) * 0.5 + 0.5)

    plt.subplot(121)
    plt.title('Zebra')
    plt.imshow(samp_z[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Zebra with random jitter')
    plt.imshow(random_jitter(samp_z[0]) * 0.5 + 0.5)

    to_zebra = gen_g(samp_h)
    to_horse = gen_f(samp_z)
    plt.figure(figsize=(8, 8))
    contrast = 8

    imgs = [samp_h, to_zebra, samp_z, to_horse]
    title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

    for i in range(len(imgs)):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        if i % 2 == 0:
            plt.imshow(imgs[i][0] * 0.5 + 0.5)
        else:
            plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
    plt.show()
    plt.pause(1) # show on startup
