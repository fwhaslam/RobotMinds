import tensorflow as tf

import matplotlib.pyplot as plt

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


def generate_images(model_g, model_f, test_input):
    r"""Display a grid of images showing progress.
    Images are original, predicted, cycled, and 'same'
    ( same = original through the same model eg. horse to horse )."""

    prediction = model_g(test_input)
    cycled = model_f(prediction)
    same = model_f(test_input)

    plt.close()     # close previous image
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0], cycled[0], same[0]]
    title = ['Input Image', 'Predicted Image', 'Cycled Image', 'Same Image']

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.3 + 0.3)
        plt.axis('off')

    plt.show()
    plt.pause(GPU_REST_SECONDS)

