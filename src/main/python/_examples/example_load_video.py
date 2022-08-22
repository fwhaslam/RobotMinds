#
#   Code copied from keras example:
#       https://keras.io/examples/vision/video_transformers/
#
#   Modified to only load one frame and display it.
#
#   Using oxford tv video dataset:
#       https://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html
#

import sys
sys.path.append('..')

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from _utilities.tf_loading_tools import *


print("Test Video functions using Oxford Human Interaction dataset")

root_path = os.path.expanduser( 'd:/Datasets/OxfordHumanInteractions/tv_human_interactions_videos' )

video_files=[root_path + '/' + f for f in os.listdir(root_path)]

IMG_WIDTH = 320     # using 16:9 image ratio, the videos are usually about twice this (width,height)
IMG_HEIGHT = 180

# def load_video(path, max_frames=0):
#     cap = cv2.VideoCapture(path)
#     print(
#         "Video=", path,
#         " width=", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
#         " height=", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
#         " nframes=", cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     )
#     frames = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = frame[:, :, [2, 1, 0]]
#             frames.append(frame)
#
#             if len(frames) == max_frames:
#                 break
#     finally:
#         cap.release()
#     return np.array(frames)


for idx, path in enumerate(video_files):
    # Gather all its frames and add a batch dimension.
    frames = load_video( os.path.join(root_path, path), 1 )
    print("shape[",path,"] = ",tf.shape(frames))

    new_size = tf.image.resize( frames[0], (IMG_HEIGHT, IMG_WIDTH) )
    draw_this = new_size / 256.

    # print("draw_this = ",draw_this)
    print("Shape[draw_this]=",tf.shape(draw_this))

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow( draw_this )
    # plt.imshow( frames[0], cmap=plt.cm.binary)
    plt.show()
    plt.pause(3)

    # time.sleep(12)