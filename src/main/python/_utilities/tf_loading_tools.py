#
#   Utilities related to loading and saving information to the file system
#

import cv2
import numpy as np


def load_video(path, max_frames=0):
    """load videos as an array of image frames"""
    cap = cv2.VideoCapture(path)
    print(
        "Video=", path,
        " width=", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        " height=", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        " nframes=", cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def EpochInfo():
    """class for storing epoch information"""


def epoch_save():
    """When saving to a checkpoint,
    also save an 'epoch' file with epoch count and statistical history
    """
    return

def epoch_load():
    """When reading from a checkpoint,
    also read an 'epoch' file with epoch count and statistical history
    """
    return
