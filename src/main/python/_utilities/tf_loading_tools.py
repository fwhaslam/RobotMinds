#
#   Utilities related to loading and saving information to the file system
#
import os
import cv2
import builtins
import numpy as np
import configparser
import tensorflow.python.framework.errors_impl as errimp


# def try_delete( file_path ):
#     r"""Attempt to delete local files.  Can be used to remove old checkpoints."""
#     if os.path.exists( file_path) :
#         os.remove (file_path )
#     else:
#         print("Cannot find file to remove ["+file_path+"]" )
#

def try_load_weights( ckpt_file, model ) :
    r"""Try to load weights for a model from a checkpoint.
    Restore default vales on failure.
    """
    temp = model.get_weights()
    if os.path.exists(ckpt_file+'.index') :
        try:
            model.load_weights( ckpt_file )
            return True
        except ( builtins.ValueError, errimp.NotFoundError ) as e1:
            model.set_weights(temp)
            print("Old model does not match new model, not loading weights")
            # removing default checkpoint files
            # try_delete( "checkpoint" )
            # try_delete( ckpt_file + ".index" )
            # try_delete( ckpt_file + ".data-00000-of-00001" )
            return False
        except Exception as e2:
            print('Failed to load weights: '+ str(e2) )
            return False

def try_save_weights( ckpt_file, model ) :
    r"""Try to load weights for a model from a checkpoint.
    Restore default vales on failure.
    """
    temp = model.get_weights()
    if os.path.exists(ckpt_file+'.index') :
        try:
            model.load_weights( ckpt_file )
        except ( builtins.ValueError, errimp.NotFoundError ) as e1:
            model.set_weights(temp)
            print("Old model does not match new model, not loading weights")
            # removing default checkpoint files
            # try_delete( "checkpoint" )
            # try_delete( ckpt_file + ".index" )
            # try_delete( ckpt_file + ".data-00000-of-00001" )
        except Exception as e2:
            print('Failed to load weights: '+ str(e2) )


def load_video(path, max_frames=0, details=False):
    """load videos as an array of image frames"""

    if (details):
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
    r"""class for storing epoch information"""


def epoch_save():
    r"""When saving to a checkpoint also save a config file with
    common properties such as current epoch.
    """
    return

def epoch_load():
    r"""When reading from a checkpoint also read a config file with
    common properties such as current epoch.
    """
    return
