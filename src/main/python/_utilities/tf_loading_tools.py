

import cv2
import numpy as np

#
#   load videos as an array of image frames
def load_video(path, max_frames=0):
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