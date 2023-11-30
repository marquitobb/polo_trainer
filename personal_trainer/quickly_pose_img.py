from typing import Union
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

from personal_trainer.tools import n_resize


def main(img_path: Union[str, Path]) -> None:
    """
    Find 33 landmarks in an image using mediapipe pose.

    Args:
        img_path: Image path.

    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        # Read test_image
        img = cv2.imread(str(img_path))
        img = n_resize(img)

        # Transform BGR into RGB
        img_rgb = img[:, :, ::-1]

        # Pose landmarks
        results = pose.process(img_rgb)

        # Draw
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img_rgb,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Image', img_rgb[:, :, ::-1])  # convert to RGB to use correctly in opencv
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':  # pragma: no cover
    IMG_PATH = './data/image.jpg'
    main(IMG_PATH)
