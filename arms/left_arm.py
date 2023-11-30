from typing import Union
from pathlib import Path
import numpy as np
import cv2
from typing import List
import math
import numpy as np
import cv2
import mediapipe as mp
import pyttsx3



class PoseEstimation:
    """
    Main class for pose estimation.
    """

    def __init__(self,
                 static_image_mode=False,
                 smooth=True):
        """

        Args:
            static_image_mode:  Whether to treat the input images as a batch of static
                                and possibly unrelated images, or a video stream. See details in
            smooth: Whether to filter landmarks across different input
                    images to reduce jitter
        """

        # Basic
        self.static_image_mode = static_image_mode
        self.smooth = smooth

        # mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # Set pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.static_image_mode,
                                      smooth_landmarks=self.smooth)

        # Output
        self.results = None
        self.list_landmarks = None


    def find_pose(self,
                  img: np.ndarray,
                  draw: bool = True) -> np.ndarray:
        """
        Process pose in an image.

        Args:
            img: A three channel RGB image represented as numpy ndarray.
            draw: Whether to draw landmarks.

        Returns:
            Image.

        """
        # Pose landmarks
        img = np.ascontiguousarray(img, dtype=np.uint8)
        self.results = self.pose.process(img)

        # Draw
        if draw:
            self.mp_drawing.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)
            return img
        return img


    def get_landmarks(self, img: np.ndarray) -> List:
        """
        List of landmarks in an image.

        Args:
            img: A three channel RGB test_image represented as numpy ndarray.

        Returns:
            Landmarks list.

        """
        try:
            assert self.results.pose_landmarks is not None, 'Pose landmarks not found'
        except AssertionError as e:
            print(e)
            return []

        # Landmarks name (33 points)
        names = [self.mp_pose.PoseLandmark(i).name for i in range(33)]

        # Landmarks list
        self.list_landmarks = []
        for name, (idx, landmark) in zip(names, enumerate(self.results.pose_landmarks.landmark)):
            height, width, _ = img.shape
            x_centroid = int(landmark.x * width)
            y_centroid = int(landmark.y * height)
            self.list_landmarks.append([idx, name, x_centroid, y_centroid])
        return self.list_landmarks


    def find_angle(self, img: np.ndarray, p1: int, p2: int, p3: int, draw: bool = True) -> float:
        """
        Find an angle between two lines build with three points.

        Args:
            img: A three channel RGB test_image represented as numpy ndarray.
            p1: Point 1.
            p2: Point 2. Intersection.
            p3: Point 3.
            draw: Whether to draw landmarks.

        Returns:
            Angle.
        """
        img = np.ascontiguousarray(img, dtype=np.uint8)

        try:
            # Get the landmarks
            x1, y1 = self.list_landmarks[p1][2:]
            x2, y2 = self.list_landmarks[p2][2:]
            x3, y3 = self.list_landmarks[p3][2:]
        except (TypeError, IndexError) as e:
            print(f"Error getting landmarks: {e}")
            return 0  # or return a default value

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        xs = [x1, x2, x3]
        ys = [y1, y2, y3]
        if draw:
            for x, y in zip(xs, ys):
                cv2.circle(img, (x, y), 10, 255, cv2.FILLED)
            color = (255, 255, 255)
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            cv2.line(img, (x3, y3), (x2, y2), color, 3)
            cv2.rectangle(img, (x2 - 50, y2 + 20), (x2 + 10, y2 + 60), color, cv2.FILLED)
            cv2.putText(img, f'{angle:.0f}', (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            return angle
        return angle





def main(arm_evaluated: str) -> None:
    detector = PoseEstimation()
    count = 0
    direction = 0

    cap = cv2.VideoCapture(1)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Convert to RGB
        img = img[:, :, ::-1]

        # Resize img by n
        # img = n_resize(img, 3)
        img = cv2.resize(img, (1280, 720))

        # Detector
        img = detector.find_pose(img, False)
        detector.get_landmarks(img)

        # detect arm left
        angle = detector.find_angle(img, 11, 13, 15)

        # Percentage: consider that an angle equals 210 is not a curl bicep,
        # and an angle equals to 310 is a perfect curl bicep
        percentage = np.interp(angle, (210, 310), (0, 100))

        # Check bicep curl
        color = (255, 255, 255)
        if percentage == 100:
            color = (127, 61, 127)
            if direction == 0:  # up
                # Count half curl
                count += 0.5
                # Change direction to down
                direction = 1

        if percentage == 0:
            color = (127, 61, 127)
            if direction == 1:  # down
                # Count half curl
                count += 0.5
                # Change direction to up
                direction = 0


        # Draw bar
        height, width, _ = img.shape
        height_bar = int(height / 7)
        width_bar = int(width * 85 / 100)
        bar = np.interp(angle, (220, 310), (height_bar + 500, height_bar))

        cv2.rectangle(img,
                      (width_bar, height_bar),
                      (width_bar + 70, height_bar + 500),
                      color,
                      3)
        cv2.rectangle(img,
                      (width_bar, int(bar)),
                      (width_bar + 70, height_bar + 500),
                      color,
                      cv2.FILLED)

        cv2.putText(img,
                    f'{percentage:.0f} %',
                    (width_bar, int(height * 10 / 100)),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    color,
                    3)

        # Draw curl count
        cv2.rectangle(img,
                      (0, int(height * 80 / 100)),
                      (int(width * 15 / 100), height),
                      (127, 61, 127),
                      cv2.FILLED)
        cv2.putText(img,
                    f'{int(count)}',
                    (int(width * 3 / 100), int(height * 97 / 100)),
                    cv2.FONT_HERSHEY_PLAIN,
                    10,
                    (255, 0, 0),
                    10)

        cv2.imshow('Video', img[:, :, ::-1])  # convert to RGB to use correctly in opencv
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':   # pragma: no cover
    ARM = 'left'
    # VIDEO_PATH = DATA_PATH / 'video.mp4'
    main(ARM)
