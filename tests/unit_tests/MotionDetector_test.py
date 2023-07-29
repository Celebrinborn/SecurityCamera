import cv2
import numpy as np
import pytest

from camera.MotionDetector import MotionDetector


@pytest.fixture
def motion_detector():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    motion_detector = MotionDetector(frame)
    return motion_detector
