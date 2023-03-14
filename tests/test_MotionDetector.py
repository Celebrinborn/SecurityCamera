import cv2
import numpy as np
import pytest

from camera.MotionDetector import MotionDetector


@pytest.fixture
def motion_detector():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    motion_detector = MotionDetector(frame)
    return motion_detector


def test_motion_detection_init(motion_detector):
    assert motion_detector._prev_frame is not None, "Prev frame is None"

def test_detect_motion_prev_frame_overwrite(motion_detector):
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8)
    
    motion_detector.detect_motion(frame1)
    prev_frame1 = motion_detector._prev_frame.copy()

    motion_detector.detect_motion(frame2)
    prev_frame2 = motion_detector._prev_frame.copy()

    assert not np.array_equal(prev_frame1, prev_frame2), "prev_frame is not overwritten when detect_motion is called"