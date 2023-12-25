import warnings
import cv2
import numpy as np
import pytest
from camera.frame import Frame
from typing import Generator, Optional

from camera.MotionDetector import MotionDetector


@pytest.fixture(scope='module')
def GetImage():
    image = np.random.randint(0, 256, (480, 640, 3), dtype = np.uint8)
    yield image

@pytest.fixture(scope='module')
def GetFrame(GetImage) -> Generator[Optional[Frame], None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        image = GetImage

        frame = Frame(image)
        yield frame
@pytest.fixture
def GetMotionDetector() -> Generator[MotionDetector, None, None]:
    motion_detector = MotionDetector()
    yield motion_detector

def test_preprocess_frame(GetMotionDetector, GetFrame):
    frame:Frame = GetFrame
    motion_detector:MotionDetector = GetMotionDetector
    new_frame = motion_detector._preprocess_frame(frame)
    assert new_frame.shape != frame.shape
    assert len(new_frame.shape) == 2, 'preprocessing changes to black and white so only 1 channel instead of 3'
    assert new_frame == frame, 'verify frame equality still works'
