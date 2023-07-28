from camera.camera import Camera

import logging
import numpy as np
import cv2
import pytest
from camera.frame import Frame

# Mocks
class MockVideoCapture:
    def __init__(self, url):
        self.url = url

    def isOpened(self):
        return True
    
    def read(self):
        # Generate a different frame each time
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        pass


# Tests
def test_Camera_ManualOpenClose():
    camera = Camera("TestCamera", "test://", 30, cv2_module=MockVideoCapture)
    camera.close()
    assert True  # if the test didn't throw an exception, it passed

def test_Camera_ContextManager():
    with Camera("TestCamera", "test://", 30, cv2_module=MockVideoCapture) as camera:
        pass
    assert True  # if the test didn't throw an exception, it passed

def test_Camera_GetFrame():
    with Camera("TestCamera", "test://", 30, cv2_module=MockVideoCapture) as camera:
        frame:Frame
        frame_generator = camera.GetFrame()
        frame = next(frame_generator)
        assert isinstance(frame, Frame), 'frame is not a type Frame'
        assert np.all(frame == np.zeros((100, 100, 3), dtype=np.uint8)) # this ends up being an array of Trues which will have an all function

        for i in range(15):
            frame = next(frame_generator)
            assert isinstance(frame, Frame), f'frame is not a type Frame on call {i}'