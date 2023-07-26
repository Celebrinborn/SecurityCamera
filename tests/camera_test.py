from camera.camera import Camera

import logging
import numpy as np
import cv2
import pytest

# Mocks
class MockVideoCapture:
    def __init__(self, url):
        self.url = url
        self.frame = np.zeros((100, 100, 3), np.uint8)

    def isOpened(self):
        return True
    
    def read(self):
        return True, self.frame

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
        frame = next(camera.GetFrame())
        assert (frame == np.zeros((100, 100, 3), np.uint8)).all()