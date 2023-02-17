import logging
logger = logging.getLogger(__name__)
import cv2
import numpy as np
import os

import typing

class Camera:
    _camera: cv2.VideoCapture
    _camera_name:str
    _camera_url: str
    _max_fps: int

    prevFrame: np.ndarray
    currentFrame: np.ndarray

    def __init__(self, camera_name:str, camera_url:str, max_fps:int, cv2_module: typing.Type[cv2.VideoCapture]=cv2.VideoCapture) -> None:
        self._camera_name = camera_name
        self._camera_url = camera_url
        self._max_fps = max_fps
        self._cv2 = cv2_module
        logger.debug('running Camera class init')
        
    # Define the __enter__ method for the Camera class
    def __enter__(self):
        # using the dependency injection approach to assist with testing
        self._camera = self._cv2(self._camera_url)
        
        # Read the current frame from the camera object and assign it to a variable
        # to ensure that prevFrame has something to populate later in the application
        _, self.currentFrame = self._camera.read()
        
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self._camera.release()
        logger.debug('running Camera class exit')

    def GetFrame(self) -> np.ndarray:
        logger.debug('running Camera class getframe')
        while True: #self._camera == True:
            ret, newFrame = self._camera.read()
            self.prevFrame = self.currentFrame.copy()
            self.currentFrame = newFrame
            yield self.currentFrame
        return self.currentFrame



if __name__ == '__main__':
    import sys
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)
    logger.critical('starting camera.py module AS MAIN')