from camera.filemanager import FileManager
import os
import numpy as np
import asyncio
import time
from queue import Queue
from dataclasses import dataclass
import cv2

def test_WriteFrame():
    test_width = 1280
    test_height= 720
    test_depth = 3
    frames = [np.random.rand(test_width, test_height,test_depth), np.random.rand(test_height,test_width,test_depth), np.random.rand(test_height,test_width,test_depth)]
    with FileManager(test_width, test_height, 25) as fileManager:
        for frame in frames:
            fileManager.WriteFrame(frame)
