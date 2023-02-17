from camera.filemanager import FileManager
import os
import numpy as np
import asyncio
import time
from queue import Queue
from dataclasses import dataclass
import cv2
import datetime
import mock

class MockVideoWriter():
    def _getVideoWriter(filename):
        return MockVideoWriter()
    def write(self, frame):
        print('writing frame')
    def release(self):
        print('closing mock writer')

def test_Filemanager():
    _width, _height, _channels, _fps = (1280, 720, 3, 25)
    framequeue = Queue()

    print('starting filemanager')
    FileManager.Start_Filemanager_thread(framequeue, _width, _height, _fps, '.', MockVideoWriter._getVideoWriter)

    print('adding itmes')
    for i in range(42):
        framequeue.put(np.random.rand(_width, _height, _channels))
    print('finshed')

def test_time_to_folder_name():
    # Create a test timestamp
    timestamp = datetime.datetime(2022, 12, 31, 23, 59, 59)

    # Call the function to format the timestamp
    folder_name = FileManager.time_to_folder_name(timestamp)

    # Check if the function returns the correct folder name
    assert folder_name == "20221231_23"