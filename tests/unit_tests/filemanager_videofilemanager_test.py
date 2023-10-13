import pytest
from pathlib import Path
from collections import namedtuple
from queue import Queue
import cv2
import os
import time
import threading
import numpy as np
import sys
from camera.frame import Frame
from camera.resolution import Resolution

# Suppose your class file is named `file_manager.py`
from camera.filemanager import VideoFileManager, File_Pair, FileManager

@pytest.fixture
def random_frame():
    return Frame((np.random.rand(480, 640, 3) * 255).astype(np.uint8))

@pytest.fixture
def video_file_manager(tmp_path):
    resolution = Resolution(width=640, height=480)
    file_manager = FileManager(root_folder=tmp_path, max_dir_size_bytes=1000)
    video_file_manager = VideoFileManager(root_video_file_location=tmp_path, resolution=resolution, fps=30, file_manager=file_manager)
    return video_file_manager


def test_start(video_file_manager:VideoFileManager):
    assert isinstance(video_file_manager._video_file_manager_thread, threading.Thread)

def test_stop(video_file_manager:VideoFileManager):
    video_file_manager.Stop(blocking=True)
    assert video_file_manager._kill_video_file_manager_thread.is_set()

def test_open_video(video_file_manager:VideoFileManager):
    video_filename_path = video_file_manager.create_video_file_name(video_file_manager._root_video_file_location, time.time())
    videowriter, file_pair = video_file_manager._open_video_writer(video_filename_path)
    assert isinstance(videowriter, cv2.VideoWriter)
    assert isinstance(file_pair, File_Pair)

def test_close_video(video_file_manager:VideoFileManager):
    video_filename_path = video_file_manager.create_video_file_name(video_file_manager._root_video_file_location, time.time())
    videowriter, _ = video_file_manager._open_video_writer(video_filename_path)
    video_file_manager._close_video_writer(videowriter)

def test_create_video_file_name(video_file_manager:VideoFileManager):
    file_path = video_file_manager.create_video_file_name(video_file_manager._root_video_file_location, time.time())
    assert isinstance(file_path, Path)
    assert file_path.parent == video_file_manager._root_video_file_location
    assert file_path.suffix in ['.mp4v', '.avi']

def test_video_file_manager_thread_function(video_file_manager:VideoFileManager, random_frame):
    # Given

    assert video_file_manager._frame_counter == 0

    # When
    
    # create a random frame
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    # push the frame to the queue
    video_file_manager.GetQueue().put(random_frame)

    # run the write function once
    video_file_manager._write(video_file_manager.GetQueue().get())

    # Then    
    # check if the frame counter has been incremented
    assert video_file_manager._frame_counter == 1
