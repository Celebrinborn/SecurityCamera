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

from camera.filemanager import VideoFileManager, VideoRecord, FileManager

import unittest.mock as mock

# Mock for VideoRecord class
class MockVideoRecord:
    def __init__(self, video_file_path: Path, batch_size: int = 10*60*5):
        self.video_file_path = video_file_path
        self.batch_size = batch_size
        self._cache = pd.DataFrame()  # Empty DataFrame, replace with your desired initial state
        self.row_curser = 0

    def write_line(self, frame, frame_counter_int: int) -> None:
        pass  # Mock method does nothing

    def delete(self) -> bool:
        return True  # Mock method always returns True

    def __eq__(self, other):
        return isinstance(other, MockVideoRecord) and self.video_file_path == other.video_file_path

    def __str__(self):
        return f'{self.video_file_path.name}'

    def file_size(self):
        return 1000  # Mock method always returns 1000, replace with your desired return value

    def create_cache(self):
        pass  # Mock method does nothing

    def flush_cache(self):
        pass  # Mock method does nothing


# Mock for FileManager class
class MockFileManager:
    def __init__(self, root_folder: Path, max_dir_size_bytes: int):
        self._video_records = []
        self.sql_manager = mock.Mock()  # Mocking SQLManager instance
        self.folder_path = root_folder
        self.max_dir_size = max_dir_size_bytes

    def scan(self):
        pass  # Mock method does nothing

    def add_file(self, video_record: MockVideoRecord):
        self._video_records.append(video_record)

    def get_record_file_size(self, video_record: MockVideoRecord) -> int:
        return video_record.file_size()

    def get_total_dir_size(self) -> int:
        return sum(video_record.file_size() for video_record in self._video_records)

    def delete_record(self, video_record: MockVideoRecord) -> bool:
        if video_record in self._video_records:
            self._video_records.remove(video_record)
        return True  # Mock method always returns True

    def _get_oldest_record(self) -> MockVideoRecord:
        return None  # Mock method always returns None, replace with your desired return value

    def _get_newest_record(self) -> MockVideoRecord:
        return None  # Mock method always returns None, replace with your desired return value

@pytest.fixture
def random_frame():
    return Frame((np.random.rand(480, 640, 3) * 255).astype(np.uint8))

@pytest.fixture
def random_4k_frame():
    return (np.random.rand(2160, 3840, 3) * 255).astype(np.uint8)

@pytest.fixture
def video_file_manager(tmp_path):
    mock_file_manager = MockFileManager(root_folder=tmp_path, max_dir_size_bytes=1000)
    resolution = Resolution(width=640, height=480)
    video_file_manager = VideoFileManager(\
        root_video_file_location=tmp_path, resolution=resolution, \
        fps=30, file_manager=mock_file_manager, max_video_length_frame_seconds=10*30, init_without_start = True)
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
    assert isinstance(file_pair, VideoRecord)

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

def test_scale_frame(video_file_manager:VideoFileManager, random_4k_frame):
    # Given
    # 4k frame
    frame = random_4k_frame
    
    # resolution is 480p
    resolution = Resolution(width=640, height=480)

    # When
    scaled_frame = video_file_manager.scale(frame, resolution)

    # Then
    # verify frame identity is preserved
    assert scaled_frame == frame, f'frame identity is not preserved'

    # verify frame shape is correct at 640x480x3
    assert scaled_frame.shape == (480, 640, 3), f'frame shape is not 480p, instead is {scaled_frame.shape}'

