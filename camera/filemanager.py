import logging
logger = logging.getLogger(__name__)
import cv2
import numpy as np
import os
import pathlib
from os import fspath
import datetime
from queue import Queue, LifoQueue
import typing
from typing import List, Optional
import threading
import sys
import inspect
from dataclasses import dataclass

from camera.frame import Frame

import os
import logging
import datetime
from pathlib import Path
import time

logger = logging.getLogger(__name__)

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Iterable, cast, TextIO, Optional
import json
from camera.resolution import Resolution


@dataclass
class VideoRecord:
    base_filename: str
    directory_path: Path
    video_file_path: Path
    text_file_path: Path

    def __init__(self, video_file_path:Path):
        if video_file_path.suffix not in ['.mp4', '.avi', '.mp4v', '']:
            raise ValueError(f'{video_file_path} should be mp4, avi, or txt')
        self.base_filename = video_file_path.stem
        self.directory_path = video_file_path.parent
        self.video_file_path = video_file_path
        self.text_file_path = video_file_path.with_suffix('.txt')
        
        
        # create file if it does not exist
        self.text_file_path.touch()


        if not all((self.video_file_path.exists(), self.text_file_path.exists())):
            raise FileNotFoundError
    def delete(self):
        self.text_file_path.unlink()
        self.video_file_path.unlink()

    def creation_date(self):
        return os.path.getctime(self.video_file_path)
    def write_line(self, frame:Frame, frame_counter_int:int) -> None:
        # Build the JSON data
        data = {frame_counter_int: {"creation_timestamp": frame.creation_timestamp,"guid": str(frame.guid)}}

        # Append the data to the text file
        try:
            with self.text_file_path.open('a') as text_file_handle:   # Open the file in append mode
                json.dump(data, text_file_handle)
        except TypeError as e:
            logger.exception(f'an error has occured while attempting to encode json {e}')
        except IOError as e:
            logger.exception(f'an IOError occured while attempting to write a record to {self.video_file_path}: {e}')

class FileManager:
    def __init__(self, root_folder:Path, max_dir_size_bytes:int):
        
        self.folder_path = root_folder
        self.max_dir_size = max_dir_size_bytes

        self.scan()

    def scan(self):
        self._files = {}
        # Get a Path object for each file in the directory
        files = [file for file in self.folder_path.iterdir() if file.is_file()]
        for file in files:
            try:
                pair = VideoRecord(file)
                self._files[pair.base_filename] = pair
            except (ValueError, FileNotFoundError) as e:
                logger.debug(f'when loading file pairs for scan got exception: {e=} on {file}')
                continue

    def add_file(self, video_filename:Path):
        # base_filename could be a Path to the text file or the video file
        # it could also be a string with or without an extension...

        # verify that the file is a video file
        if video_filename.suffix not in ['.mp4', '.avi', '.mp4v']:
            raise ValueError(f'{video_filename} should be mp4, avi, or txt')

        try:
            pair = VideoRecord(video_filename)
            self._files[pair.base_filename] = pair
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f'attempted to add filepair {video_filename=}, got exception {e}')

        # log current size of directory
        logger.debug(f'current size of directory is {self.get_total_dir_size()}')

        # check if the directory is too big
        if self.get_total_dir_size() > self.max_dir_size:
            # delete the oldest file
            oldest_file = self._get_oldest_pair()
            logger.debug(f'deleting oldest file {oldest_file.video_file_path}')
            self.delete_pair(oldest_file)

    def list_pairs(self):
        if hasattr(self, '_files'):
            return self._files
        else:
            return {}
    def get_pair_file_size(self, base_filename:Union[str, VideoRecord]):
        if isinstance(base_filename, VideoRecord):
            base_filename = base_filename.base_filename
        pair:VideoRecord = self._files.get(base_filename, None)
        if pair is None:
            return None
        txt_size = os.path.getsize(pair.text_file_path)
        vid_size = os.path.getsize(pair.video_file_path)
        return txt_size + vid_size
    def get_total_dir_size(self):
        size = 0
        for pair in self._files.values():
            _s = self.get_pair_file_size(pair)
            size += _s if _s is not None else 0
        return size
    def delete_pair(self, base_filename:Union[str, Path, VideoRecord]):
        if isinstance(base_filename, VideoRecord):
            base_filename = base_filename.base_filename
        if isinstance(base_filename, Path):
            base_filename = base_filename.stem
        pair:Optional[VideoRecord] = self._files.get(base_filename, None)
        if pair == None:
            raise KeyError('pair not in FileManager')
        
        try:
            pair.delete()
            self._files.pop(pair.base_filename)
        except Exception as e:
            logger.exception(f'an unhandled exception occured while attempting to pop {base_filename=}. {self._files=}, {e}')
            raise e

    def _get_oldest_pair(self) -> VideoRecord:
        oldest_pair:VideoRecord = min(self.list_pairs().values(), key=lambda pair: pair.creation_date())
        return oldest_pair
    
    def _get_newest_pair(self):
        newest_pair:VideoRecord = max(self.list_pairs().values(), key=lambda pair: pair.creation_date())
        return newest_pair





class VideoFileManager:
    _videowriter:Optional[cv2.VideoWriter]
    _frame_counter:int = 0
    _root_video_file_location: Path
    _resolution:Resolution
    _fps: int
    _queue:Queue = Queue()
    _video_file_manager_thread: threading.Thread
    _kill_video_file_manager_thread: threading.Event
    _fileManager:FileManager
    _file_pair:VideoRecord

    def __init__(self, root_video_file_location: Path, resolution:Resolution, fps:int, file_manager:Optional[FileManager] = None, max_video_length_frame_seconds:int = 60*5):
        self._root_video_file_location = root_video_file_location
        self._resolution = resolution
        self._fps = fps
        self.max_video_length_frame_frames = int(max_video_length_frame_seconds * fps)
        self._videowriter, self._file_pair = self._open_video_writer(self.create_video_file_name(root_video_file_location, time.time()))
        self._fileManager = file_manager if file_manager else FileManager(root_video_file_location, 10**6)
        self.Start()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.Stop(blocking=True)
    def Start(self) -> bool:
        if hasattr(self, '_video_file_manager_thread'): return False
        self._video_file_manager_thread = threading.Thread(target=self._video_file_manager_thread_function, name="Filemanager_Thread", daemon=True)
        self._kill_video_file_manager_thread = threading.Event()
        self._video_file_manager_thread.start()
        return True
    def Stop(self, blocking:bool = False, timeout:float = -1):
        self._kill_video_file_manager_thread.set()
        if blocking or timeout > 0:
            self._video_file_manager_thread.join(timeout=3 if timeout <= 0 else timeout)
    def GetQueue(self) -> Queue:
        return self._queue
    def _open_video_writer(self, video_filename_path:Path) -> tuple[cv2.VideoWriter, VideoRecord]:
        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        logger.info(f'creating cv2.videowriter at {video_filename_path} with fourcc {fourcc} at fps {self._fps} at resolution {self._resolution} with max frame length of {self.max_video_length_frame_frames} or {self._resolution} with max frame length of {self.max_video_length_frame_frames / self._fps=} seconds')
        videowriter = cv2.VideoWriter(str(video_filename_path.absolute()), fourcc, self._fps, self._resolution)
        
        # reset frame counter
        self._frame_counter = 0

        file_pair = VideoRecord(video_filename_path)
        return videowriter, file_pair
    def _close_video_writer(self, videowriter:cv2.VideoWriter):
        videowriter.release()
    def create_video_file_name(self, root_video_file_location: Path, unix_timestamp:float) -> Path:
        # filename will be humanReadableTime_unixTimestamp.extension
            if 'file_extension' in os.environ:
                _file_extension = os.environ['file_extension']
            else:
                _file_extension = '.mp4v' if sys.platform == 'win32' else '.avi'
            
            _file_name = f"{time.strftime('%Y%m%d_%H%M%S_%Z', time.localtime())}.{_file_extension}"

            file_name_path = root_video_file_location / _file_name

            return file_name_path
    def _video_file_manager_thread_function(self):
        logger.debug('started _video_file_manager_thread_function')
        while not self._kill_video_file_manager_thread.is_set():
            # get the frame
            # logger.debug('fetching file from queue')
            frame = self._queue.get()
            self._write(frame)
            self._frame_counter += 1
            # if self.max_video_length_frame_frames % self._fps*3 ==  self._frame_counter:
            # logger.debug(f'{self._frame_counter=}, {self.max_video_length_frame_frames=}')
            if self._frame_counter > self.max_video_length_frame_frames:
                logger.debug(f'creating new video file as frame counter is at {self._frame_counter}')
                self._close_video_writer(self._videowriter)
                self._videowriter, self._file_pair = self._open_video_writer(self.create_video_file_name(self._root_video_file_location, time.time()))

           
    def _write(self, frame:Frame):
        if self._videowriter is None:
            raise RuntimeError('attempted to write to a cv2.videowriter that does not exist')
        # logger.debug('writing frame')
        
        # write the frame to file
        self._videowriter.write(frame)
        # write the frame to log
        self._file_pair.write_line(frame=frame, frame_counter_int=self._frame_counter)

        
        