import logging
import uuid
logger = logging.getLogger(__name__)
import cv2
import numpy as np
import pandas as pd
import os
import pathlib
from os import fspath
import datetime
from queue import Queue, LifoQueue
import typing
from typing import List, Optional, Union, Tuple, Dict, Any
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

import atexit

logger = logging.getLogger(__name__)

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Iterable, cast, TextIO, Optional
import json
from camera.resolution import Resolution

from camera.sqlmanager import SQLManager

@dataclass
class VideoRecord:
    _cache:pd.DataFrame
    batch_size:int
    sql_manager:SQLManager
    mssql_table:str = '[cameras].[frames]'
    row_curser:int = 0
    

    def __init__(self, video_file_path: Path, batch_size:int = 10*60*5):
        # set parameters

        # verify the file exists and is a video file
        if not video_file_path.is_file():
            raise ValueError(f'{video_file_path.resolve()} is not a file')
        if video_file_path.suffix not in ['.mp4', '.avi', '.mp4v']:
            raise ValueError(f'{video_file_path} should be mp4, avi, or mp4v')
        self.video_file_path = video_file_path  # assuming this is a Path to the video file

        self.batch_size = batch_size

        # initialize the sql manager
        self.sql_manager = SQLManager()
        
        # initialize the cache
        self._cache = self.create_cache()
        
        # add the video to the database if it does not exist
        if not self.sql_manager.DoesVideoExists(self.video_file_path):
            self.sql_manager.AddVideo(self.video_file_path)

    def write_line(self, frame: Frame, frame_counter_int: int) -> None:
        # send the data to the database
        # self.sql_manager.AddFrameDetails(frame_counter_int, frame.guid, self.video_file_path)

        # if batch is full, write to database
        if self.row_curser == self.batch_size:
            self.flush_cache()

        # write the data to the cache
        self._cache.iloc[self.row_curser] = pd.Series([str(frame.guid), str(self.video_file_path.absolute())])
        self.row_curser += 1

    def delete(self) -> bool:
        # Delete the file from the filesystem
        try:
            self.sql_manager.DeleteVideo(self.video_file_path)
            self.video_file_path.unlink(missing_ok=True)
            return True
        except FileNotFoundError as e:
            logger.exception(f'Unable to delete {self.video_file_path}. {e}')
        except Exception as e:
            logger.exception(f'An unhandled exception occurred while attempting to delete {self.video_file_path}. {e}')
            raise e
        return False
    
    def __eq__(self, other):
        if not isinstance(other, VideoRecord):
            return False
        return self.video_file_path == other.video_file_path

    def __str__(self):
        return f'{self.video_file_path.name}'
    def file_size(self):
        return self.video_file_path.stat().st_size if self.video_file_path.is_file() else 0

    def create_cache(self):
        # Define the DataFrame with the correct data types and preallocate space
        index = pd.RangeIndex(start=0, stop=self.batch_size, step=1)
        data = {
            'frame_guid': pd.Series([None] * self.batch_size, dtype='object'),
            'video_file_name': pd.Series([str(self.video_file_path.absolute())] * self.batch_size, dtype='str')
        }
        df = pd.DataFrame(data, index=index)
        # logger.debug(f'created cache with shape {df.shape} and size {df.memory_usage().sum()} and types {df.dtypes}')

        return df
    def flush_cache(self):
        # Write the cached data to the database
        batch_data = self._cache.iloc[:self.row_curser]

        logger.debug(f'writing {batch_data.shape} to database')

        self.sql_manager.add_frame_batch(batch_data, self.video_file_path)
        self.row_curser = 0

    
class FileManager:
    _video_records: List[VideoRecord] = []
    def __init__(self, root_folder: Path, max_dir_size_bytes: int):
        self.sql_manager = SQLManager()
        self.folder_path = root_folder
        self.max_dir_size = max_dir_size_bytes
        self.scan()
    
    def scan(self):
        # Update the _files dictionary with VideoRecord objects
        logger.debug(f'scanning {self.folder_path.absolute()} for video files')
        for file in [f for f in self.folder_path.iterdir() if f.suffix in ['.mp4', '.avi', '.mp4v'] and f.is_file()]:
            video_record = VideoRecord(file)
            self._video_records.append(video_record)

    def add_file(self, video_record: VideoRecord):
        self._video_records.append(video_record)

        # Log current size of directory as percentage
        logger.debug(f'Current size of directory is {(self.get_total_dir_size() / self.max_dir_size * 100):.2f}%. {self.get_total_dir_size():,} of {self.max_dir_size:,}')

        # Check if the directory size exceeds the maximum allowed
        while self.get_total_dir_size() > self.max_dir_size:
            oldest_record = self._get_oldest_record()
            if oldest_record:
                logger.debug(f'Deleting file due to exceeding dir max size {oldest_record.video_file_path=}')
                self.delete_record(oldest_record)

    def get_record_file_size(self, video_record: VideoRecord) -> int:
        
        if len(self._video_records) == 0:
            raise ValueError('No files in FileManager')

        # Check if the file exists
        if video_record.video_file_path.is_file():
            try:
                return video_record.video_file_path.stat().st_size
            except Exception as e:
                logger.exception(f'Unable to get size of {video_record.video_file_path}. exception occurred {e}')
        logger.error(f'Unable to get size of {video_record.video_file_path}, file does not exist')
        return 0

    def get_total_dir_size(self) -> int:
        total_size = 0
        for video_record in self._video_records:
            assert isinstance(video_record, VideoRecord), f'{video_record=} is not a VideoRecord object'
            total_size += video_record.file_size()
        return total_size

    def delete_record(self, video_record:VideoRecord) -> bool:
        assert isinstance(video_record, VideoRecord), f'{video_record=} is not a VideoRecord object'
        if video_record in self._video_records:
            self._video_records.remove(video_record)
        else:
            logger.warning(f'{video_record=} not in {self._video_records=}')


        isDeleteSucess = video_record.delete()

        return isDeleteSucess
        
        

    def _get_oldest_record(self) -> Optional[VideoRecord]:
        if self._video_records:
            return min([vr for vr in self._video_records if vr.video_file_path.is_file()], \
                       key=lambda vr: vr.video_file_path.stat().st_ctime \
                       if vr.video_file_path.is_file() else 0)
        return None

    def _get_newest_record(self) -> Optional[VideoRecord]:
        if self._video_records:
            return max([vr for vr in self._video_records if vr.video_file_path.is_file()], \
                       key=lambda vr: vr.video_file_path.stat().st_ctime \
                       if vr.video_file_path.is_file() else 0)
        return None





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
    _video_record:VideoRecord

    def __init__(self, root_video_file_location: Path, resolution:Resolution, \
                 fps:int, file_manager:Optional[FileManager] = None, max_video_length_frame_seconds:int = 60*5, \
                    *, init_without_start = False):
        self._root_video_file_location = root_video_file_location
        self._resolution = resolution
        self._fps = fps
        self.max_video_length_frame_frames = int(max_video_length_frame_seconds * fps)
        self._fileManager = file_manager if file_manager is not None else FileManager(root_video_file_location, 10**6)
        # assert isinstance(self._fileManager, FileManager), f'{self._fileManager=} is not a FileManager object'
        self._videowriter = self._open_video_writer(self.create_video_file_name(root_video_file_location, time.time()))
        
        # register the close function to be called when the program exits to avoid corrupt video files
        atexit.register(self._close_video_writer, self._videowriter)
        
        if not init_without_start:
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
    def _open_video_writer(self, video_filename_path:Path) -> cv2.VideoWriter:
        video_filename_str = str(video_filename_path)
        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if sys.platform == 'win32' else cv2.VideoWriter_fourcc(*'FMP4')
        logger.info(f'creating cv2.videowriter at {video_filename_str} with fourcc {fourcc} at fps {self._fps} at resolution {self._resolution} with max frame length of {self.max_video_length_frame_frames} or {self._resolution} with max frame length of {self.max_video_length_frame_frames / self._fps=} seconds or {self.max_video_length_frame_frames / self._fps / 60=} minutes')

        videowriter = cv2.VideoWriter(video_filename_str, fourcc, self._fps, self._resolution) # type: ignore

        # using an assert verify the file exists
        # assert video_filename_path.is_file(), f'{video_filename_path} is not a file'

        # reset frame counter
        self._frame_counter = 0

        self._video_record = VideoRecord(video_filename_path)
        self._fileManager.add_file(self._video_record)
        return videowriter
    def _close_video_writer(self, videowriter:cv2.VideoWriter):
        self._video_record.flush_cache()
        videowriter.release()
        logger.debug(f'closing cv2.videowriter')
    def create_video_file_name(self, root_video_file_location: Path, unix_timestamp:float) -> Path:
        # filename will be humanReadableTime_unixTimestamp.extension
            if 'file_extension' in os.environ:
                _file_extension = os.environ['file_extension']
            else:
                _file_extension = '.mp4' if sys.platform == 'win32' else '.avi'
            
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
                self._close_video_writer(self._videowriter) # type: ignore
                self._videowriter = self._open_video_writer(self.create_video_file_name(self._root_video_file_location, time.time()))

           
    def _write(self, frame:Frame):
        if self._videowriter is None:
            raise RuntimeError('attempted to write to a cv2.videowriter that does not exist')
        # logger.debug('writing frame')
        
        # write the frame to file
        self._videowriter.write(self.scale(frame, self._resolution))
        # self._videowriter.write(frame)

        # write the frame to log
        self._video_record.write_line(frame=frame, frame_counter_int=self._frame_counter)
    
    @staticmethod
    def scale(frame: Frame, target_resolution: Resolution) -> Frame:
        scaled_frame = frame.preserve_identity_with(cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA))
        # logger.debug(f'scaling frame {frame.shape} to {target_resolution}, identity preserved: {scaled_frame == frame}')
        return scaled_frame
    

        
        