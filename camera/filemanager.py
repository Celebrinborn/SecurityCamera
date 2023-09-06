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
class File_Pair:
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
    def __init__(self, root_folder:Path, max_dir_size:int):
        
        self.folder_path = root_folder
        self.max_dir_size = max_dir_size

        self.scan()

    def scan(self):
        self._files = {}
        # Get a Path object for each file in the directory
        files = [file for file in self.folder_path.iterdir() if file.is_file()]
        for file in files:
            try:
                pair = File_Pair(file)
                self._files[pair.base_filename] = pair
            except (ValueError, FileNotFoundError) as e:
                logger.debug(f'when loading file pairs for scan got exception: {e=} on {file}')
                continue


    

    def add_file_pair(self, video_filename:Path):
        # base_filename could be a Path to the text file or the video file
        # it could also be a string with or without an extension...

        try:
            pair = File_Pair(video_filename)
            self._files[pair.base_filename] = pair
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f'attempted to add filepair {video_filename=}, got exception {e}')
    def list_pairs(self):
        if hasattr(self, '_files'):
            return self._files
        else:
            return {}
    def get_pair_file_size(self, base_filename:Union[str, File_Pair]):
        if isinstance(base_filename, File_Pair):
            base_filename = base_filename.base_filename
        pair:File_Pair = self._files.get(base_filename, None)
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
    def delete_pair(self, base_filename:Union[str, Path, File_Pair]):
        if isinstance(base_filename, File_Pair):
            base_filename = base_filename.base_filename
        if isinstance(base_filename, Path):
            base_filename = base_filename.stem
        pair:File_Pair = self._files.get(base_filename, None)
        if pair == None:
            raise KeyError('pair not in FileManager')
        
        try:
            pair.delete()
            self._files.pop(pair.base_filename)
        except Exception as e:
            logger.exception(f'an unhandled exception occured while attempting to pop {base_filename=}. {self._files=}, {e}')
            raise e

    def _get_oldest_file(self):
        return min([p.creation_date() for p in cast(Iterable[File_Pair], self._files)])
    def _get_newest_file(self):
        return max([p.creation_date() for p in cast(Iterable[File_Pair], self._files)])

class VideoFileManagerOld:
    _videowriter: cv2.VideoWriter
    _frame_count: int
    root_file_location:str
    frame_width:int
    frame_height:int
    fps:int
    queue:Queue
    _filemanager_thread:threading.Thread

    def __init__(self, frame_width: int, frame_height: int, fps: int, camera_name:str, root_file_location:Union[str, None] = None) -> None:
        if not isinstance(frame_width, int):
            raise TypeError(f"frame_width should be an integer, instead got {type(frame_width)}")
        if not isinstance(frame_height, int):
            raise TypeError(f"frame_height should be an integer, instead got {type(frame_height)}")
        if not isinstance(fps, int):
            raise TypeError(f"fps should be an integer, instead got {type(fps)}")
        if not isinstance(root_file_location, str) and root_file_location is not None:
            raise TypeError(f"root_file_location should be a string, instead got {type(root_file_location)}")
        if root_file_location is not None and not os.path.exists(root_file_location):
            drive, path = os.path.splitdrive(root_file_location)
            if not path.startswith(os.path.sep):
                raise ValueError(f"root_file_location '{root_file_location}' is invalid - missing path separator after drive letter")
            else: 
                raise ValueError(f"root_file_location '{root_file_location}' does not exist")

        if root_file_location == None:
            root_file_location = os.path.abspath(os.path.join('data', camera_name, 'video_cache'))
            logger.info(f'no root_file_location path provided, defaulting to {root_file_location}')
            if not os.path.exists(root_file_location):
                try:
                    os.makedirs(root_file_location)
                except OSError as e:
                    logger.exception(f'UNABLE TO CREATE root_file_location {root_file_location}, e')


        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self._frame_count = 0
        self.camera_name = camera_name
        if not os.path.exists(root_file_location):
            logger.info(f'creating file location at: {root_file_location}')
            os.makedirs(root_file_location)
        self.root_file_location = root_file_location
        self.queue = Queue()
        self._kill_the_daemon_event = threading.Event()
        self._videowriter = None


    def __enter__(self):
        logger.debug('running filemanager class enter')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.Stop()
        logger.debug('running filemanager class exit')

    def _get_file_sizes(self, directory):
        file_sizes = {}
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                file_sizes[filename] = os.path.getsize(path)
        return file_sizes
    def time_to_folder_name(self, timestamp:datetime.datetime):
        # note: use utcnow() NOT now()
        return timestamp.strftime(r'%Y%m%d_%H')

    def _start_filemanager_thread(self, kill_the_daemon_event:threading.Event, queue:Queue, fps:int, root_file_location:str, video_length_seconds:int, frame_width:int, frame_height:int, camera_name:str, _filemanager):
        assert isinstance(frame_width, int), 'frame_width is not int'
        assert isinstance(frame_height, int), 'frame_height is not int'

        def write(videoWriter:cv2.VideoWriter, frame:Frame, file_pair:File_Pair, frame_counter:int):
            # logger.debug('writing frame')
            if not isinstance(frame, Frame):
                raise TypeError('frame must be a Frame type')
            videoWriter.write(frame)
            file_pair.write_line(frame, frame_counter)
        
        def time_to_file_name(timestamp: Optional[datetime.datetime] = None):
            """
            Returns a timestamp string in the format of '%Y%m%d_%H%M%Z'.

            Parameters:
                timestamp (datetime.datetime, optional): The timestamp to convert to a filename string. Defaults to None.

            Returns:
                str: The timestamp string in the format of '%Y%m%d_%H%M%Z'.
            """
            if timestamp is None:
                timestamp = datetime.datetime.utcnow()

            # Convert UTC to local time
            timestamp = timestamp.astimezone()

            if 'file_extension' in os.environ:
                _file_extension = os.environ['file_extension']
            else:
                if sys.platform == 'win32':
                    _file_extension = 'mp4v'
                else: # sys.platform == 'linux':
                    _file_extension = 'avi'
            _file_name = f"{timestamp.strftime('%Y%m%d_%H%M%S_%Z').lower()}.{_file_extension}"
            logger.debug(f'generated filename {_file_name}')
            return _file_name


        def start_video(base_filepath:str, filename:str, fps:int, frame_width:int, frame_height:int, filemanager:VideoFileManagerOld, camera_name:str) -> cv2.VideoWriter:
            if not isinstance(base_filepath, str):
                raise TypeError(f"Expected 'filepath' argument to be of type 'str', but got {type(base_filepath)} instead.")
            if not isinstance(filename, str):
                raise TypeError(f"Expected 'filename' argument to be of type 'str', but got {type(filename)} instead.")
            if not isinstance(fps, int):
                raise TypeError(f"Expected 'fps' argument to be of type 'int', but got {type(fps)} instead.")
            if not isinstance(frame_width, int):
                raise TypeError(f"Expected 'frame_width' argument to be of type 'int', but got {type(frame_width)} instead.")
            if not isinstance(frame_height, int):
                raise TypeError(f"Expected 'frame_height' argument to be of type 'int', but got {type(frame_height)} instead.")
            if not isinstance(filemanager, VideoFileManagerOld):
                raise TypeError(f"Expected 'filemanager' argument to be of type 'FileManager', but got {type(filemanager)} instead.")
            
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            if not os.path.exists(base_filepath):
                raise Exception("Base filepath does not exist: {}".format(base_filepath))

            # Check if the folder path exists, if not create it
            if not os.path.exists(base_filepath):
                logger.warning(f'base_filepath {base_filepath} does not exist. creating it. this should not happen', exc_info=True, stack_info=True)
                os.makedirs(base_filepath)

            # Join the filepath
            _filename_path = os.path.join(base_filepath, filename)

            _resolution = (frame_width, frame_height)
            logger.debug(f'creating videowriter with filepath {_filename_path}; fourcc{fourcc}; fps: {fps}; resolution {_resolution}')
            _videowriter = cv2.VideoWriter(_filename_path, fourcc, fps, _resolution)
            filemanager._videowriter = _videowriter
            logger.debug(f'successfully created videowriter with filepath {_filename_path}; fourcc{fourcc}; fps: {fps}; resolution {_resolution}')

            # create file pair
            file_pair = File_Pair(Path(_filename_path))
            return _videowriter, _filename_path, file_pair


        def end_video(videowriter:cv2.VideoWriter):
            # Assert that the object is an instance of cv2.VideoWriter
            assert isinstance(videowriter, cv2.VideoWriter), "videowriter is not a cv2.VideoWriter object"
            logger.debug('releasing video writer')
            videowriter.release()
            logger.debug('successfully released video writer')

        logger.debug('filemanager daemon has started')

        _frame_counter = 0
        videowriter, filename, file_pair = start_video(base_filepath = root_file_location,
                                  filename=time_to_file_name(),
                                  fps=fps,
                                  frame_width=frame_width,
                                  frame_height=frame_height,
                                  filemanager=_filemanager,
                                  camera_name=camera_name)
        
        _max_dir_size = int(5e+9) # 5GB
        file_manager = FileManager(root_folder=root_file_location, max_dir_size=_max_dir_size)
        
        logger.debug('starting main filemanager loop')
        while not kill_the_daemon_event.is_set():
            if _frame_counter > video_length_seconds * fps:
                _frame_counter = 0
                file_manager.add_file_pair(file_pair)
                end_video(videowriter)
                videowriter, filename, file_pair = start_video(base_filepath = root_file_location,
                                          filename=time_to_file_name(),
                                          fps=fps,
                                          frame_width=frame_width,
                                          frame_height=frame_height,
                                          filemanager=_filemanager,
                                          camera_name=camera_name)
            frame = queue.get()
            _frame_counter += 1
            # if _frame_counter % 100 == 0:
            #     logger.debug(f'writing frame {_frame_counter} of {video_length_seconds * fps} ({video_length_seconds} seconds at {fps})')
            write(videowriter, frame, file_pair, _frame_counter)
        logger.info('kill the demon event is True', stack_info=True)
        end_video(videowriter)

    
    def Start(self):
        _caller = inspect.stack()[1]
        logger.info(f'Starting Camera.Start() from {_caller.filename}:{_caller.lineno}')

        logging.debug(f"type of frame_width: {type(self.frame_width)}")
        logging.debug(f"type of frame_height: {type(self.frame_height)}")


        logger.info(f'starting filemanager thread with fps {self.fps}')
        self._filemanager_thread = threading.Thread(target=VideoFileManagerOld._start_filemanager_thread, name="Filemanager_Thread", daemon=True,
            kwargs={
                'kill_the_daemon_event': self._kill_the_daemon_event,
                'queue': self.GetQueue(),
                'fps': self.fps,
                'root_file_location': self.root_file_location,
                'video_length_seconds': 15 * 60,
                'frame_width': self.frame_width,
                'frame_height': self.frame_height,
                'camera_name': self.camera_name,
                '_filemanager': self
            })
        logger.debug('starting filemanager daemon')
        self._filemanager_thread.start()
    def Stop(self):
        logger.debug('killing daemon')
        # set the switch make the daemon thread self abort
        self._kill_the_daemon_event.set()

        # block until the thread dies
        if self._filemanager_thread is not None:
            self._filemanager_thread.join()
        else:
            logger.warning('filemanager Stop was called however thread is already dead', stack_info=True)

    def GetQueue(self) -> Queue:
        """
        Returns the queue associated with this instance of the file manager.

        Returns:
            Queue: The associated queue.
        """
        return self.queue
    


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
    _file_pair:File_Pair

    def __init__(self, root_video_file_location: Path, resolution:Resolution, fps:int, file_manager:FileManager):
        self._root_video_file_location = root_video_file_location
        self._resolution = resolution
        self._fps = fps
        self._videowriter, self._file_pair = self._open_video_writer(self.create_video_file_name(root_video_file_location, time.time()))
        self._fileManager = file_manager
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
    def _open_video_writer(self, video_filename_path:Path) -> tuple[cv2.VideoWriter, File_Pair]:
        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        videowriter = cv2.VideoWriter(str(video_filename_path), fourcc, self._fps, self._resolution)
        
        # reset frame counter
        self._frame_counter = 0

        file_pair = File_Pair(video_filename_path)
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
        while not self._kill_video_file_manager_thread:
            # get the frame
            frame = self._queue.get()
            self._write(frame)
           
    def _write(self, frame:Frame):
         # increment the frame counter
        self._frame_counter += 1


        if self._videowriter is None:
            raise RuntimeError('attempted to write to a cv2.videowriter that does not exist')
        # write the frame to file
        self._videowriter.write(frame)

        # write the frame to log
        self._file_pair.write_line(frame=frame, frame_counter_int=self._frame_counter)

        
            

        