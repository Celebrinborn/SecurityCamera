import logging
logger = logging.getLogger(__name__)
import cv2
import numpy as np
import os
import datetime
from queue import Queue
import typing
from typing import List, Optional
import threading
import sys
import inspect

class FileManager:
    _videowriter: cv2.VideoWriter
    _frame_count: int
    base_file_location:str
    frame_width:int
    frame_height:int
    fps:int
    queue:Queue
    _filemanager_thread:threading.Thread

    def __init__(self, frame_width: int, frame_height: int, fps: int, root_file_location: str) -> None:
        if not isinstance(frame_width, int):
            raise TypeError(f"frame_width should be an integer, instead got {type(frame_width)}")
        if not isinstance(frame_height, int):
            raise TypeError(f"frame_height should be an integer, instead got {type(frame_height)}")
        if not isinstance(fps, int):
            raise TypeError(f"fps should be an integer, instead got {type(fps)}")
        if not isinstance(root_file_location, str):
            raise TypeError(f"root_file_location should be a string, instead got {type(root_file_location)}")
        if not os.path.exists(root_file_location):
            drive, path = os.path.splitdrive(root_file_location)
            if not path.startswith(os.path.sep):
                raise ValueError(f"root_file_location '{root_file_location}' is invalid - missing path separator after drive letter")
            else: 
                raise ValueError(f"root_file_location '{root_file_location}' does not exist")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self._frame_count = 0
        self._filemanager_thread = None
        if not os.path.exists(root_file_location):
            logger.info(f'creating file location at: {root_file_location}')
            os.makedirs(root_file_location)
        self.base_file_location = root_file_location
        self.queue = Queue()
        self._kill_the_daemon_event = threading.Event()
        self._videowriter = None

    def __enter__(self):
        logger.debug('running filemanager class enter')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.Stop()
        logger.debug('running filemanager class exit')

    def _get_file_sizes(directory):
        file_sizes = {}
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                file_sizes[filename] = os.path.getsize(path)
        return file_sizes
    def time_to_folder_name(timestamp:datetime.datetime):
        # note: use utcnow() NOT now()
        return timestamp.strftime(r'%Y%m%d_%H')

    def _start_filemanager_thread(kill_the_daemon_event:threading.Event, queue:Queue, fps:int, root_file_location:str, video_length_seconds:int, frame_width:int, frame_height:int, _filemanager):
        assert isinstance(frame_width, int), 'frame_width is not int'
        assert isinstance(frame_height, int), 'frame_height is not int'

        def write(videoWriter:cv2.VideoWriter, frame:np.ndarray):
            # logger.debug('writing frame')
            if not isinstance(frame, np.ndarray):
                raise TypeError('frame must be an np.ndarray')
            videoWriter.write(frame)

        
        def time_to_file_name(timestamp: Optional[datetime.datetime] = None):
            """
            Returns a timestamp string in the format of '%Y%m%d_%H%M%S'.

            Parameters:
                timestamp (datetime.datetime, optional): The timestamp to convert to a filename string. Defaults to None.

            Returns:
                str: The timestamp string in the format of '%Y%m%d_%H%M%S'.
            """
            if timestamp is None:
                timestamp = datetime.datetime.utcnow()
            if 'file_extension' in os.environ:
                _file_extension = os.environ['file_extension']
            else:
                if sys.platform == 'win32':
                    _file_extension = 'mp4v'
                else: # sys.platform == 'linux':
                    _file_extension = 'avi'
                    # Linux-specific code here
            _file_name = f"{timestamp.strftime(r'%Y%m%d_%H%M%S')}.{_file_extension}"
            logger.debug(f'generated filename {_file_name}')
            return _file_name

        def start_video(filepath:str, filename:str, fps:int, frame_width:int, frame_height:int, filemanager:FileManager) -> cv2.VideoWriter:
            if not isinstance(filepath, str):
                raise TypeError(f"Expected 'filepath' argument to be of type 'str', but got {type(filepath)} instead.")
            if not isinstance(filename, str):
                raise TypeError(f"Expected 'filename' argument to be of type 'str', but got {type(filename)} instead.")
            if not isinstance(fps, int):
                raise TypeError(f"Expected 'fps' argument to be of type 'int', but got {type(fps)} instead.")
            if not isinstance(frame_width, int):
                raise TypeError(f"Expected 'frame_width' argument to be of type 'int', but got {type(frame_width)} instead.")
            if not isinstance(frame_height, int):
                raise TypeError(f"Expected 'frame_height' argument to be of type 'int', but got {type(frame_height)} instead.")
            if not isinstance(filemanager, FileManager):
                raise TypeError(f"Expected 'filemanager' argument to be of type 'FileManager', but got {type(filemanager)} instead.")
            
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            _filename_path = os.path.join(filepath, filename)
            _resolution = (frame_width, frame_height)
            logger.debug(f'creating videowriter with filepath {_filename_path}; fourcc{fourcc}; fps: {fps}; resolution {_resolution}')
            _videowriter = cv2.VideoWriter(_filename_path, fourcc, fps, _resolution)
            filemanager._videowriter = _videowriter
            logger.debug(f'successfully created videowriter with filepath {_filename_path}; fourcc{fourcc}; fps: {fps}; resolution {_resolution}')
            return _videowriter


        def end_video(videowriter:cv2.VideoWriter):
            # Assert that the object is an instance of cv2.VideoWriter
            assert isinstance(videowriter, cv2.VideoWriter), "videowriter is not a cv2.VideoWriter object"
            logger.debug('releasing video writer')
            videowriter.release()
            logger.debug('successfully released video writer')

        logger.debug('filemanager daemon has started')

        _frame_counter = 0
        videowriter = start_video(filepath = root_file_location,
                                  filename=time_to_file_name(),
                                  fps=fps,
                                  frame_width=frame_width,
                                  frame_height=frame_height,
                                  filemanager=_filemanager)
        
        while not kill_the_daemon_event.is_set():
            if _frame_counter > video_length_seconds * fps:
                _frame_counter = 0
                end_video(videowriter)
                videowriter = start_video(filepath = root_file_location,
                                          filename=time_to_file_name(),
                                          fps=fps,
                                          frame_width=frame_width,
                                          frame_height=frame_height,
                                          filemanager=_filemanager)
            frame = queue.get()
            _frame_counter += 1
            if _frame_counter % 100 == 0:
                logger.debug(f'writing frame {_frame_counter} of {video_length_seconds * fps} ({video_length_seconds} seconds at {fps})')
            write(videowriter, frame)
        logger.debug('kill the demon event is True', stack_info=True)
        end_video(videowriter)

    
    def Start(self):
        _caller = inspect.stack()[1]
        logger.info(f'Starting Camera.Start() from {_caller.filename}:{_caller.lineno}')

        logging.debug(f"type of frame_width: {type(self.frame_width)}")
        logging.debug(f"type of frame_height: {type(self.frame_height)}")


        logger.info(f'starting filemanager thread with fps {self.fps}')
        self._filemanager_thread = threading.Thread(target=FileManager._start_filemanager_thread, name="Filemanager_Thread", daemon=True,
            kwargs={
                'kill_the_daemon_event': self._kill_the_daemon_event,
                'queue': self.GetQueue(),
                'fps': self.fps,
                'root_file_location': self.base_file_location,
                'video_length_seconds': 30,
                'frame_width': self.frame_width,
                'frame_height': self.frame_height,
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
    

if __name__ == '__main__':
    import sys
    import time
    from log_config import configure_logging
    configure_logging()
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.getLogger().setLevel(logging.DEBUG)
    logger.critical('starting filemanager.py module AS MAIN')

    

    with FileManager(480, 360, 15, os.path.join('data', 'temp_filemanager_output')) as filemanager:
        def create_test_data():
            data = np.random.randint(0, 256, size=(380, 480, 3))
            return data.astype(np.uint8)
        
        queue = filemanager.GetQueue()
        filemanager.Start()

        for i in range(15*5):
            logger.debug(f'adding test frame {i}')
            queue.put(create_test_data())
            time.sleep(1/15)
        filemanager.Stop()

        time.sleep(15)
        logger.debug('closing filemanger')
    logger.debug('closing app')
    

