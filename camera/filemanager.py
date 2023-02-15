import logging
logger = logging.getLogger(__name__)
import cv2
import numpy as np
import os
import datetime
from queue import Queue
import typing
import threading

class FileManager:
    _videowriter: cv2.VideoWriter
    _videowriter_function: typing.Callable
    _frame_count: int
    base_file_location:str
    frame_width:int
    frame_height:int
    fps:int
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    queue:Queue

    import os

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
    
    def time_to_file_name(timestamp:datetime.datetime):
        # note: use utcnow() NOT now()
        return timestamp.strftime(r'%Y%m%d_%H%M%S_%f')

    def directory_to_file(foldername:str):
        pass

    def file_to_time(filename:str):
        pass

    def __init__(self, queue:Queue, frame_width:int, frame_height:int, fps:int, root_file_location:str, videowriter_function:typing.Callable) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self._frame_count = 0
        if not os.path.exists(root_file_location):
            logger.info(f'creating file location at: {root_file_location}')
            os.makedirs(root_file_location)
        self.base_file_location = root_file_location
        self.queue = queue
        self._videowriter_function = videowriter_function
    
    def _createVideoWriter(self, filename):
        self._videowriter = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.frame_width, self.frame_height))

    def __enter__(self):
        _filename = 'test.avi'
        self._createVideoWriter(os.path.join(self.base_file_location, _filename))
        logger.debug('running filemanager class enter')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._videowriter.release()
        logger.debug('running filemanager class exit')

    def write(self, frame:np.ndarray):
        if not isinstance(frame, np.ndarray):
            raise TypeError('frame must be an np.ndarray')
        if not (frame.shape[0] == self.frame_width and 
            frame.shape[1] == self.frame_height and
            frame.shape[2] == 3):
            raise ValueError(f'frame must be in dimentions {self.frame_width}, {self.frame_height}, 3 instead got size {frame.shape}')
        self._videowriter.write(frame)  

    def _start_filemanager_thread(queue:Queue, frame_width:int, frame_height:int, fps:int, root_file_location:str, videowriter_function:typing.Callable):
        while True:
            with FileManager(queue, frame_width, frame_height, fps, root_file_location, videowriter_function) as filemanager:
                while True:
                    frame = filemanager.queue.get()
                    filemanager.write(frame)
    
    def Start_Filemanager_thread(queue:Queue, frame_width:int, frame_height:int, fps:int, root_file_location:str, videowriter_function:typing.Callable):
        thread = threading.Thread(target=FileManager._start_filemanager_thread, name="Filemanager_Thread", daemon=True
            , args=(queue, frame_width, frame_height, fps, root_file_location, videowriter_function))
        thread.start()
        return thread

