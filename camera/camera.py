import logging
logger = logging.getLogger(__name__)
import cv2
import numpy as np
import os
from queue import Queue
import threading
import time
import inspect

import typing

class Camera:
    """
    Class for a camera that captures frames and sends them to subscribed queues.
    """
    _camera: cv2.VideoCapture
    _camera_name:str
    _camera_url: str
    _max_fps: int
    _killDaemon: bool  # flag to abort capture daemon

    prevFrame: np.ndarray
    currentFrame: np.ndarray

    def __init__(self, camera_name:str, camera_url:str, max_fps:int, cv2_module: typing.Type[cv2.VideoCapture]=cv2.VideoCapture) -> None:
        """
        Initializes the camera instance.

        Parameters:
            camera_name (str): The name of the camera.
            camera_url (str): The URL of the camera stream.
            max_fps (int): The maximum number of frames per second that the camera should capture.
            cv2_module (cv2.VideoCapture): The module to use for capturing the frames.
        """

        # log where it was called from to catch an annoying bug
        caller_frame = inspect.currentframe().f_back
        caller_filename = caller_frame.f_code.co_filename
        caller_line_no = caller_frame.f_lineno
        caller_name = caller_frame.f_code.co_name
        logger.debug(f'{caller_filename}:{caller_line_no} {caller_name} called camera init')


        self._camera_name = camera_name
        self._camera_url = camera_url
        self._max_fps = max_fps
        self._cv2:cv2.VideoCapture = cv2_module
        logger.debug('running Camera class init')
        
        # using the dependency injection approach to assist with testing
        self._camera = self._cv2(self._camera_url)

        # check if camera opened successfully
        if not self._camera.isOpened():
            logger.error(f'unable to open camera {camera_name} at {camera_url}')
            logger.error(f'type of _camera: {type(self._camera)} type of _cv2 module {type(self._cv2)}')
        else:
            logger.info(f'opened camera {camera_name} at {camera_url}')

        # Read the current frame from the camera object and assign it to a variable
        # to ensure that prevFrame has something to populate later in the application
        logger.debug('reading first frame from camera')
        for i in range (5):
            _successful_camera_read, self.currentFrame = self._camera.read()
            if _successful_camera_read:
                logger.info('successfully read first frame from camera')
                break
            else:
                _sleep_time = 1
                logger.warning(f'unable to read from camera. attempt {i} sleeping {_sleep_time} seconds...')
                time.sleep(_sleep_time)

        if not isinstance(self.currentFrame, np.ndarray):
            logger.critical(f"currentFrame is not an ndarray, type: {type(self.currentFrame)}", stack_info=True)
            #logger.critical()
            raise TypeError("currentFrame is not an ndarray")

        self._frame_height, self._frame_width, _ = self.currentFrame.shape

        self._subscription_manager = self.SubscriptionManager()

        
    # Define the __enter__ method for the Camera class
    def __enter__(self):
        """
        Enters the camera instance.
        """
        return self

    def close(self):
        """
        Closes the camera instance.
        """
        self._camera.release()
        logger.debug('running Camera class exit')

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the camera instance.
        """
        self.Stop()
        self.close()

    def GetFrame(self) -> np.ndarray:
        """
        Captures frames from the camera.

        Returns:
            np.ndarray: The current frame from the camera.
        """
        # logger.debug('starting Camera class getframe')
        while True: #self._camera == True:
            ret, newFrame = self._camera.read()
            self.prevFrame = self.currentFrame.copy()
            self.currentFrame = newFrame
            yield self.currentFrame
        return self.currentFrame
    
    def Subscribe_queue(self, queue: Queue):
        """
        Subscribes a queue to the camera instance.

        Parameters:
            queue (Queue): The queue to subscribe to the camera instance.
        """
        caller_name = inspect.stack()[1].function
        logger.info(f'{caller_name} called Subscribe_queue')
        self._subscription_manager.subscribe_queue(queue)


    def Unsubscribe_queue(self, queue: Queue):
        """
        Unsubscribes a queue from the camera instance.

        Parameters:
            queue (Queue): The queue to unsubscribe from the camera instance.
        """
        self._subscription_manager.unsubscribe_queue(queue)
    
    class SubscriptionManager:
        """
        Manages a list of subscribed Queues and provides methods to add or remove queues.

        Attributes:
            _subscribed_queues (List[Queue]): A list of subscribed Queue objects.
        
        Methods:
            subscribe_queue(queue: Queue) -> None: Adds a new queue to the list of subscribed queues.
            unsubscribe_queue(queue: Queue) -> None: Removes a queue from the list of subscribed queues.
            _add_frame_to_queues(frame: np.ndarray) -> None: Puts a new frame into all subscribed queues.
        """
        _subscribed_queues: typing.List[Queue]

        def __init__(self):
            self._subscribed_queues = []

        def subscribe_queue(self, queue: Queue):
            if queue not in self._subscribed_queues:
                self._subscribed_queues.append(queue)

        def unsubscribe_queue(self, queue: Queue):
            if queue in self._subscribed_queues:
                self._subscribed_queues.remove(queue)

        def _add_frame_to_queues(self, frame: np.ndarray):
            for queue in self._subscribed_queues:
                queue.put(frame)
    
    def Start(self):
        """
        Starts the worker thread that reads from the camera and adds frames to any subscribed queues.
        """
        _caller = inspect.stack()[1]
        logger.debug(f'Starting Camera.Start() from {_caller.filename}:{_caller.lineno}')

        self._killDaemon = False  # initialize flag to False
        def _capture(subscriptionManager:self.SubscriptionManager, fps:int):
            """Captures frames from the camera and adds them to subscribed queues.

            Args:
                subscriptionManager: An instance of `Camera.SubscriptionManager` to manage
                    the subscribed queues.
                fps: An integer representing the desired max frames per second (FPS) rate.

            Returns:
                None
            """
            for frame in self.GetFrame():
                if self._killDaemon:  # check flag to stop thread
                    break
                start = time.perf_counter()
                subscriptionManager._add_frame_to_queues(frame)
                end = time.perf_counter()
                elapsed_time = end - start
                time_to_sleep = max(1.0 / fps - elapsed_time, 0)
                time.sleep(time_to_sleep)
        thread = threading.Thread(target=_capture, 
                                  name="Camera_Thread", 
                                  daemon=True,
                                  args=(self._subscription_manager, self._max_fps))
        thread.start()
    
    def Stop(self):
        """
        Stops the camera worker daemon
        """
        pass
        self._killDaemon = True  # set flag to stop capture thread

    def GetFrameWidth(self) -> int:
        """
        Returns the width of the frame captured by the camera.

        Returns:
            int: The width of the frame.
        """
        return self._frame_width

    def GetFrameHeight(self) -> int:
        """
        Returns the height of the frame captured by the camera.

        Returns:
            int: The height of the frame.
        """
        return self._frame_height



if __name__ == '__main__':
    import sys
    import logging
    from log_config import configure_logging
    configure_logging()
    

    logger.critical('starting camera.py module AS MAIN')

    # connect to camera
    # example connection string "rtsp://admin:@192.168.50.30:554/h264Preview_01_main"
    with Camera("my_camera", 0 , 30) as camera:
        print('creating queue')
        queue = Queue()

        print('subscribing to queue')
        camera.Subscribe_queue(queue)


        # create filemanager
        from filemanager import FileManager

        with FileManager(camera.GetFrameWidth(), camera.GetFrameHeight(), 30, 
                         os.path.join('E:','security_camera','data')) as filemanager:
            
            logger.info('subscribing filemanager')
            camera.Subscribe_queue(filemanager.GetQueue())

            logger.info('starting camera')
            camera.Start()

            logger.info('starting filemanager')
            filemanager.Start()

            logger.info('starting video')
            while True:
                frame = queue.get()
                assert isinstance(frame, np.ndarray), f'frame is not an ndarray, frame is: {type(frame)}'
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # camera.Stop()
                    break
            # # display video in new window
            # for frame in camera.GetFrame():
            #     cv2.imshow('frame',frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

    cv2.destroyAllWindows()
