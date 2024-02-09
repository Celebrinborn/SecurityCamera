import logging
import platform
import subprocess
import re


logger = logging.getLogger(__name__)
# import cv2
import av
import av.container
import av.video
import av.error
import av.codec
import av.codec
import numpy as np
import os
from queue import Queue
import threading
import time
import inspect

from camera.frame import Frame
from camera.resolution import Resolution
import warnings

import typing
from typing import Generator, Optional, Union


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

    def _add_frame_to_queues(self, frame: Frame):
        assert isinstance(frame, Frame), 'camera is attempting to put a non-Frame object into a queue'
        for queue in self._subscribed_queues:
            queue.put(frame)
    

class Camera:
    """
    Class for a camera that captures frames and sends them to subscribed queues.
    """
    _camera: av.container.InputContainer
    _stream: av.video.VideoStream
    _camera_url: Union[str, int]
    _max_fps: int
    _killDaemon: bool  # flag to abort capture daemon

    prevFrame: Frame
    currentFrame: Frame

    camera_thread:threading.Thread # the thread that captures frames from the camera

    _time_of_last_frame:float = 0.0


    @staticmethod
    def find_ip_address(input_string):
        # Regular expression pattern for matching an IP address
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        
        # Search for the pattern in the input string
        match = re.search(ip_pattern, input_string)
        
        # If a match is found, return the matched IP address, else return None
        return match.group(0) if match else None
    @staticmethod
    def _ping(host):
        # Determine the current operating system
        current_os = platform.system().lower()
        
        # Command prefix: different options for Windows
        command_prefix = ['ping', '-c', '1'] if current_os != "windows" else ['ping', '-n', '1']
        
        # Construct the full command
        command = command_prefix + [host]
        
        try:
            # Execute the ping command
            output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Return True if the host is reachable (returncode is 0), False otherwise
            return output.returncode == 0
                
        except Exception:
            return False

    def _connect_to_camera(self, camera_url:Union[str, int]):
        # close the camera if it is already open
        logger.debug('checking if _camera attribute exists')
        if hasattr(self, '_camera'):
            logger.debug('closing camera')
            self._camera.close()
        
        # connect to the camera
        try:
            logger.debug(f'connecting to camera {camera_url}')
            options = {'rtsp_transport': 'tcp'}
            self._camera = av.open('rtsp://admin:@10.1.1.17:554/h264Preview_01_main', options=options)
            
            logger.debug('getting video stream')
            stream = next((s for s in self._camera.streams if s.type == 'video'), None)
            if stream is None:
                logger.error(f'no video stream found for camera {self._camera_url}')
                raise NotImplementedError("error handling for a connection refusal has not been implemented yet")
                return False
            self._stream = stream

            self._frame_width = self._stream.width
            self._frame_height = self._stream.height

            logger.debug(f'camera connected successfully')

        except av.AVError as e:
            #TODO: handle the error
            logger.exception(f'camera {self._camera_url} refused to connect. There may already be an active connection to the camera')
            raise NotImplementedError("error handling for a connection refusal has not been implemented yet") from e
        except Exception as e:
            logger.exception('An unexpected error occurred while connecting to the camera.', exc_info=True, stack_info=True)
            raise e

    def __init__(self, camera_url:Union[str, int], max_fps:int) -> None:

        """
        Initializes the camera instance.

        Parameters:
            camera_name (str): The name of the camera.
            camera_url (str): The URL of the camera stream.
            max_fps (int): The maximum number of frames per second that the camera should capture.
            cv2_module (cv2.VideoCapture): The module to use for capturing the frames.
        """
        self._camera_url = camera_url
        self._max_fps = max_fps

        # checking if the camera_url is an IP address
        if isinstance(camera_url, str):
            ip_address = self.find_ip_address(camera_url)
            if ip_address:
                logger.debug(f'camera_url is an IP address: {camera_url}')
            else:
                logger.debug(f'camera_url is not an IP address: {camera_url}')
        # ping the camera to check if it is reachable
        if ip_address:
            logger.debug(f'pinging camera {camera_url}')
            if not self._ping(ip_address):
                logger.error(f'camera {camera_url} is not reachable')
            else:
                logger.debug(f'camera {camera_url} is reachable')

        # log av.__version__
        logger.debug(f'av.__version__ = {av.__version__}')

        logger.debug(f'codexes available: {[codex for codex in av.codec.codecs_available]}')


        # connect to the camera
        logger.debug(f'connecting to camera {camera_url}')
        self._connect_to_camera(camera_url)

        self._subscription_manager = SubscriptionManager()

        logger.debug(f'starting to read frames from camera {camera_url}')
        self.Start()

        # start the health check thread
        # logger.info(f'starting health monitor for camera {camera_url}')
        self._health_check_thread = threading.Thread(target=self._heartbeat, name='Camera_Health_Check', daemon=True)
        self._health_check_thread.start()

        logger.info(f'camera {camera_url} has been initialized')

    def _heartbeat(self):
        logger.info(f'starting health monitor for camera {self._camera_url}')
        while True:
            if time.time() - self._time_of_last_frame > 5:
                logger.warning(f'heartbeat camera {self._camera_url} has not received a frame in 5 seconds')

                # kill the camera thread
                logger.info(f'killing camera thread {self._camera_url}')
                self.Stop()
                logger.info(f'waiting for camera thread {self._camera_url} to die')
                self.camera_thread.join()

                logger.info(f'restarting camera thread {self._camera_url}')
                self.Start()
            else:
                logger.debug(f'heartbeat camera {self._camera_url} has received a frame {time.time() - self._time_of_last_frame:.02f} seconds ago')
            time.sleep(30)

    # Define the __enter__ method for the Camera class
    def __enter__(self):
        """
        Enters the camera instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the camera instance.
        """
        self.Stop()
        if hasattr(self, '_camera'):
            self._camera.close()

    def GetFrame(self) -> Generator[Frame, None, None]:
        """
        Captures frames from the camera.

        Returns:
            Frame: The current frame from the camera.
        """
        
        for i in range(10):
            try:
                # logger.debug('starting Camera class getframe')
                for packet in self._camera.demux(self._stream):
                    try:
                        np_frames:list[np.ndarray] = [frame.to_ndarray(format='bgr24') for frame in packet.decode()]
                    except av.error.InvalidDataError as e:
                        logger.warning(f'camera {self._camera_url} has experienced an InvalidDataError while reading frames, skipping frame', exc_info=True, stack_info=True)
                    except av.error.CorruptDataError as e:
                        logger.warning(f'camera {self._camera_url} has experienced a CorruptDataError while reading frames, skipping frame', exc_info=True, stack_info=True)
                    except Exception as e:
                        logger.exception(f'camera {self._camera_url} has experienced an unexpected error while reading frames', exc_info=True, stack_info=True)                        
                    # suppress the Frame creation warning and create the frame
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        frames:list[Frame] = [Frame(np_frame) for np_frame in np_frames]
                    for frame in frames:
                        yield frame
            except av.error.EOFError as e:
                logger.exception("av.error.EOFError: The camera has stopped sending frames. The camera may have been disconnected or the stream may have ended.", exc_info=True, stack_info=True)
                raise NotImplementedError("error handling for a connection refusal has not been implemented yet") from e
                #TODO: handle the error
            except Exception as e:
                logger.exception(f'camera {self._camera_url} has experienced an error while reading frames, retrying attempt {i} of 10', exc_info=True, stack_info=True)
                time.sleep(1)
        raise NotImplementedError("error handling for a full demux exception has not been implemented yet")




        while True: #self._camera == True:
            # read the camera frame to a temp variable
            ret, _frame = self._camera.read()
            if not ret:
                # if a bad frame is sent then continue
                continue
            # suppress the Frame creation warning and create the frame
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                newFrame = Frame(_frame)
            # assign the prev frame to what the current frame is
            self.prevFrame = self.currentFrame
            # now assign the current frame to the newly read frame
            self.currentFrame = newFrame
            self._time_of_last_frame = time.time()
            yield self.currentFrame
    
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
    def Start(self):
        """
        Starts the worker thread that reads from the camera and adds frames to any subscribed queues.
        """
        _caller = inspect.stack()[1]
        logger.debug(f'Starting Camera.Start() from {_caller.filename}:{_caller.lineno}')

        self._killDaemon = False  # initialize flag to False
        def _capture(subscriptionManager:SubscriptionManager, fps:int):
            """Captures frames from the camera and adds them to subscribed queues.

            Args:
                subscriptionManager: An instance of `Camera.SubscriptionManager` to manage
                    the subscribed queues.
                fps: An integer representing the desired max frames per second (FPS) rate.

            Returns:
                None
            """
            last_frame_time = time.time()
            fps_cache = 1/fps
            logger.debug(f'fps_cache = {fps_cache}')
            for frame in self.GetFrame():
                if self._killDaemon:  # check flag to stop thread
                    break
                # check if time sense last frame is less than 1/fps
                if time.time() - last_frame_time < fps_cache:
                    # logger.debug(f'frame rate is too high, skipping frame')
                    continue
                last_frame_time = time.time()
                subscriptionManager._add_frame_to_queues(frame)
        self.camera_thread = threading.Thread(target=_capture, 
                                  name="Camera_Thread", 
                                  daemon=True,
                                  args=(self._subscription_manager, self._max_fps))
        self.camera_thread.start()
    
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
    def GetCameraResolution(self) -> Resolution:
        return Resolution(self.GetFrameWidth(), self.GetFrameHeight())


if __name__ == '__main__':
    import sys
    import logging
    from log_config import configure_logging

    from camera.filemanager import VideoFileManager
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
        from filemanager import VideoFileManagerOld

        with VideoFileManagerOld(camera.GetFrameWidth(), camera.GetFrameHeight(), 30, 
                         os.path.join('E:','security_camera','data'), 'test_camera') as filemanager:
            
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
