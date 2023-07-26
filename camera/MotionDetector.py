import cv2
import os
import logging
import threading
from queue import Queue, LifoQueue
import numpy as np
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, TooManyRedirects, InvalidURL, InvalidHeader, RetryError, RequestException
import time
import base64
import json

import uuid

logger = logging.getLogger(__name__)


class MotionDetector:
    _kill_the_daemon_event: threading.Event()  # flag to abort capture daemon
    def __init__(self, camera_name:str, threshold=1000, mask=None, detector_post_cooldown_seconds:float=1.0):
        """
        Constructor for MotionDetector class.

        Args:
            camera_name: str representing the camera name
            frame: ndarray representing the initial frame to use for motion detection.
            threshold: int representing the minimum contour area required to be considered motion.
            mask: ndarray representing a mask to apply to the frame. Defaults to None.
            detector_post_cooldown_seconds: float representing minimum time between post requests to detector service in seconds
        """
        self.camera_name = camera_name
        self.threshold = threshold
        self.mask = mask
        self.detector_post_cooldown_seconds = detector_post_cooldown_seconds

        # Initialize the queue object
        self.inbound_frame_queue = Queue()
        self._kill_the_daemon_event = threading.Event()

        # initilize the queue for send_detection_post_request
        self.post_request_stack = LifoQueue()

    def __enter__(self):
        logger.debug('running motionDetector class enter')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.Stop()
        logger.debug('running motionDetector class exit')

    def GetQueue(self):
        """
        Returns the queue object associated with this instance of the motion detector.

        Returns:
            Queue: The associated queue object.
        """
        logger.debug(f'retriving motion_detector queue {type(self.inbound_frame_queue)}')
        return self.inbound_frame_queue
    
    def _start_send_post_request_thread(endpoint:str, post_request_stack:Queue, camera_name:str, _kill_the_daemon_event:threading.Event, minimum_post_request_delay:float = 5.0):
            logger.info('entered _start_send_post_request_thread')
            last_request_timestamp = time.time()

            # session
            # session = requests.Session()

            while _kill_the_daemon_event:
                # avoid spamming the system
                _delay = time.time() - last_request_timestamp
                if _delay < minimum_post_request_delay:
                    logger.debug(f'sleeping {minimum_post_request_delay - _delay} seconds')
                    time.sleep(minimum_post_request_delay - _delay)

                _top_stack_item = post_request_stack.get()
                frame, guid = _top_stack_item
                # convert guid to str if it isn't already
                guid = guid if isinstance(guid, str) else str(guid)

                # Convert the image to a JPEG file in memory
                success, encoded_image = cv2.imencode('.jpg', frame)
                if not success:
                    raise ValueError("Could not encode image")
                # Encode this memory file in base64
                image_base64 = base64.b64encode(encoded_image).decode('utf-8')

                # Construct the JSON body
                json_body = {
                    "priority": time.time() + (time.time() - last_request_timestamp),
                    "camera_name": camera_name,
                    "image_guid": guid,
                    "timestamp": time.time(),
                    "timeout": 5*60,
                    "frame": image_base64
                }

                # Make a copy of json_body without the 'frame' property
                json_body_log = json.dumps({k: v for k, v in json_body.items() if k != 'frame'})
                
                # Convert the Python dictionary to a JSON string
                json_body_dumps = json.dumps(json_body)

                # Send the request
                _headers = {'Content-Type': 'application/json'}
                logger.debug(f'sending post request endpoint: {endpoint} data: {json_body_log} headers: {_headers} GUID: {guid}')
                try:
                    response = requests.post(endpoint, data=json_body_dumps, headers=_headers, timeout=15)
                    response.raise_for_status()  # If the response was successful, no Exception will be raised
                    # The raise_for_status() method will check the HTTP response status code.
                    # If the status code indicates a successful request (i.e., 200-399), 
                    # it will return None and the program continues to execute.
                    # However, if the status code indicates an error (i.e., 400-599),
                    # it raises a requests.exceptions.HTTPError exception.

                    logger.info(f'response was: {response.json()}')
                except HTTPError as e:
                    if response.status_code == 404:
                        logger.warning(f'404 error occurred while sending post request. Endpoint: {endpoint}')
                    elif response.status_code == 409:
                        logger.warning(f'409 error occured. this indicates a duplicate GUID {guid}')
                    else:
                        logger.warning(f'{response.status_code} occurred while sending post request. Endpoint: {endpoint}, Data: {json_body_log}, Headers: {_headers}, Exception: {e}')
                except ConnectionError as e:
                    logger.warning(f'ConnectionError occurred while sending post request. Endpoint: {endpoint} exception {e}')
                except Timeout as e:
                    logger.warning(f'Timeout occurred while sending post request. Endpoint: {endpoint} Exception: {e}')
                except TooManyRedirects as e:
                    logger.warning(f'TooManyRedirects occurred while sending post request. Endpoint: {endpoint} Exception: {e}')
                except InvalidURL as e:
                    logger.warning(f'InvalidURL occurred while sending post request. Endpoint: {endpoint} Exception: {e}')
                except InvalidHeader as e:
                    logger.warning(f'InvalidHeader occurred while sending post request. Endpoint: {endpoint}, Headers: {_headers}, Exception: {e}')
                except RetryError as e:
                    logger.warning(f'RetryError occurred while sending post request. Endpoint: {endpoint}, Data: {json_body_log}, Headers: {_headers}, Exception: {e}')
                except RequestException as e:  # This will catch any other exceptions from the requests library
                    logger.exception(f'An unknown error occurred while sending post request. Endpoint: {endpoint}, Data: {json_body_log}, Headers: {_headers}, Exception: {e}')
            logger.warning(f'exiting send_requests thread. _kill_the_daemon_event is {_kill_the_daemon_event}')

    
    def _start_motion_detection_thread(queue:Queue, _kill_the_daemon_event: threading.Event(), threshold:int, detector_post_cooldown_seconds:float, post_request_stack:LifoQueue):
        def detect_motion(threshold, frame, drawFrame=False):
            """
            Generator function that detects motion between frames.

            Args:
                threshold (int): motion threshold to trigger detection
                frame (ndarray): first frame to initialize previous frame

            Yields:
                bool: True if motion is detected, False otherwise
            """
            frame_height, frame_width, _ = frame.shape
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame = frame # preload the frame
            while True:
                # Convert frames to grayscale
                grey = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
                blured_grey = cv2.GaussianBlur(grey, (21, 21), 0)

                # Find absolute difference between frames
                diff = cv2.absdiff(prev_frame, blured_grey)

                # Apply threshold to difference image
                thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

                # Dilate thresholded image to fill in holes
                thresh = cv2.dilate(thresh, None, iterations=2)

                # Find contours in thresholded image
                cnts, res = cv2.findContours(thresh.copy(),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Check if motion is detected
                bounding_image = None
                is_motion = False
                if drawFrame:
                    bounding_image = current_frame.copy()
                for contour in cnts:
                    if cv2.contourArea(contour) < threshold:
                        continue
                    is_motion = True
                    if drawFrame:
                        # Draw bounding box on frame with motion
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(bounding_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                prev_frame = grey  # update previous frame with current frame

                if drawFrame:
                    current_frame = yield is_motion, bounding_image
                else:
                    current_frame = yield is_motion, None
        
        # get first frame
        current_frame = queue.get()
        logger.debug(f'initial pull of queue from motiondetector is {type(current_frame)}')

        # type check
        if not isinstance(current_frame, np.ndarray): raise TypeError(f'currentframe is type {type(current_frame)} should be np.ndarray')

        # initiate motion_detector
        logger.debug('initiating detect_motion')
        detector = detect_motion(threshold= threshold, frame=current_frame, drawFrame=False) # todo need to implement logic to have drawframe populate queues like Camera does
        next(detector)
        last_post_request_time = time.time()
        while not _kill_the_daemon_event.is_set():
            # logger.debug(f'current frame type: {type(current_frame)}')
            frame = queue.get()
            if not isinstance(frame, np.ndarray):
                logger.warning(f'queue has a non-ndarray passed. type was {type(frame)}')
                continue
            isMotion, _frame = detector.send(frame)

            # # debug code, comment out in production!!!!
            # cv2.imshow('motion', _frame)
            # cv2.waitKey(1)

            

            if isMotion:
                # avoid overloading the endpoint
                if time.time() - last_post_request_time > 5:
                    last_post_request_time = time.time()
                    logger.debug(f'motion detected, adding frame to stack, time has been {time.time() - last_post_request_time} out of {detector_post_cooldown_seconds}')
                    try:
                        # TODO: Change this so that the GUID is linked to the frame not generated on the fly like this
                        guid = uuid.uuid4()
                        post_request_stack.put((frame, guid))
                    except queue.TimeoutError:
                        logger.warning('timed out while attempting to add frame')
            else:
                pass
                # logger.debug('no motion detected')
        logger.warning('shutting down motion detector object')
    


    def Start(self):
        logger.debug(f'type of queue is {type(self.inbound_frame_queue)}')
        logger.info('starting _start_motion_detection_thread')
        thread = threading.Thread(target=MotionDetector._start_motion_detection_thread,
            name="motion_detection_thread",
            daemon=True,
            args=(self.inbound_frame_queue, self._kill_the_daemon_event, self.threshold, self.detector_post_cooldown_seconds, self.post_request_stack))
        thread.start()


        endpoint = os.environ['motion_detector_endpoint'] if 'motion_detector_endpoint' in os.environ else r'http://127.0.0.1:6666/detect_objects'
        if 'motion_detector_endpoint' not in os.environ:
            logger.warning(f'unable to find motion_detector_endpoint, defaulting to {endpoint}')

        logger.info('starting _start_send_post_request_thread')
        thread = threading.Thread(target=MotionDetector._start_send_post_request_thread,
            name="send_post_requests_thread",
            daemon=True,
            args=(endpoint, self.post_request_stack, self.camera_name, self._kill_the_daemon_event, self.detector_post_cooldown_seconds))
        thread.start()


    def Stop(self):
        """
        Stops the camera worker daemon
        """
        self._kill_the_daemon_event.set()  # set flag to stop capture thread


