import cv2
import os
import logging
import threading
from queue import Queue, LifoQueue
import numpy as np
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout, TooManyRedirects, InvalidURL, InvalidHeader, RetryError, RequestException
import time
import datetime
import base64
import json

from typing import Generator, Optional, Tuple, Union

from camera.frame import Frame
from dataclasses import dataclass



logger = logging.getLogger(__name__)


@dataclass
class PostRequest:
    headers = {'Content-Type': 'application/json'}
    endpoint:str
    camera_name:Union[str, int]
    frame:Frame
    last_motion_detected_time:float
    timeout:float = 5
    @property
    def data(self) -> str:
        # Convert the image to a JPEG file in memory
        success, encoded_image = cv2.imencode('.jpg', self.frame) # type: ignore
        if not success:
            raise ValueError("Could not encode image")
        # Encode this memory file in base64
        image_base64 = base64.b64encode(encoded_image).decode('utf-8')
        json_body = {
            "priority": time.time() + (time.time() - self.last_motion_detected_time),
            "camera_name": self.camera_name,
            "image_guid": str(self.frame.guid),
            "timestamp": time.time(),
            "timeout": 15,
            "frame": image_base64
        }
        return json.dumps(json_body)
    def send(self):
        '''
        sends the post requst. NOTE: BLOCKING
        '''
        print(f'sending post requst {self.frame.guid}')
        response:Union[requests.Response, None] = None
        try:
            response = requests.post(self.endpoint, data=self.data, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()  # If the response was successful, no Exception will be raised
            # The raise_for_status() method will check the HTTP response status code.
            # If the status code indicates a successful request (i.e., 200-399), 
            # it will return None and the program continues to execute.
            # However, if the status code indicates an error (i.e., 400-599),
            # it raises a requests.exceptions.HTTPError exception
            logger.info(f'response was: {response.json()}')
            print(f'response was: {response.json()}')
        except ConnectionError as e:
            logger.warning(f'ConnectionError occurred while sending post request. Endpoint: {self.endpoint} exception {e}')
        except Timeout as e:
            logger.warning(f'Timeout occurred while sending post request. Endpoint: {self.endpoint} Exception: {e}')
        except TooManyRedirects as e:
            logger.warning(f'TooManyRedirects occurred while sending post request. Endpoint: {self.endpoint} Exception: {e}')
        except InvalidURL as e:
            logger.warning(f'InvalidURL occurred while sending post request. Endpoint: {self.endpoint} Exception: {e}')
        except InvalidHeader as e:
            logger.warning(f'InvalidHeader occurred while sending post request. Endpoint: {self.endpoint}, Headers: {self.headers}, Exception: {e}')
        except RetryError as e:
            logger.warning(f'RetryError occurred while sending post request. Endpoint: {self.endpoint}, Data: {self.data}, Headers: {self.headers}, Exception: {e}')
        except HTTPError as e:
            if response is None:
                logger.critical(f'you should never be able to have response == None while having an HTTPError')
                raise Exception('you should never be able to have response == None while having an HTTPError')
            if response.status_code == 404:
                logger.warning(f'404 error occurred while sending post request. Endpoint: {self.endpoint}')
            elif response.status_code == 409:
                logger.warning(f'409 error occured. this indicates a duplicate GUID {self.frame.guid}')
            else:
                logger.warning(f'{response.status_code} occurred while sending post request. Endpoint: {self.endpoint}, Data: {self.data}, Headers: {self.headers}, Exception: {e}')
        except RequestException as e:  # This will catch any other exceptions from the requests library
            logger.exception(f'An unknown error occurred while sending post request. Endpoint: {self.endpoint}, Data: {self.data}, Headers: {self.headers}, Exception: {e}')
        print(self.frame.guid, response)




class MotionDetector:
    _inbound_frame_queue: Queue[Frame] = Queue()
    _outbound_post_queue: Queue = LifoQueue()
    _kill_motion_detection_thread: threading.Event
    _motion_detection_thread: threading.Thread
    _kill_post_thread: threading.Event
    _post_thread: threading.Thread
    _last_post_request_timestamp: float = 0.0
    _last_motion_detected_time: float = time.time()

    def __init__(self, camera_name:str, alert_endpoint:Union[str, int, None] = None, motion_threshold:Optional[int] = None, post_rate_limit:Optional[float] = None) -> None:
        self.alert_endpoint:Union[int, str] = alert_endpoint if alert_endpoint \
            else os.environ['motion_detector_endpoint'] if 'motion_detector_endpoint' in os.environ \
            else r'http://127.0.0.1:8888/detect_objects'
        self.motion_threshold = motion_threshold if motion_threshold \
            else int(os.environ['motion_threshold']) if 'motion_threshold' in os.environ and os.environ['motion_threshold'].isnumeric() \
            else 50
        self.post_rate_limit = post_rate_limit if post_rate_limit \
            else float(os.environ['object_detection_second_per_request_rate_limit']) if 'object_detection_second_per_request_rate_limit' in os.environ and os.environ['object_detection_second_per_request_rate_limit'].isnumeric() \
            else 5.0
        self.camera_name = camera_name if camera_name \
            else os.environ['camera_name'] if 'camera_name' in os.environ \
            else 0
        self.Start()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.Stop()

    def GetQueue(self) -> Queue:
            return self._inbound_frame_queue
    
    def Start(self) -> bool:
        if hasattr(self, '_video_file_manager_thread'): return False
        self._motion_detection_thread = threading.Thread(target=self._detectMotionThread, name="motion_detection_thread", daemon=True)
        self._kill_motion_detection_thread = threading.Event()
        self._motion_detection_thread.start()

        if hasattr(self, '_post_thread'): return False
        self._kill_post_thread=threading.Event()
        self._post_thread = threading.Thread(target=self._send_post_requests_thread, name="_send_post_requests_thread", daemon=True)
        self._post_thread.start()
        return True
    
    def Stop(self, blocking:bool = False, timeout:float = -1):
        self._kill_motion_detection_thread.set()
        if blocking or timeout > 0:
            self._motion_detection_thread.join(timeout=3 if timeout <= 0 else timeout)
    
    def _onMotion(self, frame:Frame, motion_amount:float) -> None:
        self._last_motion_detected_time = time.time()
        if self._last_post_request_timestamp + self.post_rate_limit > time.time():
            # if the rate limit is being exceeded then skip
            # print(f'rate limit exceeded, skipping {frame.guid} motion amount: {motion_amount} resumingin {time.time() - (self._last_post_request_timestamp + self.post_rate_limit)}')
            return
        
        # print(f'motion detected {motion_amount} frame {frame.guid}')
        self._outbound_post_queue.put(
            PostRequest(
                endpoint=self.alert_endpoint,
                camera_name=self.camera_name,
                frame=frame,
                last_motion_detected_time = self._last_motion_detected_time)
        )
        self._last_post_request_timestamp = time.time()
    def _onNoMotion(self, frame:Frame) -> None:
        # print(f'no motion detected {0} frame {frame.guid}')
        pass

    def _detectMotionThread(self) -> None:
         # assign init frame
         frame:Frame = self._inbound_frame_queue.get()
         motion = self._detect_motion(frame)
         next(motion)
         while not self._kill_motion_detection_thread.is_set():
            frame = self._inbound_frame_queue.get()
            _res = motion.send(frame)
            if _res is not None:
                motion_amount, bounding_box_image = _res
                if motion_amount > self.motion_threshold:
                    self._onMotion(frame, motion_amount)
                else:
                    self._onNoMotion(frame)

    @staticmethod
    def _preprocess_frame(frame:Frame) -> Frame:
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # type: ignore
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0) # type: ignore
        return frame.preserve_identity_with(prepared_frame)
    
    @staticmethod
    def _detect_motion(frame:Frame) -> Generator[Optional[Tuple[float, np.ndarray]], Frame, None]:
        contourAreaThreshold = 50
        prev_frame:Frame = MotionDetector._preprocess_frame(frame)
        prepared_frame:Frame
        while True:
            prepared_frame = MotionDetector._preprocess_frame(frame)
            # 3. calculate difference
            diff_frame = cv2.absdiff(src1=prev_frame, src2=prepared_frame) # type: ignore

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1) # type: ignore

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1] # type: ignore

            # 6. Find and optionally draw contours
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) # type: ignore
            
            area_of_motion_detected = sum([cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > contourAreaThreshold]) # type: ignore

            contour_image = frame
            cv2.drawContours(contour_image, [contour for contour in contours if cv2.contourArea(contour) > contourAreaThreshold], -1, (0, 255, 0), 3) # type: ignore
            
            prev_frame = prepared_frame
            frame = yield area_of_motion_detected, contour_image

    def _send_post_requests_thread(self):
        while not self._kill_post_thread.is_set():
            postRequest:PostRequest = self._outbound_post_queue.get()
            print(f'sending post request: {datetime.datetime.now()}')
            threading.Thread(target=postRequest.send, daemon=True).start()



        
# class MotionDetector:
#     _kill_the_daemon_event: threading.Event()  # flag to abort capture daemon
#     def __init__(self, camera_name:str, threshold=1000, mask=None, detector_post_cooldown_seconds:float=1.0):
#         """
#         Constructor for MotionDetector class.

#         Args:
#             camera_name: str representing the camera name
#             frame: ndarray representing the initial frame to use for motion detection.
#             threshold: int representing the minimum contour area required to be considered motion.
#             mask: ndarray representing a mask to apply to the frame. Defaults to None.
#             detector_post_cooldown_seconds: float representing minimum time between post requests to detector service in seconds
#         """
#         self.camera_name = camera_name
#         self.threshold = threshold
#         self.mask = mask
#         self.detector_post_cooldown_seconds = detector_post_cooldown_seconds

#         # Initialize the queue object
#         self.inbound_frame_queue = Queue()
#         self._kill_the_daemon_event = threading.Event()

#         # initilize the queue for send_detection_post_request
#         self.post_request_stack = LifoQueue()

#     def __enter__(self):
#         logger.debug('running motionDetector class enter')
#         return self
    
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.Stop()
#         logger.debug('running motionDetector class exit')

#     def GetQueue(self):
#         """
#         Returns the queue object associated with this instance of the motion detector.

#         Returns:
#             Queue: The associated queue object.
#         """
#         logger.debug(f'retriving motion_detector queue {type(self.inbound_frame_queue)}')
#         return self.inbound_frame_queue
    
#     def _start_send_post_request_thread(endpoint:str, post_request_stack:Queue, camera_name:str, _kill_the_daemon_event:threading.Event, minimum_post_request_delay:float = 5.0):
#             logger.info('entered _start_send_post_request_thread')
#             last_request_timestamp = time.time()

#             # session
#             # session = requests.Session()

#             while _kill_the_daemon_event:
#                 # avoid spamming the system
#                 _delay = time.time() - last_request_timestamp
#                 if _delay < minimum_post_request_delay:
#                     logger.debug(f'sleeping {minimum_post_request_delay - _delay} seconds')
#                     time.sleep(minimum_post_request_delay - _delay)

#                 _top_stack_item = post_request_stack.get()
#                 frame, guid = _top_stack_item
#                 # convert guid to str if it isn't already
#                 guid = guid if isinstance(guid, str) else str(guid)

#                 # Convert the image to a JPEG file in memory
#                 success, encoded_image = cv2.imencode('.jpg', frame)
#                 if not success:
#                     raise ValueError("Could not encode image")
#                 # Encode this memory file in base64
#                 image_base64 = base64.b64encode(encoded_image).decode('utf-8')

#                 # Construct the JSON body
#                 json_body = {
#                     "priority": time.time() + (time.time() - last_request_timestamp),
#                     "camera_name": camera_name,
#                     "image_guid": guid,
#                     "timestamp": time.time(),
#                     "timeout": 5*60,
#                     "frame": image_base64
#                 }

#                 # Make a copy of json_body without the 'frame' property
#                 json_body_log = json.dumps({k: v for k, v in json_body.items() if k != 'frame'})
                
#                 # Convert the Python dictionary to a JSON string
#                 json_body_dumps = json.dumps(json_body)

#                 # Send the request
#                 _headers = {'Content-Type': 'application/json'}
#                 logger.debug(f'sending post request endpoint: {endpoint} data: {json_body_log} headers: {_headers} GUID: {guid}')
#                 try:
#                     response = requests.post(endpoint, data=json_body_dumps, headers=_headers, timeout=15)
#                     response.raise_for_status()  # If the response was successful, no Exception will be raised
#                     # The raise_for_status() method will check the HTTP response status code.
#                     # If the status code indicates a successful request (i.e., 200-399), 
#                     # it will return None and the program continues to execute.
#                     # However, if the status code indicates an error (i.e., 400-599),
#                     # it raises a requests.exceptions.HTTPError exception.

#                     logger.info(f'response was: {response.json()}')
#                 except HTTPError as e:
#                     if response.status_code == 404:
#                         logger.warning(f'404 error occurred while sending post request. Endpoint: {endpoint}')
#                     elif response.status_code == 409:
#                         logger.warning(f'409 error occured. this indicates a duplicate GUID {guid}')
#                     else:
#                         logger.warning(f'{response.status_code} occurred while sending post request. Endpoint: {endpoint}, Data: {json_body_log}, Headers: {_headers}, Exception: {e}')
#                 except ConnectionError as e:
#                     logger.warning(f'ConnectionError occurred while sending post request. Endpoint: {endpoint} exception {e}')
#                 except Timeout as e:
#                     logger.warning(f'Timeout occurred while sending post request. Endpoint: {endpoint} Exception: {e}')
#                 except TooManyRedirects as e:
#                     logger.warning(f'TooManyRedirects occurred while sending post request. Endpoint: {endpoint} Exception: {e}')
#                 except InvalidURL as e:
#                     logger.warning(f'InvalidURL occurred while sending post request. Endpoint: {endpoint} Exception: {e}')
#                 except InvalidHeader as e:
#                     logger.warning(f'InvalidHeader occurred while sending post request. Endpoint: {endpoint}, Headers: {_headers}, Exception: {e}')
#                 except RetryError as e:
#                     logger.warning(f'RetryError occurred while sending post request. Endpoint: {endpoint}, Data: {json_body_log}, Headers: {_headers}, Exception: {e}')
#                 except RequestException as e:  # This will catch any other exceptions from the requests library
#                     logger.exception(f'An unknown error occurred while sending post request. Endpoint: {endpoint}, Data: {json_body_log}, Headers: {_headers}, Exception: {e}')
#             logger.warning(f'exiting send_requests thread. _kill_the_daemon_event is {_kill_the_daemon_event}')

    
#     def _start_motion_detection_thread(queue:Queue, _kill_the_daemon_event: threading.Event(), threshold:int, detector_post_cooldown_seconds:float, post_request_stack:LifoQueue):
#         def detect_motion(threshold, frame, drawFrame=False):
#             """
#             Generator function that detects motion between frames.

#             Args:
#                 threshold (int): motion threshold to trigger detection
#                 frame (ndarray): first frame to initialize previous frame

#             Yields:
#                 bool: True if motion is detected, False otherwise
#             """
#             frame_height, frame_width, _ = frame.shape
#             prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             current_frame = frame # preload the frame
#             while True:
#                 # Convert frames to grayscale
#                 grey = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#                 prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
#                 blured_grey = cv2.GaussianBlur(grey, (21, 21), 0)

#                 # Find absolute difference between frames
#                 diff = cv2.absdiff(prev_frame, blured_grey)

#                 # Apply threshold to difference image
#                 thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

#                 # Dilate thresholded image to fill in holes
#                 thresh = cv2.dilate(thresh, None, iterations=2)

#                 # Find contours in thresholded image
#                 cnts, res = cv2.findContours(thresh.copy(),
#                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                 # Check if motion is detected
#                 bounding_image = None
#                 is_motion = False
#                 if drawFrame:
#                     bounding_image = current_frame.copy()
#                 for contour in cnts:
#                     if cv2.contourArea(contour) < threshold:
#                         continue
#                     is_motion = True
#                     if drawFrame:
#                         # Draw bounding box on frame with motion
#                         (x, y, w, h) = cv2.boundingRect(contour)
#                         cv2.rectangle(bounding_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

#                 prev_frame = grey  # update previous frame with current frame

#                 if drawFrame:
#                     current_frame = yield is_motion, bounding_image
#                 else:
#                     current_frame = yield is_motion, None
        
#         # get first frame
#         current_frame = queue.get()
#         logger.debug(f'initial pull of queue from motiondetector is {type(current_frame)}')

#         # type check
#         if not isinstance(current_frame, np.ndarray): raise TypeError(f'currentframe is type {type(current_frame)} should be np.ndarray')

#         # initiate motion_detector
#         logger.debug('initiating detect_motion')
#         detector = detect_motion(threshold= threshold, frame=current_frame, drawFrame=False) # todo need to implement logic to have drawframe populate queues like Camera does
#         next(detector)
#         last_post_request_time = time.time()
#         while not _kill_the_daemon_event.is_set():
#             # logger.debug(f'current frame type: {type(current_frame)}')
#             frame = queue.get()
#             if not isinstance(frame, np.ndarray):
#                 logger.warning(f'queue has a non-ndarray passed. type was {type(frame)}')
#                 continue
#             isMotion, _frame = detector.send(frame)

#             # # debug code, comment out in production!!!!
#             # cv2.imshow('motion', _frame)
#             # cv2.waitKey(1)

            

#             if isMotion:
#                 # avoid overloading the endpoint
#                 if time.time() - last_post_request_time > 5:
#                     last_post_request_time = time.time()
#                     logger.debug(f'motion detected, adding frame to stack, time has been {time.time() - last_post_request_time} out of {detector_post_cooldown_seconds}')
#                     try:
#                         # TODO: Change this so that the GUID is linked to the frame not generated on the fly like this
#                         guid = uuid.uuid4()
#                         post_request_stack.put((frame, guid))
#                     except queue.TimeoutError:
#                         logger.warning('timed out while attempting to add frame')
#             else:
#                 pass
#                 # logger.debug('no motion detected')
#         logger.warning('shutting down motion detector object')
    


#     def Start(self):
#         logger.debug(f'type of queue is {type(self.inbound_frame_queue)}')
#         logger.info('starting _start_motion_detection_thread')
#         thread = threading.Thread(target=MotionDetector._start_motion_detection_thread,
#             name="motion_detection_thread",
#             daemon=True,
#             args=(self.inbound_frame_queue, self._kill_the_daemon_event, self.threshold, self.detector_post_cooldown_seconds, self.post_request_stack))
#         thread.start()


#         endpoint = os.environ['motion_detector_endpoint'] if 'motion_detector_endpoint' in os.environ else r'http://127.0.0.1:6666/detect_objects'
#         if 'motion_detector_endpoint' not in os.environ:
#             logger.warning(f'unable to find motion_detector_endpoint, defaulting to {endpoint}')

#         logger.info('starting _start_send_post_request_thread')
#         thread = threading.Thread(target=MotionDetector._start_send_post_request_thread,
#             name="send_post_requests_thread",
#             daemon=True,
#             args=(endpoint, self.post_request_stack, self.camera_name, self._kill_the_daemon_event, self.detector_post_cooldown_seconds))
#         thread.start()


#     def Stop(self):
#         """
#         Stops the camera worker daemon
#         """
#         self._kill_the_daemon_event.set()  # set flag to stop capture thread


