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
from camera.sqlmanager import SQLManager

import pandas as pd

from collections import namedtuple

logger = logging.getLogger(__name__)


motion = namedtuple('motion', ['mean', 'p25', 'p50', 'p75', 'p95', 'std'])

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
    _outbound_frame_queue: Queue = Queue() # consumed by app for previewing motion detection
    _kill_motion_detection_thread: threading.Event
    _motion_detection_thread: threading.Thread
    _kill_post_thread: threading.Event
    _post_thread: threading.Thread
    _last_post_request_timestamp: float = 0.0
    _last_motion_detected_time: float = time.time()
    _sql_manager:SQLManager

    _current_motion_amount: float = 0.0

    # getter method for current_motion_amount
    @property
    def current_motion_amount(self) -> float:
        return self._current_motion_amount

    # motion log with data types of str, int, float and 4k rows initated
    _initilize_motion_logs:pd.DataFrame
    _motion_log_curser:int = 0
    _cache_size_rows:int = 4000

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
        
        self._sql_manager: SQLManager = SQLManager()
        
        # initilize motion log cache
        self._initilize_motion_logs = self._create_motion_log_cache()
        self.Start()
    
    def __enter__(self):
        return self

    def _create_motion_log_cache(self):
        # Define the DataFrame with the correct data types and preallocate space
        
        # inserts Series([frame.guid, motion_amount, time.time()])
        data = {
            'frame_guid': pd.Series([None] * self._cache_size_rows, dtype='object'),
            'motion_amount': pd.Series([int(-1)] * self._cache_size_rows, dtype='int'),
            'timestamp': pd.Series([float(-1)] * self._cache_size_rows, dtype='float')
        }
        df = pd.DataFrame(data)

        return df    
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
        self._sql_manager.AddMotion(frame.guid, int(motion_amount))
        # print(f'motion detected {motion_amount} frame {frame.guid}')
        # TODO: uncomment
        # self._outbound_post_queue.put(
        #     PostRequest(
        #         endpoint=self.alert_endpoint,
        #         camera_name=self.camera_name,
        #         frame=frame,
        #         last_motion_detected_time = self._last_motion_detected_time)
        # )
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
                self._current_motion_amount, bounding_box_image = _res
                # log motion
                if self._motion_log_curser >= self._cache_size_rows: self._motion_log_curser = 0
                self._initilize_motion_logs.iloc[self._motion_log_curser] = pd.Series([frame.guid, self._current_motion_amount, time.time()])
                self._motion_log_curser += 1

                # send frame to outbound queue
                if self._current_motion_amount > self.motion_threshold:
                    self._onMotion(frame, self._current_motion_amount)
                else:
                    self._onNoMotion(frame)

    def get_average_motion(self, seconds:float = 30) -> motion:
        '''
        returns the average motion of a frame for the last x seconds
        '''
        # get the current time
        current_time = time.time()

        # get the motion log for the last x seconds
        motion_log = self._initilize_motion_logs[self._initilize_motion_logs['timestamp'] > (current_time - seconds)]

        mean = motion_log['motion_amount'].mean()
        p25 = motion_log['motion_amount'].quantile(0.25)
        p50 = motion_log['motion_amount'].quantile(0.50)
        p75 = motion_log['motion_amount'].quantile(0.75)
        p95 = motion_log['motion_amount'].quantile(0.95)
        std = motion_log['motion_amount'].std()
        # return in named tuple
        return motion(mean, p25, p50, p75, p95, std)
        

        


    @staticmethod
    def _preprocess_frame(frame:Frame) -> Frame:
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # type: ignore
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0) # type: ignore
        return frame.preserve_identity_with(prepared_frame)
    
    @staticmethod
    def _detect_motion(frame:Frame, render_image = False) -> Generator[Optional[Tuple[float, np.ndarray]], Frame, None]:
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

            if render_image:
                contour_image = frame.copy()
                cv2.drawContours(contour_image, [contour for contour in contours if cv2.contourArea(contour) > contourAreaThreshold], -1, (0, 255, 0), 3) # type: ignore
            
            prev_frame = prepared_frame
            frame = yield area_of_motion_detected, contour_image if render_image else frame # type: ignore

    def _send_post_requests_thread(self):
        while not self._kill_post_thread.is_set():
            postRequest:PostRequest = self._outbound_post_queue.get()
            print(f'sending post request: {datetime.datetime.now()}')
            threading.Thread(target=postRequest.send, daemon=True).start()

