import asyncio
import numpy as np
import requests
import datetime
import io
from queue import Queue
import queue
import threading
import logging
import pytest
import uuid
from typing import List, NamedTuple, Tuple


logger = logging.getLogger(__name__)

class ImageRecognitionSender:
    class ScreenshotRequest(NamedTuple):
        data: dict
        files: dict
    last_screenshot_sent_at:datetime.datetime
    _ip_address:str


    def _get_filename(self, file_datetime:datetime.datetime) -> str:
        """
        Get the filename for a screenshot based on its timestamp

        :param timestamp: The timestamp of the screenshot
        :return: The filename of the screenshot
        """
        unix_time = int(datetime.datetime.timestamp(file_datetime))

        return f'{unix_time}.npy'
    
    def _priority(self):
        #TODO: Implement priority logic
        # based on unix datetimestamp which is the number of seconds from some random date in the 70s.
        # works well as for a priority queue
        return f'str(int(datetime.datetime.timestamp(datetime.datetime.now())))'

    def Send_screenshot(self, screenshot:np.ndarray, camera_name:str):
        # Save the screenshot to an IO stream
        screenshot_stream = io.BytesIO()
        np.save(screenshot_stream, screenshot)
        screenshot_stream.seek(0)

        data = {
            'event_id': str(uuid.uuid4())
            , 'timestamp': datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S_%f')
            , 'camera_name': camera_name
            , 'priority': self._priority()
        }
        files = {
            f'str(int(datetime.datetime.timestamp(datetime.datetime.now())))': screenshot_stream
        }

        # Log file name and size before sending the screenshot
        file_name = list(files.keys())[0]
        file_size = len(files[file_name].getbuffer())
        logger.debug(f"adding screenshot to queue: file_name={file_name}, file_size={file_size} bytes at {datetime.datetime.now()}")

        request = self.ScreenshotRequest(data=data, files=files)

        #todo: make this async
        self._send_post_request(request.data, request.files)

        # example post request for later
        # response = requests.post(url, data={'event_id': event_id, 'datetime': datetime
        # , 'camera_name': camera_name}, files={'screenshot': screenshot_stream})

    def _send_post_request(self, data: dict, files: dict):
        logger.debug(f"Sending POST request to {self._ip_address} with data {data}")
        _response_timer = datetime.datetime.now()
        with requests.post(self._ip_address, data=data, files=files, timeout=0.25) as response:
            if 200 < response.status_code < 300:
                logger.info(f"Successful response: {response.status_code}")
            elif 300 <= response.status_code < 400:
                raise NotImplementedError(f"Redirection response: {response.status_code}")
            elif 400 <= response.status_code < 500:
                logger.error(f"Client error response: {response.status_code}")
            elif 500 <= response.status_code:
                logger.error(f"Server error response: {response.status_code}")
            logger.debug(f"Sent POST request to {self._ip_address} with data {data}. Response: {response.status_code}. Duration: {datetime.datetime.now() - _response_timer}. Response headers: {response.headers}")
            return response
        
    def __init__(self, ip_address):
        self.last_screenshot_sent_at = datetime.datetime.fromtimestamp(0)
        self._ip_address = ip_address

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass