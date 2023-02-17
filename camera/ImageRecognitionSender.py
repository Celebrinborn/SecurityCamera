import asyncio
import numpy as np
import requests
import datetime
import io
from queue import Queue
import threading
import logging
import pytest
import uuid
from typing import List, NamedTuple


logger = logging.getLogger(__name__)

class ImageRecognitionSender:
    class ScreenshotRequest(NamedTuple):
        data: dict
        files: dict
    _screenshot_queue:Queue[ScreenshotRequest]
    last_screenshot_sent_at:datetime.datetime
    _ip_address:str
    _tasks: List[asyncio.Task] = []

    


    def _get_filename() -> str:
        """
        Get the filename for a screenshot based on its timestamp

        :param timestamp: The timestamp of the screenshot
        :return: The filename of the screenshot
        """
        current_datetime = datetime.datetime.now()
        unix_time = int(datetime.datetime.timestamp(current_datetime))

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

        request = self.ScreenshotRequest(data=data, files=files)
        self._screenshot_queue.put(request)
        
        # example post request for later
        # response = requests.post(url, data={'event_id': event_id, 'datetime': datetime
        # , 'camera_name': camera_name}, files={'screenshot': screenshot_stream})

    async def _send_post_request(url: str, data: dict, files: dict):
        async with requests.post(url, data=data, files=files) as response:
            return response

    async def _send_screenshots_task(self):
        while True:
            _data, _files = self._screenshot_queue.get()
            # _data = _request.data
            # _files = _request.files
            _ip_address = self._ip_address
            print('debug', type(_ip_address), type(_data), type(_files))
            response = await self._send_post_request(url=_ip_address, data=_data, files=_files)
            if 200 < response.status_code < 300:
                logger.info(f"Successful response: {response.status_code}")
            elif 300 <= response.status_code < 400:
                raise NotImplementedError(f"Redirection response: {response.status_code}")
            elif 400 <= response.status_code < 500:
                logger.error(f"Client error response: {response.status_code}")
            elif 500 <= response.status_code:
                logger.error(f"Server error response: {response.status_code}")




    def __init__(self, ip_address: str):
        """
        Initialize the ImageRecognitionSender with the given IP address and screenshot queue

        :param ip_address: The IP address of the remote server
        """
        self._ip_address = ip_address
        self.last_screenshot_sent_at = None
        self._screenshot_queue = Queue()
        self._tasks = [asyncio.create_task(self._send_screenshots_task())]