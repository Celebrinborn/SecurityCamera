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
    _screenshot_queue:Queue[ScreenshotRequest]
    last_screenshot_sent_at:datetime.datetime
    _ip_address:str
    _task: asyncio.Task
    _abort = {'abort':False} # a dict so that you can pass by reference 
    _event_loop: asyncio.AbstractEventLoop
    _thread: threading.Thread


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
        """
        This method adds the screenshot to the queue to be sent to the remote server. The actual sending is done by a coroutine. 

        :param screenshot: A numpy ndarray of the screenshot image
        :param camera_name: A string representing the name of the camera that took the screenshot
        """

        # sanity check to make sure that the event loop is actually running
        assert self._event_loop.is_running(), f'error: ImageRecognitionSender event loop is not running'

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
        self._screenshot_queue.put(request) 
        
        # example post request for later
        # response = requests.post(url, data={'event_id': event_id, 'datetime': datetime
        # , 'camera_name': camera_name}, files={'screenshot': screenshot_stream})

    async def _send_post_request(url: str, data: dict, files: dict):
        logger.debug(f"Sending POST request to {url} with data {data}")
        _response_timer = datetime.datetime.now()
        async with requests.post(url, data=data, files=files) as response:
            if 200 < response.status_code < 300:
                logger.info(f"Successful response: {response.status_code}")
            elif 300 <= response.status_code < 400:
                raise NotImplementedError(f"Redirection response: {response.status_code}")
            elif 400 <= response.status_code < 500:
                logger.error(f"Client error response: {response.status_code}")
            elif 500 <= response.status_code:
                logger.error(f"Server error response: {response.status_code}")
            logger.debug(f"Sent POST request to {url} with data {data}. Response: {response.status_code}. Duration: {datetime.now}")
            return response

    async def _queue_sender_task(self, screenshot_queue, abort):
        while True:
            if abort['abort'] == True:
                if screenshot_queue.empty():
                    break
            try:
                screenshotRequest: self.ScreenshotRequest = await screenshot_queue.get(timeout=1)
            except queue.Empty:
                # if 1 second passes, check if we are aborting again then keep waiting
                continue
            try:
                response = await self._send_post_request(
                    url=self._ip_address, data=screenshotRequest._data
                    , files=screenshotRequest._files)
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}", exc_info=True)
                raise e

    def __init__(self, ip_address: str):
        """
        Initialize the ImageRecognitionSender with the given IP address and screenshot queue

        :param ip_address: The IP address of the remote server
        """
        self._ip_address = ip_address
        self.last_screenshot_sent_at = datetime.datetime.fromtimestamp(0) # zero out the last timestamp
        self._screenshot_queue = Queue()
        self._abort = {'abort':False}

    def Abort(self, timeout=0, *, force=False):
        if force:
            self._task.cancel()
        else:
            self._abort['abort'] = True
            self._event_loop.call_later(timeout, self._event_loop.stop)
    
    def _create_event_loop(self, screenshot_queue, abort):
        _event_loop = asyncio.new_event_loop()
        _task = self._event_loop.run_until_complete(self._queue_sender_task(screenshot_queue, abort))

    def __enter__(self):
        print(dir())
        self._thread = threading.Thread(target=self._create_event_loop, daemon=True,
                                  name=f"ImageRecognitionSender-worker"
                                  , args=(self._screenshot_queue, self._abort))
        self._thread.start()
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.Abort(force=True)
        self._event_loop.close()