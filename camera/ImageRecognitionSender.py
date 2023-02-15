import asyncio
import numpy as np
import requests
import datetime
import io
import queue
import threading
import logging
import pytest

logger = logging.getLogger(__name__)

class ImageRecognitionSender:
    def __init__(self, ip_address: str, screenshot_queue: queue.Queue):
        self.ip_address = ip_address
        self.last_screenshot_sent_at = None
        self.screenshot_queue = screenshot_queue

    def _get_filename(self, timestamp: int) -> str:
        return str(timestamp) + ".npy"

    async def send_screenshot(self, screenshot: np.ndarray):
        now = datetime.datetime.now()
        if self.last_screenshot_sent_at is not None and (now - self.last_screenshot_sent_at).total_seconds() < 1:
            raise Exception("Rate limit exceeded")

        timestamp = int(datetime.datetime.timestamp(now))
        filename = self._get_filename(timestamp)

        file = io.BytesIO()
        np.save(file, screenshot)
        # Move the file pointer back to the start of the file so that the entire contents of the file can be read and sent in the requests.post call
        file.seek(0)

        async with requests.post(f"http://{self.ip_address}/upload", files={filename: file}) as response:
            if response.status_code == 200:
                logger.info(f"Sent screenshot {filename} successfully")
                self.last_screenshot_sent_at = now
            else:
                logger.error(f"Failed to send screenshot {filename}: {response.text}")
                raise requests.HTTPError(f"Failed to send screenshot: {response.text}")


    def run_async(self):
        asyncio.run(self._run_async())

    async def _run_async(self):
        while True:
            screenshot = self.screenshot_queue.get()
            await self.send_screenshot(screenshot)
