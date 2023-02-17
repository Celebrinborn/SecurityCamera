import pytest
import asyncio
import numpy as np
from camera.ImageRecognitionSender import ImageRecognitionSender
import datetime
import pytest_mock
from unittest.mock import MagicMock
import time



# Helper function to run the async function
async def run_async(coroutine):
    return await asyncio.get_event_loop().run_until_complete(coroutine)


# Unit tests for the ImageRecognitionSender class
class TestImageRecognitionSender:
    # Test that the send_screenshot method sends a screenshot to the server and logs a success message
    @pytest.mark.asyncio
    async def test_send_screenshot_success(self, mocker):
        # Create a mock for the _send_post_request function
        send_post_request_mock = mocker.patch(
            'camera.ImageRecognitionSender.ImageRecognitionSender._send_post_request', 
            return_value=MagicMock(status_code=200), 
            autospec=True)
        
        screenshot = np.array([[1, 2, 3], [4, 5, 6]])
        image_sender = ImageRecognitionSender("127.0.0.1")

        # Call the send_screenshot method
        image_sender.Send_screenshot(screenshot, 'test_camera')

        # Check that the screenshot was added to the queue
        assert image_sender._screenshot_queue.qsize() == 1