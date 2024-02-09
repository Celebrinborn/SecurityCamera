import os
from queue import Queue
import tempfile
import time
from camera.camera import Camera

import logging
import numpy as np
import cv2
import pytest
from camera.frame import Frame
from typing import Iterator

@pytest.fixture
def video_source() -> Iterator[str]:
    # Create a temporary file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file_path: str = temp_video_file.name
    temp_video_file.close()  # Close the file so that CV2 can open and write to it

    # Define video parameters
    import cv2  # Import the missing module

    width, height, fps = 640, 480, 30
    frame_count = 60  # Example frame count

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #type: ignore
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    # Generate and write frames to the video file
    for _ in range(frame_count):
        # Create a dummy frame, each frame is a numpy array of shape (height, width, 3) and dtype uint8 and zeroed out
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)

    # Clean up
    out.release()

    # Yield the path to the video file for tests to use
    yield file_path

    # Cleanup code after test use
    os.remove(file_path)




def test_Camera_ContextManager(video_source):
    with Camera(video_source, 30) as camera:
        pass
    assert True  # if the test didn't throw an exception, it passed

def test_Camera_Subscribe(video_source):
   queue: Queue[Frame] = Queue()
   print("Testing Camera.Subscribe_queue()... expect 5 second wait")

   with Camera(video_source, 30) as camera:
         camera.Subscribe_queue(queue)
         # Wait for the camera to start reading frames
         time.sleep(5)

         frame = queue.get(timeout=6)
         assert frame is not None, "Frame is None"
         assert isinstance(frame, Frame), "Frame is not of type Frame"


def test_Camera_FrameRate(video_source):
    queue: Queue[Frame] = Queue()
    with Camera(video_source, 15) as camera:
        camera.Subscribe_queue(queue)
        # measure the time between each frame
        time_diffs = []
        print("Measuring frame rate...")
        for _ in range(10):
            print(f"Frame {_ + 1}")
            start = time.perf_counter_ns()
            frame = queue.get(timeout=2)
            assert frame is not None, "Frame is None"
            assert isinstance(frame, Frame), "Frame is not of type Frame"
            end = time.perf_counter_ns()
            elapsed_time = (end - start) / 1e9
            time_diffs.append(elapsed_time)
        # Calculate the average frame rate
        frame_rate = 1 / np.mean(time_diffs)
        # Assert that the frame rate is within 10% of the expected value
        assert np.isclose(frame_rate, 15, rtol=0.1), f"Frame rate is {frame_rate} instead of 15"
        print(f'Average frame rate: {frame_rate} fps across 10 frames')