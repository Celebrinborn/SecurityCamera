from camera.camera import Camera
import os
import numpy as np
import asyncio
import time
from queue import Queue
from dataclasses import dataclass

def test_GetFrame():
    with Camera('test', os.path.join('tests', 'test_data', 'testcamerafootage.avi'), 15) as camera:
        # height, width, number of channels in image
        # height = img.shape[0]
        # width = img.shape[1]
        # channels = img.shape[2]
        frame_generator = camera.GetFrame()
        f1 = next(frame_generator)

        assert isinstance(f1, np.ndarray), 'frame is not an numpy ndarray'
        assert len(f1.shape) == 3, 'frame does not have the shape of x, y, channel count'

        assert isinstance(camera.currentFrame, np.ndarray)
        assert isinstance(camera.prevFrame, np.ndarray)

@dataclass
class cQueue:
    frameQueue: Queue

async def camera_manager(frameQueueObject:cQueue):
    with Camera('test', os.path.join('tests', 'test_data', 'testcamerafootage.avi'), 15) as camera:
        fps = 25
        timeDelay = 1/fps
        _loop_start = time.perf_counter()
        _frameCounter = 0
        for frame in camera.GetFrame():
            _frameCounter = _frameCounter + 1
            if _frameCounter == 25: break
            print(f'adding frame {frame.shape}')
            frameQueueObject.frameQueue.put(frame)
            print(frameQueueObject.frameQueue.qsize())
            _start_of_next_loop = _loop_start + timeDelay
            _sleeptime = _start_of_next_loop - time.perf_counter()
            print(_sleeptime)
            await asyncio.sleep(_sleeptime)
            _loop_start = time.perf_counter()
    return True
def test_GetFrameAsync():
    mock_queueOutput = cQueue(Queue(0))
    mock_queueOutput.frameQueue = Queue(maxsize = 0)
    asyncio.run(camera_manager(frameQueueObject=mock_queueOutput))
    assert not mock_queueOutput.frameQueue.empty , 'test_GetFrameAsync queue was not populated'

    queueList = []
    while not mock_queueOutput.frameQueue.empty: queueList.append(mock_queueOutput.frameQueue.get())
    
    for frame in queueList:
        assert isinstance(frame, np.ndarray), 'frame in queuelist is not a numpy ndarray'
        assert len(frame.shape) == 3, 'frame in queuelist does not have the shape of x, y, channel count'