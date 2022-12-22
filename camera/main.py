import cv2
import os
import sys

import logging
logging.basicConfig(filename=os.path.join('logs', 'securityCamera.log'),level=logging.DEBUG)
logger = logging.getLogger(__name__)
#root = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# motion loop

# object reconition loop

# notification

import cv2
import time
import subprocess
from queue import Queue, PriorityQueue
import multiprocessing
from dataclasses import dataclass
import typing
from datetime import datetime, timedelta

@dataclass(order=True)
class YoloFrame:
    priority: int
    timestamp: datetime
    frame: typing.Any

def DetectMotion(currentFrame, prevFrame, threshold:int) -> bool:
    diff = cv2.absdiff(prevFrame, currentFrame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    isMotion = False
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < threshold:
            continue
        cv2.rectangle(prevFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        isMotion = True
    cv2.drawContours(prevFrame, contours, -1, (0, 255, 0), 2)
    return isMotion

def VideoName(camera_name:str, video_start_time:datetime) -> str:
    return os.path.join('data', f'{camera_name}_{video_start_time.strftime(r"%Y%m%d_%H%M%S")}.mp4v')
def CreateVideoWriter(camera_name:str, video_start_time:datetime, fourcc, frame_rate:int, frame_width:int, frame_height:int) -> cv2.VideoWriter:
    video_name = VideoName(camera_name, video_start_time)
    return cv2.VideoWriter(video_name, fourcc, frame_rate, (frame_width, frame_height)), video_name


def Camera(camera_name:str, camera_path:str, yoloqueue:PriorityQueue, motion_threshold:int = 900, frame_rate = 30):
    try:
        logger.info(f'loading camera {camera_name}')
        cap = cv2.VideoCapture(camera_path)
        ret, prevFrame = cap.read()
        ret, currentFrame = cap.read()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f'frame is {frame_width} x { frame_height} at {frame_rate} per second')

        video_start_time = datetime.now()

        # for windows
        #fourcc = cv2.VideoWriter_fourcc(*'xvid')
        # for linux
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer, video_name = CreateVideoWriter(camera_name, video_start_time, fourcc, frame_rate, frame_width, frame_height)
        # cv2.VideoWriter(VideoName(camera_name, video_start_time), fourcc, frame_rate, (frame_width, frame_height))

        

        lastYoloRun = datetime.now()

        while cap.isOpened():
            prevFrame = currentFrame
            ret, currentFrame = cap.read()
            # self.VideoWriter.write(currentFrame)
            isMotion = DetectMotion(currentFrame, prevFrame, motion_threshold)
            if isMotion == True:
                pass
                # _delta = datetime.now() - lastYoloRun
                # if _delta.seconds < 1:
                #     # if yolo has been ran in the last 1 second skip it
                #     pass
                # elif _delta.seconds < 5*60:
                #     # if this is the first movement in the last 5 minutes run yolo on 10th priority
                #     yoloqueue.put(YoloFrame(10, datetime.now(), currentFrame))
                # else:
                #     # if this is the first movement in over 5 minutes run yolo on high priority
                #     yoloqueue.put(YoloFrame(1, datetime.now(), currentFrame))
                #     #logger.info('adding frame to yolo queue')

            # Saves for video
            video_writer.write(currentFrame)

            _delta = datetime.now() - video_start_time
            _video_run_time = 5*60
            if _delta.seconds > _video_run_time:
                logger.info(_video_run_time, _delta.seconds)
                logger.info(f'closing file {video_name}')
                video_writer.release()
                video_writer, video_name = CreateVideoWriter(camera_name, video_start_time, fourcc, frame_rate, frame_width, frame_height)
                logger.info(f'opening file {video_name}')
                video_start_time = datetime.now()

            # cv2.imshow('Video', currentFrame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except BaseException as e:
        raise e
    finally:
        cap.release()
        video_writer.release()

print('starting video')
cameras = [('cats', 'demo.mp4')]

# with multiprocessing.Pool(len(cameras)) as pool:
#     pool.map(Camera, cameras)

with open(os.path.join('data', "helloworld.txt"), "a") as f:
    f.write(f"starting at{datetime.now()}!")
    f.close()

ping = str(os.system('ping -c 1 192.168.50.30'))
logger.info(f'ping results: {ping}')


YoloQueue = PriorityQueue()
Camera('testcamera', r'rtsp://admin:@192.168.50.30:554/h264Preview_01_main', YoloQueue, motion_threshold = 900, frame_rate=30)



print(f'queue size is {YoloQueue.qsize()}')