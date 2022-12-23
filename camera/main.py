import os
import sys
import cv2
import time
import subprocess
from queue import Queue, PriorityQueue
import multiprocessing
from dataclasses import dataclass
import typing
from datetime import datetime, timedelta

def FormatTime(time:datetime):
    return time.strftime(r"%Y%m%d_%H%M%S")

if 'camera_url' in os.environ:
    camera_url = os.environ['camera_url']
else:
    camera_url = r'rtsp://admin:@192.168.50.30:554/h264Preview_01_main'
if 'camera_name' in os.environ:
    camera_name = os.environ['camera_name']
else:
    camera_name = 'testcamera'


import logging
import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_format =  '%(asctime)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(funcName)s - %(message)s'
formatter = logging.Formatter(log_format)

handler_stdout = logging.StreamHandler(sys.stdout)
handler_stdout.setLevel(logging.DEBUG)
handler_stdout.setFormatter(formatter)
logger.addHandler(handler_stdout)

_log_filename = os.path.join('logs', f'{camera_name}_{FormatTime(datetime.now())}_camera_controller.log')
handler_files = logging.handlers.RotatingFileHandler(
    filename=_log_filename,
    maxBytes=52428800, backupCount=4)
handler_files.setLevel(logging.WARNING)
handler_files.setFormatter(formatter)
logger.addHandler(handler_files)

# logging.basicConfig(filename=os.path.join('logs', f'{camera_name}_camera_controller.log'),level=logging.DEBUG, filemode='w', format=log_format)

logger.info(f'camera_url = {camera_url}')

@dataclass(order=True)
class YoloFrame:
    priority: int
    timestamp: datetime
    frame: typing.Any

def DetectMotion(currentFrame, prevFrame, threshold:int) -> bool:
    if currentFrame is None:
        logger.warning('DetectMotion was passed a None currentFrame')
        return False, currentFrame
    if prevFrame is None:
        logger.warning('DetectMotion was passed a None prevFrame')
        return False, prevFrame

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
    return isMotion, prevFrame

def VideoName(camera_name:str, video_file_extention:str, video_start_time:datetime) -> str:
    return os.path.join('data', f'{camera_name}_{FormatTime(video_start_time)}.{video_file_extention}')

def CreateVideoWriter(camera_name:str, frame_rate:int, cap:cv2.VideoCapture, video_start_time:datetime = datetime.now()) -> cv2.VideoWriter:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_file_extention = '.avi' # mp4v for windows
    logger.info(f'creating video writer {camera_name} with fourcc {fourcc}, frame rate {frame_rate}, and dimentions {frame_width, frame_height}')
    video_name = VideoName(camera_name, video_file_extention, video_start_time)
    return cv2.VideoWriter(video_name, fourcc, frame_rate, (frame_width, frame_height)), video_name

# connect to camera

# set up recording

# main loop

# check frame for motion
# if motion then do stuff
def Main(Camera_name:str, Camera_path:str, YoloQueue:PriorityQueue, Motion_Threshold:int = 900, Frame_rate = 30):
    # connect to camera
    logger.info(f'loading camera {Camera_name} at {Camera_path}')
    logger.info(f'cv2 build info: {str(cv2.getBuildInformation())}')
    try:
        cap = cv2.VideoCapture(Camera_path)
        # read first two frames
        try:
            ret, prevFrame = cap.read()
            ret, currentFrame = cap.read()
        except BaseException as e:
            logger.critical('unable to read initial two frames for unknown reason. the application will now close', exc_info=True)
            logger.critical(e)
            raise e

        # get video writer
        try:
            video_writer, video_name = CreateVideoWriter(camera_name = Camera_name, frame_rate=Frame_rate, cap = cap)
        except BaseException as e:
            logger.critical('unable to get video writer', exc_info=True)
            logger.critical(e)
            raise e
        
        # set the initial start time and when to finish the video
        video_start_time = datetime.now()
        if 'video_max_length' in os.environ:
            try: 
                video_max_length = int(os.environ.get('video_max_length'))
            except ValueError as e:
                logger.error(f'environ video_max_length is NOT an int, value is {os.environ.get("video_max_length")}')
        else:
            video_max_length = 5*60
        video_end_time = video_start_time + timedelta(seconds=video_max_length)
        while cap.isOpened():
            try:
                prevFrame = currentFrame
                try:
                    flag, currentFrame = cap.read()
                except BaseException as e:
                    logging.error(f'unable to read current frame, {str(e)}', exc_info=True)
                    raise e

                if currentFrame is None:
                    logging.warning(f'currentFrame is none at ')


                # check for motion
                try:
                    _isMotion, outlinedFrame = DetectMotion(currentFrame, prevFrame, Motion_Threshold)
                except BaseException as e:
                    logging.error('DetectMotion is crashing', exc_info=True)
                if _isMotion:
                    _image_file_name = os.path.join('data', 'images', f'{FormatTime(datetime.now())}.jpg')
                    try:
                        cv2.imwrite(_image_file_name, outlinedFrame)
                    except BaseException as e:
                        logger.error(f'unable to write outlined frame to image file, {str(e)}', exc_info=True)
                # save frame to video
                try:
                    video_writer.write(currentFrame)
                except BaseException as e:
                    logger.error('unable to save frame for unknown reason', exc_info=True)

                # check if max video length is reached and if so start a new video
                if (datetime.now()) > video_end_time:
                    video_start_time = datetime.now()
                    video_end_time = video_start_time + timedelta(seconds=video_max_length)
                    logger.info('video max lenght reached. starting new video')
                    try:
                        video_writer.release()
                        video_writer, video_name = video_writer, video_name = CreateVideoWriter(camera_name = Camera_name, frame_rate=Frame_rate, cap = cap)
                    except BaseException as e:
                        logger.critical('unable to release video_writer and recreate for unknown reason. waiting 15 seconds then trying again', exc_info=True)
                        try:
                            time.sleep(15)
                            video_writer.release()
                            video_writer, video_name = video_writer, video_name = CreateVideoWriter(camera_name = Camera_name, frame_rate=Frame_rate, cap = cap)
                        except BaseException as e:
                            logger.critical('FINAL ATTEMPT: unable to release video_writer and recreate for unknown reason', exc_info=True)
                            raise e
                    logger.info(f'new video filename = {video_name}')

            except BaseException as e:
                logger.critical('an unknown frame reading error has occured. continuing...')
                logger.critical(e, exc_info=True)

    except BaseException as e:
        logger.fatal('unhandled exception has occured. this may be an issue with the capture failing', exc_info=True)
        logger.fatal(e, exc_info=True)
    finally:
        cap.release()
        try:
            video_writer.release()
        except BaseException:
            logger.critical('unable to release videowriter', exc_info=True)

# with multiprocessing.Pool(len(cameras)) as pool:
#     pool.map(Camera, cameras)

ping = str(os.system('ping -c 1 192.168.50.30'))
logger.info(f'ping results: {str(ping)}')


YoloQueue = PriorityQueue()

if 'motion_threshold' in os.environ:
    motion_threshold = os.environ['motion_threshold']
else:
    motion_threshold = 900

Main(Camera_name = camera_name, Camera_path = camera_url, YoloQueue = YoloQueue, Motion_Threshold = motion_threshold, Frame_rate=30)

print(f'queue size is {YoloQueue.qsize()}')