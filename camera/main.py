import os
import sys
import cv2
import time
import numpy as np
import subprocess
from queue import Queue, PriorityQueue
import multiprocessing
from dataclasses import dataclass
import typing
from datetime import datetime, timedelta

# from mask_generator import DrawMask
def DrawMask(image:np.ndarray, mask:np.array):
    img = image.copy()
    return cv2.fillPoly(img, pts=[mask], color=(255,255,255))

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
logger.setLevel(logging.INFO)

log_format =  '%(asctime)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(funcName)s - %(message)s'
formatter = logging.Formatter(log_format)

handler_stdout = logging.StreamHandler(sys.stdout)
handler_stdout.setLevel(logging.INFO)
handler_stdout.setFormatter(formatter)
logger.addHandler(handler_stdout)

#_{FormatTime(datetime.now())}
_log_filename = os.path.join('logs', f'{camera_name}_camera_controller.log')
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

def DetectMotion(Frame_width, Frame_height, CurrentFrame, PrevFrame, Threshold:int, Mask=None) -> bool:
    if CurrentFrame is None:
        logger.warning('DetectMotion was passed a None currentFrame')
        return False, CurrentFrame
    if PrevFrame is None:
        logger.warning('DetectMotion was passed a None prevFrame')
        return False, PrevFrame


    # height scale ratioq
    height_ratio = 100 / Frame_height
    width_ratio = 100 / Frame_width

    area_ratio = (Frame_height * Frame_width)

    background = cv2.cvtColor(PrevFrame,cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background,(21,21), 0)

    gray = cv2.cvtColor(CurrentFrame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21), 0)
    
    diff = cv2.absdiff(background,gray)

    thresh = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations = 2)

    cnts,res = cv2.findContours(thresh.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    isMotion = False
    bounding_image = CurrentFrame.copy()
    for contour in cnts:
        if cv2.contourArea(contour) < Threshold :
            continue
        #(x,y,w,h) = cv2.boundingRect(contour)
        #cv2.rectangle(bounding_image,(x,y),(x+w,y+h),(0,255,0), 3)
        isMotion = True
    return isMotion

def VideoName(camera_name:str, video_file_extention:str, video_start_time:datetime) -> str:
    # create directory if it does not exist
    _path = os.path.join('data', camera_name)
    
    if not os.path.exists(_path):
        logger.info(f'path {_path} does not exist. creating...')
        logger.info(f'at: {os.getcwd()}')
        os.mkdir(_path)

    return os.path.join('data', camera_name, f'{camera_name}_{FormatTime(video_start_time)}.{video_file_extention}')

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
    logger.debug(f'cv2 build info: {str(cv2.getBuildInformation())}')
    
    # check if mask file is present
    _files_in_directory = os.listdir(os.path.join('data'))
    logger.debug(f'files in data directory: {_files_in_directory}')
    _file_in_directory = f'{camera_name}_mask.npy'
    if _file_in_directory in _files_in_directory:
        _file_in_directory_full_path = os.path.join('data', _file_in_directory)
        logger.info(f'loading mask file {_file_in_directory_full_path}')
        try:
            mask = np.load(_file_in_directory_full_path)
        except ValueError:
            mask = None
            logging.warning('unable to load mask')
    else:
        mask = None

    try:
        cap = cv2.VideoCapture(Camera_path)
        # read first two frames
        try:
            ret, prevFrame = cap.read()
            ret, currentFrame = cap.read()

            # generate first two masked images
            if mask is not None:
                masked_prevFrame = DrawMask(prevFrame, mask)
                masked_currentFrame = DrawMask(currentFrame, mask)
            #save preview of mask if it doesn't already exist
            _mask_preview_filename = os.path.join('data', f'{camera_name}_mask_preview.jpg')
            if not os.path.exists(_mask_preview_filename):
                masked_currentFrame = cv2.imwrite(_mask_preview_filename, currentFrame.copy())
        except BaseException as e:
            logger.critical('unable to read initial two frames for unknown reason. the application will now close', exc_info=True)
            logger.critical(e)
            raise e

        # generate preview image for generating masks
        _view_preview_filename = f'{camera_name}_preview.jpg'
        logger.info(f'saving preview file {_view_preview_filename}')
        cv2.imwrite(os.path.join('data', _view_preview_filename), currentFrame)

        # get dimentions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        read_frame_attempts = 0
        while cap.isOpened():
            try:
                prevFrame = currentFrame
                try:
                    flag, currentFrame = cap.read()
                except BaseException as e:
                    logging.error(f'unable to read current frame, {str(e)}', exc_info=True)
                    raise e
                if flag == False:
                    logging.info('no frame available')
                    read_frame_attempts = read_frame_attempts + 1
                    logging.info(f'waiting {2**read_frame_attempts} seconds before reestablishing connection')
                    time.sleep(2**read_frame_attempts) # exponential backoff 2 ^ read_frame_attempts
                    if read_frame_attempts > 4:
                        cap = cv2.VideoCapture(Camera_path)
                        flag, currentFrame = cap.read()
                    continue

                if currentFrame is None:
                    logging.warning(f'currentFrame is none at ')

                # apply mask
                masked_prevFrame = masked_currentFrame
                if mask is not None:
                    masked_currentFrame = DrawMask(currentFrame, mask)

                # check for motion
                try:
                    # Frame_width, Frame_height, CurrentFrame, PrevFrame, Threshold:int, Mask=None) -> bool:
                    _isMotion = DetectMotion(
                        Frame_width=frame_width, Frame_height=frame_height, CurrentFrame=currentFrame, PrevFrame=prevFrame, Mask=None, Threshold=100)
                except BaseException as e:
                    logging.error('DetectMotion is crashing', exc_info=True)
                if _isMotion:
                    _image_file_name = os.path.join('data', 'images', f'{FormatTime(datetime.now())}.jpg')
                    try:
                        cv2.imwrite(_image_file_name, currentFrame)
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



YoloQueue = PriorityQueue()

if 'motion_threshold' in os.environ:
    motion_threshold = os.environ['motion_threshold']
else:
    motion_threshold = 900

Main(Camera_name = camera_name, Camera_path = camera_url, YoloQueue = YoloQueue, Motion_Threshold = motion_threshold, Frame_rate=30)

print(f'queue size is {YoloQueue.qsize()}')