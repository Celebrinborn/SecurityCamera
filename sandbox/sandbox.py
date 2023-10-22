from camera.camera import Camera
from camera.filemanager import VideoFileManager
from camera.resolution import Resolution
from camera.MotionDetector import MotionDetector
import time
from queue import Queue
import cv2
from pathlib import Path
from log_config import configure_logging
import logging
# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

_fps = 15
logger.info('starting camera')
camera = Camera('webcam', 0, _fps)

logger.info('creating filemanager')
video_filemanager = VideoFileManager(Path('data', 'webcam'), camera.GetCameraResolution(), _fps)

logger.info('subscribing')
camera.Subscribe_queue(video_filemanager.GetQueue())


logger.info('waiting')
time.sleep(60*2) # wait 6 minutes for camera to finish saving video


# pausing writer
logger.info('stopping writer')
video_filemanager.Stop(blocking=True)


logger.info('ending app')