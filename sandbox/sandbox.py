from camera.camera import Camera
from camera.MotionDetector import MotionDetector
import time
from queue import Queue
import cv2

print('creating camera')
with Camera('webcam', 0, 15) as camera:
    print('creating motion detector')
    with MotionDetector() as motion_detector:
        print('subscribing')
        camera.Subscribe_queue(motion_detector.GetQueue())
        print('starting timer')
        time.sleep(15)