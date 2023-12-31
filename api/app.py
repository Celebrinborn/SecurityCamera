from collections import namedtuple
import logging
import os
from queue import Queue
import cv2
from flask import Flask, Response, render_template, current_app
import numpy as np
from camera.camera import Camera
from camera.filemanager import VideoFileManager, Resolution, FileManager
from camera.MotionDetector import MotionDetector
from pathlib import Path
import time

import logging
from log_config import configure_logging

from camera.MotionDetector import motion


# Configure logging
if 'PRODUCTION' in os.environ and os.environ['PRODUCTION'] == 'True':
    configure_logging()
    logger = logging.getLogger()
else:
    configure_logging(clear_log_file=True)
    logger = logging.getLogger()
    logger.warning('CLEARED LOG FILE AS production ENVIRONMENT VARIABLE IS NOT SET')


# For local debugging load the environment variables from the .env file in the secrets folder if it exists
from dotenv import load_dotenv
if Path('secrets','.env').is_file():
    load_dotenv(Path('secrets','.env'))
# verify SA_PASSWORD is set
if 'SA_PASSWORD' not in os.environ:
    raise ValueError('SA_PASSWORD environment variable is not set')
else:
    logger.debug('SA_PASSWORD environment variable is set')

logger.debug('creating flask app')
app = Flask(__name__)
try:
    # Open the Camera and FileManager objects when the Flask app starts up
    camera_name = os.environ.get('CAMERA_NAME', 'webcam')
    logger.info(f'camera_name: {camera_name}')
    fps = 10
    camera_url = os.environ.get('CAMERA_URL', 0)
    logger.info(f'camera_url: {camera_url}')

    _root_file_location = Path('data', camera_name, 'video_cache')
    _root_file_location.mkdir(parents=True, exist_ok=True)
    max_folder_size = int(5e+8)# 500 mb #int(5e+9)
    
    logger.debug('creating filemanager')
    file_Manager = FileManager(_root_file_location, max_folder_size)

    logger.debug('creating camera')
    camera = Camera(camera_name=camera_name, camera_url = camera_url, max_fps=fps)
    # set the camera resolution to 240p
    _resolution = Resolution(320,240)
    video_file_manager = VideoFileManager(root_video_file_location=_root_file_location, resolution=_resolution, fps= fps, file_manager=file_Manager)
    motion_detector = MotionDetector(camera_name=camera_name)
    camera.Subscribe_queue(video_file_manager.GetQueue())
    camera.Subscribe_queue(motion_detector.GetQueue())

    # Add camera and file_manager objects to the app context
    app.camera = camera # type: ignore
    app.motion_detector = motion_detector # type: ignore
    app.file_manager = video_file_manager # type: ignore

except Exception as e:
    # Log the exception and raise it to crash the app
    logging.exception('Failed to open Camera or FileManager')
    raise e


# Route for the index page
@app.route('/')
def index():
    # return 'hello world'
    return render_template('index.html')

# Route for the status API
@app.route('/status')
def status():
    # status_dict = {
    #     'camera_status': app.cm.status(),
    #     'file_status': app.fm.status()
    # }
    return 'this page is not yet implemented'

@app.route('/camera')
def camera_route():
    logger.debug(r'entered /camera')
    return render_template('camera.html')



@app.route('/video_feed')
def video_feed():
    logger.debug(r'entered /video_feed')
    def generate_frames():
        logger.info('subscribing camera')
        camera_instance: Camera = app.camera
        if not isinstance(camera_instance, Camera): raise TypeError('camera_instance is not a Camera object')
        frame_queue = Queue()
        camera_instance.Subscribe_queue(frame_queue)
        try:
            i = -1
            screen_text = 'na'
            motion_detector: MotionDetector = app.motion_detector
            while True:
                i += 1
                frame = frame_queue.get()
                assert isinstance(frame, np.ndarray), f'frame is not an ndarray, frame is: {type(frame)}'

                # draw motion on frame
                # black background

                screen_text = f'motion amount: {motion_detector.current_motion_amount}. {"MOTION DETECTED" if motion_detector.current_motion_amount > motion_detector.motion_threshold else ""}'
                # Calculate text size and position dynamically
                text_scale = frame.shape[1] / 1600  # Adjust the denominator to fit your needs
                text_thickness = max(1, int(text_scale * 3))
                (text_width, text_height), _ = cv2.getTextSize(screen_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                start_x = 10
                start_y = frame.shape[0] - 10  # 10 pixels from the bottom

                cv2.putText(frame, screen_text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness + 2)  # Black Background
                cv2.putText(frame, screen_text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)  # White Text

                _successful,buffer = cv2.imencode('.jpg',frame)
                if _successful:
                    frame=buffer.tobytes()
                    yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            # This block will theoretically be executed when the client disconnects
            camera_instance.Unsubscribe_queue(frame_queue)
            logger.info('Client disconnected, unsubscribed camera.')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(use_reloader=False)