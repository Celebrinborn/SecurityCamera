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
# Configure logging
configure_logging()
logger = logging.getLogger()

logger.debug('creating flask app')
app = Flask(__name__)
try:
    # Open the Camera and FileManager objects when the Flask app starts up
    camera_name = 'webcam'
    fps = 10
    camera_url = 0

    _root_file_location = Path('data', camera_name, 'video_cache')
    _root_file_location.mkdir(parents=True, exist_ok=True)
    max_folder_size = int(5e+8)# 500 mb #int(5e+9)
    
    logger.debug('creating filemanager')
    file_Manager = FileManager(_root_file_location, max_folder_size)

    logger.debug('creating camera')
    camera = Camera(camera_name=camera_name, camera_url = camera_url, max_fps=fps)
    _resolution = Resolution(640, 480)
    video_file_manager = VideoFileManager(root_video_file_location=_root_file_location, resolution=_resolution, fps= fps, file_manager=file_Manager)
    motion_detector = MotionDetector(camera_name=camera_name)
    camera.Subscribe_queue(video_file_manager.GetQueue())

    # Add camera and file_manager objects to the app context
    app.camera = camera # type: ignore
    app.file_manager = video_file_manager # type: ignore

    logger.info('ending app')
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
    return None# jsonify(status_dict)

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
        while True:
            frame = frame_queue.get()
            assert isinstance(frame, np.ndarray), f'frame is not an ndarray, frame is: {type(frame)}'
            _successful,buffer = cv2.imencode('.jpg',frame)
            if _successful:
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run()