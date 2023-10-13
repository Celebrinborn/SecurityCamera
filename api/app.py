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



app = Flask(__name__)


# Route for the index page
@app.route('/')
def index():
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
try:
    # Open the Camera and FileManager objects when the Flask app starts up
    camera_name = 'webcam'
    fps = 10
    camera_url = 0

    _root_file_location = Path('data', camera_name, 'video_cache')
    _root_file_location.mkdir(parents=True, exist_ok=True)
    _5gb = int(5e+9)
    fileManager = FileManager(_root_file_location, _5gb)

    with Camera(camera_name=camera_name, camera_url = camera_url, max_fps=fps) as camera:
        _resolution = Resolution(640, 480)
        with VideoFileManager(root_video_file_location=_root_file_location, resolution=_resolution, fps= fps, file_manager=fileManager) as file_manager:
            with MotionDetector(camera_name=camera_name) as motion_detector:
                camera.Subscribe_queue(file_manager.GetQueue())
                
                
                # Add camera and file_manager objects to the app context
                app.camera = camera # type: ignore
                app.file_manager = file_manager # type: ignore

                
                # Start the Flask app to keep the objects open
                app.run()
    logger.info('ending app')
except Exception as e:
    # Log the exception and raise it to crash the app
    logging.exception('Failed to open Camera or FileManager')
    raise e