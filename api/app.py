import logging
import os
from queue import Queue
import cv2
from flask import Flask, Response, render_template, current_app
import numpy as np
from camera.camera import Camera
from camera.filemanager import VideoFileManagerOld
from camera.MotionDetector import MotionDetector

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

    _root_file_location = os.path.abspath(os.path.join('data', camera_name, 'video_cache'))
    try:
        if not os.path.exists(_root_file_location):
            os.makedirs(_root_file_location)
            logger.info(f"Created directory: {_root_file_location}")
        else:
            logger.info(f"Directory already exists: {_root_file_location}")
    except OSError as e:
        logger.exception(f"Error creating directory: {e}")

    with Camera(camera_name=camera_name, camera_url = camera_url, max_fps=fps) as camera:
        with VideoFileManagerOld(frame_width= 640, frame_height= 480, fps= fps,
                        root_file_location=_root_file_location, 
                        camera_name='webcam') as file_manager:
            with MotionDetector(camera_name=camera_name, threshold=1000, mask=None, detector_post_cooldown_seconds=1.0) as motion_detector:
                camera.Subscribe_queue(file_manager.GetQueue())

                _queue = motion_detector.GetQueue()
                logger.debug(f'type of queue is {_queue}')
                camera.Subscribe_queue(_queue)
                camera.Start()
                file_manager.Start()
                motion_detector.Start()



                # Add camera and file_manager objects to the app context
                app.camera = camera
                app.file_manager = file_manager

                
                # Start the Flask app to keep the objects open
                app.run()

except Exception as e:
    # Log the exception and raise it to crash the app
    logging.exception('Failed to open Camera or FileManager')
    raise e