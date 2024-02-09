from collections import namedtuple
import logging
import os
from queue import Queue
from typing import Union
import cv2
from flask import Flask, Response, jsonify, render_template, current_app, request
import numpy as np
from camera.camera import Camera
from camera.filemanager import VideoFileManager, Resolution, FileManager
from camera.MotionDetector import MotionDetector
from pathlib import Path
import time

import logging
from log_config import configure_logging

from camera.MotionDetector import motion

configure_logging(log_file_name=f'{os.environ.get("CAMERA_NAME", "unknown_camera")}.jsonl')
logger = logging.getLogger()

# list all environment variable names
# logger.debug(f'environment variables: {os.environ.keys()}')

# check for env file
if Path('secrets','.env').is_file():
    logger.info('found .env file')
    import dotenv
    dotenv.load_dotenv(Path('secrets','.env'))

    logger.debug(f'environment variables: {os.environ.keys()}')

if 'SA_PASSWORD' not in os.environ:
    logger.warning('SA_PASSWORD environment variable is not set')
    # check if /run/secrets/SA_PASSWORD exists
    if Path('/run/secrets/SA_PASSWORD').is_file():
        with open('/run/secrets/SA_PASSWORD', 'r') as f:
            os.environ['SA_PASSWORD'] = f.read()
            logger.info(f'SA_PASSWORD environment variable is set from /run/secrets/SA_PASSWORD with len {len(os.environ["SA_PASSWORD"])}')
    else:
        logger.error('SA_PASSWORD environment variable is not set, and /run/secrets/SA_PASSWORD does not exist')
        logger.debug(f'does /run/ exist: {Path("/run").is_dir()}')
        logger.debug(f'does /run/secrets exist: {Path("/run/secrets").is_dir()}')
        logger.debug(f'contents of /run/secrets: {os.listdir("/run/secrets")}')
        raise Exception('SA_PASSWORD environment variable is not set, and /run/secrets/SA_PASSWORD does not exist')
else:
    logger.info('SA_PASSWORD environment variable is set directly')
# verify SA_PASSWORD is not null or empty
if os.environ['SA_PASSWORD'] is None or os.environ['SA_PASSWORD'] == '':
    logger.error('SA_PASSWORD environment variable is null or empty')
    raise Exception('SA_PASSWORD environment variable is null or empty')

logger.debug('creating flask app')
app = Flask(__name__)
try:
    # Open the Camera and FileManager objects when the Flask app starts up
    camera_name = os.environ.get('CAMERA_NAME', 'webcam')
    logger.info(f'camera_name: {camera_name}')
    fps = 10
    camera_url = os.environ.get('CAMERA_URL', 0)
    logger.info(f'camera_url: {camera_url}')

    logger.info(f'app running from {os.getcwd()}')

    _root_file_location = Path('data', camera_name, 'video_cache')
    _root_file_location.mkdir(parents=True, exist_ok=True)
    max_folder_size_str : Union[str, int] = os.environ.get('max_folder_size_bytes', int(5e+8)) # 500 mb #int(5e+9)
    try:
        max_folder_size = int(max_folder_size_str)
        logger.info(f'max_folder_size_bytes: {max_folder_size=}')
    except ValueError:
        logger.warning(f'max_folder_size_bytes environment variable is not an int or float, it is: {max_folder_size_str=}')
        max_folder_size = int(5e+8)
    
    logger.debug('creating filemanager')
    file_Manager = FileManager(_root_file_location, max_folder_size)

    logger.debug('creating camera')
    camera = Camera(camera_url = camera_url, max_fps=fps)
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


camera_name = os.environ.get('CAMERA_NAME', 'webcam')

# Route for the index page
@app.route('/')
def index():
    # return 'hello world'
    return render_template('index.html', camera_name = camera_name)

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
    return render_template('camera.html', camera_name = camera_name)

@app.route('/motion_config')
def motion_config():
    logger.debug(r'entered /motion_config')
    current_threshold = app.motion_detector.contour_threshold
    return render_template('motion_config.html', camera_name=camera_name, current_threshold=current_threshold)

@app.route('/update_motion_threshold', methods=['POST'])
def update_threshold():
    try:
        # Extract the new threshold value from the request
        new_threshold = request.json['threshold']
        # Update the motion_detector's contour_threshold
        app.motion_detector.contour_threshold = float(new_threshold)
        return jsonify({'status': 'success', 'new_threshold': new_threshold})
    except Exception as e:
        logger.error(f"Error updating threshold: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/motion_feed')
def motion_feed():
    logger.debug(r'entered /motion_feed')
    def generate_motion_frames():
        logger.info('subscribing motion')
        motion_detector: MotionDetector = app.motion_detector
        if not isinstance(motion_detector, MotionDetector): raise TypeError('motion_detector is not a MotionDetector object')
        motion_queue = Queue()
        motion_detector.Subscribe_queue(motion_queue)
        try:
            while True:
                np_frame = motion_queue.get()
                assert isinstance(np_frame, np.ndarray), f'frame is not an ndarray, frame is: {type(np_frame)}'
                _successful,buffer = cv2.imencode('.jpg',np_frame)
                np_frame=buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + np_frame + b'\r\n')
        except GeneratorExit:
            # This block will theoretically be executed when the client disconnects
            motion_detector.Unsubscribe_queue(motion_queue)
            logger.info('Client disconnected, unsubscribed motion.')
    return Response(generate_motion_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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

                frame_copy = frame.copy() # prevent overwriting the original frame

                cv2.putText(frame_copy, screen_text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness + 2)  # Black Background
                cv2.putText(frame_copy, screen_text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)  # White Text

                _successful,buffer = cv2.imencode('.jpg',frame_copy)
                if _successful:
                    frame=buffer.tobytes()
                    yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            # This block will theoretically be executed when the client disconnects
            camera_instance.Unsubscribe_queue(frame_queue)
            logger.info('Client disconnected, unsubscribed camera.')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# app.run(use_reloader=False)