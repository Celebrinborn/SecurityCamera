from flask import render_template, Response, current_app
from . import views
from camera.camera import Camera
import cv2
from queue import Queue
import numpy as np

import logging
logger = logging.getLogger()

@views.route('/camera')
def camera_route():
    return render_template('camera.html')

@views.route('/video_feed')
def video_feed():
    def generate_frames():
        logger.info('subscribing camera')
        camera_instance: Camera = current_app.config.get('camera')
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
