from flask import render_template, Response
from . import views
from camera.camera import Camera
import cv2
from queue import Queue
import numpy as np

@views.route('/camera')
def camera():
    return render_template('camera.html')

@views.route('/video_feed')
def video_feed():
    def generate_frames():
        print('creating camera')
        with Camera('main', 0, 25) as camera:        
            print('creating queue')
            frame_queue = Queue()
            print('subscribing queue')
            camera.Subscribe_queue(frame_queue)
            print('starting camera')
            camera.Start()
            while True:
                frame = frame_queue.get()
                assert isinstance(frame, np.ndarray), f'frame is not an ndarray, frame is: {type(frame)}'
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
