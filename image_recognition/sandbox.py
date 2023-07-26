from detect import ObjectDetector
from camera.camera import Camera
from queue import Queue
import cv2
import os

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

cv2.imshow('image',frame)
cv2.waitKey(1)
cv2.imwrite(os.path.join('data', 'frame.jpg'), frame)

print(frame.shape)
print(len(frame))
print(type(frame))

print('loading detector')
# Create ObjectDetector instance
detector = ObjectDetector(weights=os.path.join('image_recognition','yolov7.pt'), conf_threshold=0.25, iou_threshold=0.45, img_size=640)

print('running detector')
# Call detect_objects on an image
objects = detector.detect_objects(frame)

# Print results
for obj in objects:
    print(obj)
