import torch
from pathlib import Path
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh#, plot_one_box


class ObjectDetector:
    def __init__(self, weights, conf_threshold, iou_threshold, img_size):
        print('loading weights')
        self.model = attempt_load(weights, map_location=torch.device('cpu'))
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

    def detect_objects(self, image):
        # Convert image to torch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Resize image to img_size
        image = torch.nn.functional.interpolate(image, size=self.img_size, mode='bilinear', align_corners=False)

        # Detect objects using model
        with torch.no_grad():
            outputs = self.model(image)[0]
            detections = non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        print(len(detections))
        # Convert detections to list of dictionaries
        results = []
        for detection in detections[0]:
            if detection is not None:
                box = scale_coords(image.shape[2:], detection[:4], (image.shape[2], image.shape[3])).round()
                x1, y1, x2, y2 = [int(i) for i in box]
                w, h = x2 - x1, y2 - y1
                label = self.model.module.names[int(detection[5])]
                confidence = float(detection[4])
                result = {'label': label, 'confidence': confidence, 'x': x1, 'y': y1, 'width': w, 'height': h}
                results.append(result)

        return results



if __name__ == '__main__':
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
