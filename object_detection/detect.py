import os
import torch
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjectDetector:    
    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def __init__(self, weights_path=Path('data', 'yolov5s.pt'), data_path='data/coco128.yaml', imgsz=(640,640), device=None):
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        logger.info(f'yolo running with device {str(device)}')

        self.weights = os.path.join(weights_path)
        self.dnn = False  # use OpenCV for ONNX interface
        self.data = os.path.join(data_path)  # dataset path
        self.half = False  # use fp16 half precision inference
        self.imgsz = imgsz  # inference size (height, width)
        
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # ensure imgsz is a multiple of the stride
        
        bs = 1  # batch size
        _batch_size, _channel_count, _height, _width = bs, 3, *self.imgsz
        _warmup_args = (_batch_size, _channel_count, _height, _width)

        logger.info('starting warmup')
        self.model.warmup(_warmup_args)
        logger.info('completed warmup. ready to begin detections')

    def detect(self, image_ndarray:np.ndarray, image_id:str):
        if not isinstance(image_ndarray, np.ndarray):
            raise TypeError("The image_ndarray parameter must be a NumPy array.")

        if image_ndarray.shape[2] != 3:
            raise ValueError("The image_ndarray parameter should have 3 channels (RGB).")
        
        im = self._letterbox(image_ndarray, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model.forward(im=im)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)  # NMS

        pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], image_ndarray.shape).round()
        results = []
        for i, (*xyxy, _conf, _cls) in enumerate(pred[0]):
            bounding_box = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            confidence = float(_conf)
            category = self.model.names[int(_cls)]
            result = {
                'id': i,
                'image_id': image_id,
                'category_id': int(_cls),
                'category_name': category,
                'bbox': bounding_box,
                'area': bounding_box[2] * bounding_box[3],  # Calculating area based on width and height
                'confidence': confidence
            }
            results.append(result)
        return {'annotations':results}



if __name__ == '__main__': #not actually used, just an example of use
    detector = ObjectDetector()

    file_path = 'test_image.jpg'
    image = cv2.imread(file_path)

    import time
    # record the time before running the function
    start_time = time.perf_counter()
    print(start_time)

    # run the detection
    results = detector.detect(image, 'testimage')

    # calculate and print the time it took to run the function
    execution_time = time.perf_counter() - start_time
    print(f'The detection took {execution_time} seconds.')
    for annotation in results['annotations']:
        category = annotation['category_name']
        confidence = annotation['confidence']
        bounding_box = annotation['bbox']
        print(f'{category} detected with confidence {confidence} at {bounding_box}')
