import time
import calendar


# import torch

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

import os
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel=logging.DEBUG

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel=logging.DEBUG

file_handler = logging.FileHandler(filename=os.path.join('logs', 'test.log'))
file_handler.setLevel=logging.ERROR

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info('hello world info')
logger.error('hello world error')

