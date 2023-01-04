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
logging.basicConfig(filename=os.path.join('logs', 'test.log'), level=logging.DEBUG)
logger = logging.getLogger(__name__)


logger.info('hello world')