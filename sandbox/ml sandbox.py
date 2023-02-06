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

# try:
#     os.environ['hello world this will crash']
#     print('hi')
#     err = 5/0
# except KeyError:
#     logger.error('caught my key error')
# except BaseException as e:
#     print('caught my other error')
#     logger.error('caught my other error', stack_info=True, exc_info=True)

print(int(time.time()))
l = ['1672168064781.jpg', '1672168064780.jpg', '1672168064779.jpg', '1672168064778.jpg', '1672168064777.jpg', '1672168064776.jpg', '1672168064775.jpg', '1672168064774.jpg', '1672168064773.jpg', '1672168064772.jpg', '1672168064771.jpg', '1672168064770.jpg']

path = os.path.join('16722','thumbs')
lf = [os.path.join(path, x) for x in l]

print(lf)