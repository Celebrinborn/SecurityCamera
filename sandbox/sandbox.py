import os
print("Current working directory: ", os.getcwd())
print("Files and directories in current directory: ", os.listdir())

import logging
from log_config import configure_logging

configure_logging('hello world')

logger = logging.getLogger(__name__)

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')




print(os.path.exists(os.path.join('E:','security_camera','data')))
print(os.path.join('E:','security_camera','data'))

print(os.path.exists(os.path.abspath(r"E:/security_camera/data")))
print(os.path.abspath(r"E:/security_camera/data"))

root_file_location = os.path.join('E:','security_camera','data')

if not os.path.exists(root_file_location):
    drive, path = os.path.splitdrive(root_file_location)
    if not path.startswith(os.path.sep):
        print(f"root_file_location '{root_file_location}' is invalid - missing path separator after drive letter")
    else:        
        print(f"root_file_location '{root_file_location}' does not exist")
else:
    print('valid path')