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
