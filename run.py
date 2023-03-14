from app import create_app
import threading
import logging
from log_config import configure_logging

import os

from camera.camera import Camera
from camera.filemanager import FileManager

app = create_app()
configure_logging()
logger = logging.getLogger()

if __name__ == '__main__':
    logger.info('creating camera')
    with Camera('main', 0, 25) as camera:
        logger.info('creating filemanager')
        with FileManager(camera.GetFrameWidth, camera.GetFrameHeight, 25, os.path.join('data', 'recordings')) as fileManager:
            camera.Subscribe_queue(fileManager.GetQueue())
    server = threading.Thread(target=app.run, kwargs={'debug': True})
    server.start()
