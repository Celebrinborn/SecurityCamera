from app import create_app
import logging
from log_config import configure_logging
import os

from camera.camera import Camera
from camera.filemanager import FileManager

from flask import Blueprint
# import the view functions
from app.views.index import index
from app.views.camera import *
from app.views.stats import stats

# Configure logging
configure_logging()
logger = logging.getLogger()

# Create Flask app
app = create_app()

if __name__ == '__main__':
    fps = 25

    # Create camera object
    with Camera(camera_name='webcam', camera_url=0, max_fps=fps) as camera_route:
        # Create file manager object
        logger.info('Creating file manager...')
        with FileManager(frame_width=camera_route.GetFrameWidth(), frame_height=camera_route.GetFrameHeight(), fps=fps, root_file_location=os.path.join('data', 'temp_filemanager_output')) as fileManager:
            # Subscribe the file manager's queue to the camera
            camera_route.Subscribe_queue(fileManager.GetQueue())

            # Start capturing frames from the camera and writing them to file
            camera_route.Start()
            fileManager.Start()

            app.config['camera'] = camera_route

            logger.info('Starting Flask app')
            app.run(debug=True, use_reloader=False)