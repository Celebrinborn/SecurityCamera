from app import create_app
import logging
from log_config import configure_logging
import os

from camera.camera import Camera
from camera.filemanager import FileManager

# Configure logging
configure_logging()
logger = logging.getLogger()

# Create Flask app
app = create_app()

if __name__ == '__main__':
    fps = 25

    # Create camera object
    logger.info('Creating camera...')
    with Camera(camera_name='webcam', camera_url=0, max_fps=fps) as camera:
        # Create file manager object
        logger.info('Creating file manager...')
        with FileManager(frame_width=camera.GetFrameWidth(), frame_height=camera.GetFrameHeight(), fps=fps, root_file_location=os.path.join('data', 'temp_filemanager_output')) as fileManager:
            # Subscribe the file manager's queue to the camera
            camera.Subscribe_queue(fileManager.GetQueue())

            # Start capturing frames from the camera and writing them to file
            camera.Start()
            fileManager.Start()

            # Start the Flask app
            logger.info('Starting Flask app...')
            app.run(debug=True)
