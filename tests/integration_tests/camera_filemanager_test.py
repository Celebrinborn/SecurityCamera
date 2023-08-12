from queue import Queue
import time
from camera.camera import Camera
from camera.filemanager import FileManager, VideoFileManager

from camera.resolution import Resolution


import numpy as np
import cv2

import pytest

# mock camera
class MockVideoCapture:  
    _frame_number = 0
    def __init__(self, url):
        self.url = url

    def isOpened(self):
        return True
    
    def read(self):
        # Generate a different frame each time
        self._frame_number += 1


        # frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = self.draw_number(self._frame_number)
        
        return True, frame

    def release(self):
        pass
    def draw_number(self, number: int, image_size: tuple = (100, 100), font_scale: float = 1, thickness: int = 2) -> np.ndarray:
        """
        Function to draw a black number on a white background using OpenCV.
        
        Parameters:
        number (int): The number to be drawn.
        image_size (tuple): The size of the image to be created, in pixels. Default is (100, 100).
        font_scale (float): Font scale factor that is multiplied by the font-specific base size. Default is 1.
        thickness (int): Thickness of the lines used to draw a text. Default is 2.
        
        Returns:
        image (np.ndarray): The resulting image.
        """
        # Create a white image
        image = np.ones((*image_size, 3), dtype=np.uint8) * 255

        # Define font and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)  # Black

        # Get text size
        text = str(number)
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate center position for text
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2

        # Put text on image
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        return image

@pytest.fixture
def video_file_manager(tmp_path):
    resolution = Resolution(width=640, height=480)
    file_manager = FileManager(root_folder=tmp_path, max_dir_size=1000)
    video_file_manager = VideoFileManager(root_video_file_location=tmp_path, resolution=resolution, fps=30, file_manager=file_manager)
    return video_file_manager




def test_write_camera_video_to_file(tmp_path, video_file_manager):
    # Given
    fps = 15
    resolution = Resolution(100, 100)

    queue = Queue()
    # When

    # mockVideoCapture will simply keep outputting frames
    with Camera('test_camera', 0, 15, cv2_module=MockVideoCapture) as camera:
        with VideoFileManager(tmp_path, resolution, fps, video_file_manager) as video_file_manager:
            camera.Subscribe_queue(video_file_manager.GetQueue())
    # Then
    time.sleep(10)