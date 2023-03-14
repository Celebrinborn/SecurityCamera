import cv2
import logging
import threading
from queue import Queue

logger = logging.getLogger(__name__)

class MotionDetector:
    _killDaemon: bool  # flag to abort capture daemon
    def __init__(self, frame, threshold=5000, mask=None):
        """
        Constructor for MotionDetector class.

        Args:
            frame: ndarray representing the initial frame to use for motion detection.
            threshold: int representing the minimum contour area required to be considered motion.
            mask: ndarray representing a mask to apply to the frame. Defaults to None.
        """

        self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_frame = cv2.GaussianBlur(self._prev_frame, (21, 21), 0)
        self.threshold = threshold
        self.mask = mask

        # Get frame dimensions from initial frame
        self.frame_height, self.frame_width = self._prev_frame.shape[:2]

        # Initialize the queue object
        self.queue = Queue()

    def GetQueue(self):
        """
        Returns the queue object associated with this instance of the motion detector.

        Returns:
            Queue: The associated queue object.
        """
        return self.queue
    
    def detect_motion(self, prev_frame, current_frame):
        """
        Detect motion between the current frame and the previous frame.

        Args:
            current_frame: ndarray representing the current frame to use for motion detection.

        Returns:
            Tuple of boolean (whether motion is detected or not) and the current frame.
        """
        if current_frame is None:
            logger.warning('detect_motion was passed a None current_frame')
            return False, current_frame

        # Check that current frame is same size as previous frame
        if current_frame.shape[:2] != prev_frame[:2]:
            logger.warning('current_frame is not same size as previous frame')
            return False, current_frame

        # Scale frames to 100px height
        height_ratio = 100 / self.frame_height
        width_ratio = 100 / self.frame_width

        area_ratio = (self.frame_height * self.frame_width)

        # Convert frames to grayscale
        background = prev_frame
        grey = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        background = cv2.GaussianBlur(background, (21, 21), 0)
        blured_grey = cv2.GaussianBlur(grey, (21, 21), 0)

        # Find absolute difference between frames
        diff = cv2.absdiff(background, blured_grey)

        # Apply threshold to difference image
        thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

        # Dilate thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in thresholded image
        cnts, res = cv2.findContours(thresh.copy(),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if motion is detected
        is_motion = False
        bounding_image = current_frame.copy()
        for contour in cnts:
            if cv2.contourArea(contour) < self.threshold:
                continue
            is_motion = True

        # Update previous frame
        prev_frame = blured_grey.copy()

        return is_motion, current_frame
    
    def _start_motion_detection_thread(self, queue:Queue, prev_frame):
        while True:
            if self._killDaemon:
                print('killing motion detection demon')
                break
            current_frame = queue.get()
            isMotion = self.detect_motion(prev_frame, current_frame)
            if isMotion:
                print('motion detected!!!')
                #todo: NEED TO IMPLEMENT SEND LOGIC
    
    def Start(self):
        self._killDaemon = False  # initialize flag to False
        thread = threading.Thread(target=self._start_motion_detection_thread, 
                                  name="motion_detection_thread", daemon=True
            , args=(self.queue, self._prev_frame))
        thread.start()
    def Stop(self):
        """
        Stops the camera worker daemon
        """
        pass
        self._killDaemon = True  # set flag to stop capture thread


