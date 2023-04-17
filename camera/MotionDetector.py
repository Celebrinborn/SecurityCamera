import cv2
import logging
import threading
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)

class MotionDetector:
    _kill_the_daemon_event: threading.Event()  # flag to abort capture daemon
    def __init__(self, threshold=1000, mask=None):
        """
        Constructor for MotionDetector class.

        Args:
            frame: ndarray representing the initial frame to use for motion detection.
            threshold: int representing the minimum contour area required to be considered motion.
            mask: ndarray representing a mask to apply to the frame. Defaults to None.
        """

        self.threshold = threshold
        self.mask = mask

        # Initialize the queue object
        self.queue = Queue()
        self._kill_the_daemon_event = threading.Event()

    def __enter__(self):
        logger.debug('running motionDetector class enter')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.Stop()
        logger.debug('running motionDetector class exit')

    def GetQueue(self):
        """
        Returns the queue object associated with this instance of the motion detector.

        Returns:
            Queue: The associated queue object.
        """
        logger.debug(f'retriving motion_detector queue {type(self.queue)}')
        return self.queue
    
    
    def _start_motion_detection_thread(queue:Queue, _kill_the_daemon_event: threading.Event(), threshold:int):
        def detect_motion(threshold, frame, drawFrame=False):
            """
            Generator function that detects motion between frames.

            Args:
                threshold (int): motion threshold to trigger detection
                frame (ndarray): first frame to initialize previous frame

            Yields:
                bool: True if motion is detected, False otherwise
            """
            frame_height, frame_width, _ = frame.shape
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame = frame # preload the frame
            while True:

                # Convert frames to grayscale
                grey = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
                blured_grey = cv2.GaussianBlur(grey, (21, 21), 0)

                # Find absolute difference between frames
                diff = cv2.absdiff(prev_frame, blured_grey)

                # Apply threshold to difference image
                thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

                # Dilate thresholded image to fill in holes
                thresh = cv2.dilate(thresh, None, iterations=2)

                # Find contours in thresholded image
                cnts, res = cv2.findContours(thresh.copy(),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Check if motion is detected
                bounding_image = None
                is_motion = False
                if drawFrame:
                    bounding_image = current_frame.copy()
                for contour in cnts:
                    if cv2.contourArea(contour) < threshold:
                        continue
                    is_motion = True
                    if drawFrame:
                        # Draw bounding box on frame with motion
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(bounding_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                prev_frame = grey  # update previous frame with current frame

                if drawFrame:
                    current_frame = yield is_motion, bounding_image
                else:
                    current_frame = yield is_motion, None

        # get first frame
        current_frame = queue.get()
        logger.debug(f'initial pull of queue from motiondetector is {type(current_frame)}')

        # type check
        if not isinstance(current_frame, np.ndarray): raise TypeError(f'currentframe is type {type(current_frame)} should be np.ndarray')

        # initiate motion_detector
        logger.debug('initiating detect_motion')
        detector = detect_motion(threshold= threshold, frame=current_frame, drawFrame=True) # TODO REMOVE DRAWFRAME
        next(detector)
        while not _kill_the_daemon_event.is_set():
            # logger.debug(f'current frame type: {type(current_frame)}')
            frame = queue.get()
            if not isinstance(frame, np.ndarray):
                logger.warning(f'queue has a non-ndarray passed. type was {type(frame)}')
                continue
            isMotion, _frame = detector.send(frame)

            # debug code, comment out in production!!!!
            cv2.imshow('motion', _frame)
            cv2.waitKey(1)

            

            if isMotion:
                logger.debug('motion detected')
                #todo: NEED TO IMPLEMENT SEND LOGIC
            else:
                logger.debug('no motion detected')
        logger.warning('shutting down motion detector object')
    
    def Start(self):
        logger.debug(f'type of queue is {type(self.queue)}')
        thread = threading.Thread(target=MotionDetector._start_motion_detection_thread,
            name="motion_detection_thread",
            daemon=True,
            args=(self.queue, self._kill_the_daemon_event, self.threshold))
        thread.start()


    def Stop(self):
        """
        Stops the camera worker daemon
        """
        self._kill_the_daemon_event.set()  # set flag to stop capture thread


