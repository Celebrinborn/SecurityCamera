import logging
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta

import cv2
import os

Motion_Threshold = 500000

def FormatTime(time:datetime):
    return time.strftime(r"%Y%m%d_%H%M%S")

def DetectMotion(currentFrame, prevFrame, threshold:int) -> bool:
    print(threshold)
    if currentFrame is None:
        logger.warning('DetectMotion was passed a None currentFrame')
        return False, currentFrame
    if prevFrame is None:
        logger.warning('DetectMotion was passed a None prevFrame')
        return False, prevFrame

    diff = cv2.absdiff(prevFrame, currentFrame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    isMotion = False
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < threshold:
            continue
        cv2.rectangle(prevFrame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        isMotion = True  
    cv2.drawContours(prevFrame, contours, -1, (0, 255, 0), 2)
    return isMotion, prevFrame
# on open find all files and add them to queue

#fifo queue
# add files to queue. delete from queue when reaching size





print('starting')
try:
    _video_path = os.path.join('camera','samples','Front East facing East - Mon Dec 19 13-48-45 2022.mp4')
    print(_video_path)
    cap = cv2.VideoCapture(_video_path)
    # read first two frames
    try:
        ret, prevFrame = cap.read()
        ret, currentFrame = cap.read()
    except BaseException as e:
        logger.critical('unable to read initial two frames for unknown reason. the application will now close', exc_info=True)
        logger.critical(e)
        raise e
    while cap.isOpened():
        try:
            print('reading frame')
            prevFrame = currentFrame
            try:
                flag, currentFrame = cap.read()
            except BaseException as e:
                logging.error(f'unable to read current frame, {str(e)}', exc_info=True)
                raise e

            if currentFrame is None:
                logging.warning(f'currentFrame is none at ')


            # check for motion
            _isMotion, outlinedFrame = DetectMotion(currentFrame, prevFrame, Motion_Threshold)
            if ret:
                # outlinedFrame
                cv2.imshow('Frame', prevFrame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # save frame to video
            # try:
            #     video_writer.write(currentFrame)
            # except BaseException as e:
            #     logger.error('unable to save frame for unknown reason', exc_info=True)

        except BaseException as e:
            logger.critical('an unknown frame reading error has occured. continuing...')
            logger.critical(e, exc_info=True)

except BaseException as e:
    logger.fatal('unhandled exception has occured. this may be an issue with the capture failing', exc_info=True)
    logger.fatal(e, exc_info=True)
finally:
    cap.release()