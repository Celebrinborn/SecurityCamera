import logging
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta

import cv2
import os

Motion_Threshold = 100

def FormatTime(time:datetime):
    return time.strftime(r"%Y%m%d_%H%M%S")

def DetectMotion(mask, frame_width, frame_height, currentFrame, prevFrame, threshold:int) -> bool:
    # print(threshold)
    print(type(prevFrame))
    if currentFrame is None:
        logger.warning('DetectMotion was passed a None currentFrame')
        return False, currentFrame
    if prevFrame is None:
        logger.warning('DetectMotion was passed a None prevFrame')
        return False, prevFrame


    # height scale ratioq
    height_ratio = 100 / frame_height
    width_ratio = 100 / frame_width

    area_ratio = (frame_height * frame_width)

    background = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background,(21,21), 0)

    gray = cv2.cvtColor(currentFrame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21), 0)
    
    diff = cv2.absdiff(background,gray)

    thresh = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations = 2)

    cnts,res = cv2.findContours(thresh.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_image = currentFrame.copy()
    for contour in cnts:
        if cv2.contourArea(contour) < threshold :
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(bounding_image,(x,y),(x+w,y+h),(0,255,0), 3)

    # return currentFrame


    # on open find all files and add them to queue

#fifo queue
# add files to queue. delete from queue when reaching size





print('starting')
try:
    _video_path = os.path.join('camera','samples','Front East looking West - Thu Dec 22 17-10-40 2022.mp4')
    print(_video_path)
    cap = cv2.VideoCapture(_video_path)
    # read first two frames
   
    ret, prevFrame = cap.read()
    ret, currentFrame = cap.read()
    while cap.isOpened():
    
        # print('reading frame')
        prevFrame = currentFrame

        flag, currentFrame = cap.read()


        if currentFrame is None:
            logging.warning(f'currentFrame is none at ')


        # check for motion
        # _isMotion, outlinedFrame = 
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detected_frame = DetectMotion(frame_width, frame_height, currentFrame, prevFrame, Motion_Threshold)
        if ret:
            # outlinedFrame
            pass
            # cv2.imshow('Frame', currentFrame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # save frame to video
        # try:
        #     video_writer.write(currentFrame)
        # except BaseException as e:
        #     logger.error('unable to save frame for unknown reason', exc_info=True)

except BaseException as e:
    logger.fatal('unhandled exception has occured. this may be an issue with the capture failing', exc_info=True)
    logger.fatal(e, exc_info=True)
finally:
    cap.release()