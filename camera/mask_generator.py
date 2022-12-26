import cv2
import numpy as np
import os
from typing import Tuple

coordinates = []
frames = []
mask = None

def FillPoly(x, y, coordinates, frames):
    img = frames[-1]
    coordinate = coordinates[-1]

    edge_x, edge_y = CoordinatesofNearestEdge(coordinate[0], coordinate[1], img)
    lastLineImage = DrawLine(img, edge_x, edge_y, x, y)
    mask = coordinates

def DrawLine(img:np.ndarray, prevX:int, prevY:int, x:int, y:int) -> np.ndarray:
    new_image = img.copy()
    return cv2.line(new_image,(prevX,prevY),(x,y),(0,0,255),5)

def CoordinatesofNearestEdge(x:int, y:int, img:np.ndarray) -> Tuple[int, int]:
    height, width, _ = img.shape
    distance_to_nearest_edge = {width-x:'r', x:'l', height-y:'b', y:'t'}
    if distance_to_nearest_edge[min(distance_to_nearest_edge)] == 'l':
        return (0, y)
    elif distance_to_nearest_edge[min(distance_to_nearest_edge)] == 'r':
        return (width, y)
    elif distance_to_nearest_edge[min(distance_to_nearest_edge)] == 't':
        return (x, 0)
    elif distance_to_nearest_edge[min(distance_to_nearest_edge)] == 'b':
        return (x, height)
    else:
        raise Exception('unable to detect nearest edge. this should be impossible')






def onClick(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        # deep copy the latest frame and cache it as img
        img = frames[-1].copy()

        # place marker at mouse click location
        cv2.circle(img,(x,y),3,(255,255,255),-1)
        
        coordinates.append((x,y))

        # write the coordinates on the screen
        strXY='(x:'+str(x)+',y:'+str(y)+')'
        font=cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img,strXY,(x+10,y-10),font,1,(255,255,255))
        
        if len(coordinates) > 1:
            # get last coordinate
            prevX, prevY = coordinates[-2]
            # draw line from previous mouse click position to current mouse click position
            img = DrawLine(img, prevX, prevY, x, y)
            
        else:
            # if this is the first postion, draw a line from the nearest edge of the screen
            edge_x, edge_y = CoordinatesofNearestEdge(x, y, img)
            print(f'edge at: {edge_x}, {edge_y}')
            # img = DrawLine(img, edge_x, edge_y, x, y)
            img = DrawLine(img, edge_x, edge_y, x, y)

        # append image to list of frames so I can undo later
        frames.append(img)

        cv2.imshow("image",img)
        print(f'len {len(frames)}')
    elif event==cv2.EVENT_RBUTTONDOWN:
        if len(frames) > 1:
            # remove last added coordinate
            frames.pop()
            coordinates.pop()
            # display new last drawn coordinate
            cv2.imshow("image",frames[-1])


# get camera base image
print(os.listdir())
Camera_path = os.path.join('camera','samples', r'Front East looking West - Thu Dec 22 13-18-24 2022.mp4') 
cap = cv2.VideoCapture(Camera_path)
print('opened cap')
_framecount = 0
while cap.isOpened():
    _framecount = _framecount + 1
    ret, frame = cap.read()
    if ret:
        cv2.imshow('background', frame)
    else:
        print('reached end of video')
        break
    if _framecount > 15:
        print('retreived frame')
        break
cap.release()
cv2.destroyAllWindows()
print('continuing')

frames.append(frame)
_instructions = """press q to quit, enter to submit. Place markers with left mouse and undo with right mouse.   
Once finished drawing mask press f to fill and i to invert. finally press enter when finished"""
font=cv2.FONT_HERSHEY_PLAIN
cv2.putText(frames[-1],_instructions,(50,100),font,1,(255,255,255))
cv2.imshow("image",frames[-1])

print(frame.shape)
cv2.setMouseCallback("image",onClick)
while True:
    button = cv2.waitKey()
    if button == 13:
        # submit (enter)
        break
    if button == 113 or button == 81:
        # cancel (q and Q)
        break
    if button == 105 or button ==  73: #i
        print('inverting')
    if button == 102 or button ==  70: #f
        pass
        FillPoly()
    print(button)
    # enter is 13
    # q is 113
cv2.destroyAllWindows()