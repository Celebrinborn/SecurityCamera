import cv2
import numpy as np
import os
from typing import Tuple
from dataclasses import dataclass

@dataclass(init=False)
class PointCloud:
    coordinates:list
    frames:list
    mask:np.ndarray
    base_image:np.ndarray

    user_instructions:str
    def __init__(self, base_image:np.ndarray) -> None:
        self.coordinates = []
        self.frames = []
        self.mask = None
        self.base_image = base_image

    def CoordinatesofNearestEdge(self, x:int, y:int) -> Tuple[int, int]:
        height, width, _ = self.base_image.shape
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

    def AddPoint(self, x:int, y:int):
        coordinate = (x,y)
        if len(self.coordinates) > 1:
            self.coordinates.append(coordinate)
            return
        elif len(self.coordinates) == 0:
            edge_coordinate_x, edge_coordinate_y = self.CoordinatesofNearestEdge(coordinate[0], coordinate[1])
            edge_coordinate = (edge_coordinate_x, edge_coordinate_y)
            self.coordinates.append(edge_coordinate)
            self.coordinates.append(coordinate)

    def RemovePoint(self):
        if len(self.coordinates) > 1:
            self.coordinates.pop()
            return
        elif len(self.coordinates) == 0:
            return
        raise Exception('add point had 1 coordinate, this should never happen')
            

    def DrawLine(self, image:np.ndarray, prevX:int, prevY:int, x:int, y:int) -> np.ndarray:
        img = image.copy()
        return cv2.line(img,(prevX,prevY),(x,y),(0,0,255),5)
    
    def Get_Display_Image(self):
        # deep copy of image to avoid overwriting
        img = self.base_image.copy()
        
        # if there is at least 1 coordinate
        if len(self.coordinates) > 0:
            for coordinate in self.coordinates:
                x, y = coordinate
                # place marker at mouse click location
                cv2.circle(img,(x,y),3,(255,255,255),-1)
                # write the coordinates on the screen
                strXY='(x:'+str(x)+',y:'+str(y)+')'
                font=cv2.FONT_HERSHEY_PLAIN
                cv2.putText(img,strXY,(x+10,y-10),font,1,(255,255,255))
                # skip first coordinate
                if self.coordinates[0] != coordinate:
                    prev_x, prev_y = prevCoordinate
                    img = self.DrawLine(img, prev_x, prev_y,x,y)
                prevCoordinate = coordinate
        return img

    def Display(self) -> str:
        img = self.Get_Display_Image()
        window_name = 'image'
        cv2.imshow(window_name,img)
        return window_name

    def FillPoly(self, x, y, coordinates, frames):
        img = frames[-1]
        coordinate = coordinates[-1]

        edge_x, edge_y = self.CoordinatesofNearestEdge(coordinate[0], coordinate[1], img)
        lastLineImage = self.DrawLine(img, edge_x, edge_y, x, y)
        mask = coordinates
    
    def drawInstructions():
        pass
        # _instructions = """press q to quit, enter to submit. Place markers with left mouse and undo with right mouse.   
        # Once finished drawing mask press f to fill and i to invert. finally press enter when finished"""
        # font=cv2.FONT_HERSHEY_PLAIN
        # cv2.putText(frames[-1],_instructions,(50,100),font,1,(255,255,255))
        # cv2.imshow("image",frames[-1])

# coordinates = []
# frames = []
# mask = None

def FirstClick(): pass

def SecondClick(): pass

def Fill(): pass

def Invert(): pass



def onClick(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        # deep copy the latest frame and cache it as img
        pointcloud.AddPoint(x,y)
        img = pointcloud.Get_Display_Image()
        cv2.imshow("image",img)
    elif event==cv2.EVENT_RBUTTONDOWN:
        pointcloud.RemovePoint()
        img = pointcloud.Get_Display_Image()
        cv2.imshow("image",img)


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

pointcloud = PointCloud(frame)
window_name = pointcloud.Display()

cv2.setMouseCallback(window_name,onClick)
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
        # FillPoly()
    print(button)
    # enter is 13
    # q is 113
cv2.destroyAllWindows()