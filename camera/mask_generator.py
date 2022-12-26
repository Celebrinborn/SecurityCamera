import cv2
import numpy as np
import os
from typing import Tuple
from dataclasses import dataclass

def DrawMask(image:np.ndarray, mask:np.array):
    img = image.copy()
    return cv2.fillPoly(img, pts=[mask], color=(255,255,255))

@dataclass(init=False)
class PointCloud:
    coordinates:list
    frames:list
    mask:np.array
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
        if self.mask is not None:
            self.mask = None
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
        if self.mask is not None:
            DrawMask(img, self.mask)
            
        return img

    def UpdateDisplay(self) -> str:
        img = self.Get_Display_Image()
        window_name = 'image'
        cv2.imshow(window_name,img)
        return window_name

    def BuildMask(self):
        # if ran redundantly
        if self.mask is not None:
            return

        # add last point
        prev_coordinate = self.coordinates[-1]
        edge_coordinate = self.CoordinatesofNearestEdge(prev_coordinate[0], prev_coordinate[1])
        self.AddPoint(edge_coordinate[0], edge_coordinate[1])

        # if on seperate sides, add corner point
        first_coordinate = self.coordinates[0]
        height, width, _ = self.base_image.shape

        # left = 0, 
        # top = 0
        # right = width
        # bot = height

        # top left corner
        if (first_coordinate[0] == 0 and edge_coordinate[1] == 0) or (edge_coordinate[0] == 0 and first_coordinate[1]==0): 
            corner_coordinate = (0,0)
        # top right corner
        if (first_coordinate[0] == width and edge_coordinate[1] == 0) or (edge_coordinate[0] == width and first_coordinate[1] == 0):
            corner_coordinate = (width, 0)
        # bottom left corner
        if (first_coordinate[0] == 0 and edge_coordinate[1] == height) or (edge_coordinate[0] == 0 and first_coordinate[1]==height):
            corner_coordinate = (0, height)
        # bottom right corner
        if (first_coordinate[0]== width and edge_coordinate[1] == height) or (edge_coordinate[0] == width and first_coordinate[1] == 0):
            corner_coordinate = (width, height)
        
        # if the line is all the way across
        if (first_coordinate[0] == 0 and edge_coordinate[0] == width) or (first_coordinate[0] == width and edge_coordinate[0] == 0):
            raise NotImplementedError()
        if (first_coordinate[1] == 0 and edge_coordinate[1] == height) or (first_coordinate[1] == height and edge_coordinate[1] == 0):
            raise NotImplementedError()
        self.AddPoint(corner_coordinate[0], corner_coordinate[1])

        # generate nparray as mask
        self.mask = np.array(self.coordinates, dtype=np.int32)
    
    def DeleteMask(self):
        self.mask = None
    
    def drawInstructions():
        pass
        # font=cv2.FONT_HERSHEY_PLAIN
        # cv2.putText(frames[-1],_instructions,(50,100),font,1,(255,255,255))
        # cv2.imshow("image",frames[-1])

# coordinates = []
# frames = []
# mask = None



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
_preview_files = [x for x in os.listdir() if 'preview.jpg' in x]
for file in _preview_files:
    print(f'{_preview_files.index(file)}: {file}')
_invalid_selection = True
while _invalid_selection:
    selection  = input('please select a preview image number to work on: ')
    if not str(selection).isdigit():
        print(f'{str(selection)} is not a valid int')
        continue
    if int(selection) not in range(len(_preview_files)):
        print(f'{str(selection)} is not in the list of available files')
        continue
    _invalid_selection = False

preview_file_name = _preview_files[int(selection)]
base_file_name = preview_file_name.partition('_preview.jpg')[0]


print(f'you are now working on camera: {base_file_name}')

frame = cv2.imread(preview_file_name)
cv2.destroyAllWindows()
print('continuing')

pointcloud = PointCloud(frame)
window_name = pointcloud.UpdateDisplay()

_instructions = """press q to quit, enter to submit. Place markers with left mouse and undo with right mouse.   
        # Once finished drawing mask press f to fill and i to invert. finally press enter when finished"""
print(_instructions)


cv2.setMouseCallback(window_name,onClick)
while True:
    button = cv2.waitKey()
    if button == 13:
        # submit (enter)
        if pointcloud.mask is not None:
            np.save(os.path.join(f'{base_file_name}_mask'), pointcloud.mask)
        break
    if button == 113 or button == 81:
        # cancel (q and Q)
        break
    if button == 105 or button ==  73: #i
        print('inverting')
    if button == 102 or button ==  70: #f
        pointcloud.BuildMask()
        pointcloud.UpdateDisplay()
    print(button)
    # enter is 13
    # q is 113
cv2.destroyAllWindows()