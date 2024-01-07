from dataclasses import dataclass, todict
import numpy as np
from typing import List, Optional






# Dataclass for YOLOs output
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    def asdict(self):
        # Return a dictionary of bounding box values
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

@dataclass
class Detection:
    bounding_box: BoundingBox # You can use a namedtuple if needed
    classification: str
    certainty: float
        # # example of how to render the image with bounding boxes
        # def render(self, detection_result: DetectionResult):
            # import cv2
            # import numpy as np
            # # convert bytes to numpy array
            # nparr = np.frombuffer(detection_result.jpg, np.uint8)
            # # decode image
            # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # # draw bounding boxes
            # # Print the values of detection.bounding_box
            # for detection in detection_result.detections:
            #     x1, y1, x2, y2 = map(int, detection.bounding_box)
            #     cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #     cv2.putText(img_np, detection.classification, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # # show image
            # cv2.imshow("image", img_np)
            # cv2.waitKey(1)
    def to_dict(self):
        return {
            'bounding_box': self.bounding_box.asdict(),
            "classification": self.classification,
            "certainty": self.certainty,
        }
        

@dataclass
class DetectionResult:
    frame_id: str
    camera_name: str
    # jpg: Image.Image
    jpg_ndarray: np.ndarray
    detections: List[Detection]

    def __repr__(self) -> str:
        return f'DetectionResult: {self.frame_id}: {len(self.detections)})'

    def __str__(self) -> str:
        return f'DetectionResult: {self.frame_id}: {len(self.detections)})'

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            # "jpg": self.jpg_ndarray.tobytes(),
            "detections": [detection.to_dict() for detection in self.detections],
        }

