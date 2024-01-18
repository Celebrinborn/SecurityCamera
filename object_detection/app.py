import dataclasses
from enum import unique
from math import log
from optparse import Option
import os
from queue import PriorityQueue
import io
import queue
import re
import threading
import logging
from dataclasses import dataclass, asdict
import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import KafkaError
from avro.io import DatumWriter, DatumReader, BinaryEncoder, BinaryDecoder, AvroTypeException
from avro.datafile import DataFileReader, DataFileWriter
from avro import schema

import json

import numpy as np


from transformers import YolosImageProcessor, YolosForObjectDetection, PreTrainedModel
from PIL import Image
from pathlib import Path
import torch
from typing import List, Optional

import subprocess


# Set up logging
from log_config import configure_logging

configure_logging()
logger = logging.getLogger()

# log running location
logger.debug(f"Running from {Path.cwd()}")
# log what files are in the current directory
logger.debug(f"Files in current directory: {list(Path.cwd().iterdir())}")

logger.debug(f"Environment variables: {os.environ}")


logger.debug(f"pip freeze: {subprocess.run(['pip', 'freeze'], capture_output=True, text=True)}")

# mute logging from kafka to exception and above only
logging.getLogger("kafka").setLevel(logging.ERROR)
# mute logging from avro to exception and above only
logging.getLogger("avro").setLevel(logging.ERROR)

def replace_bytes_in_exception(exception):
    def replace_bytes(obj):
        if isinstance(obj, bytes):
            return f"Byte: {len(obj)}"
        elif isinstance(obj, (list, tuple)):
            return type(obj)(replace_bytes(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: replace_bytes(value) for key, value in obj.items()}
        return obj

    if exception.args:
        new_args = replace_bytes(exception.args)
        exception.args = new_args

    return exception


# Dataclass for Kafka message
@dataclass
class MotionMessageQueueItem:
    camera_name: str
    priority: float
    guid: str
    creation_timestamp: float
    frame_jpg: Image.Image
    frame_ndarray: np.ndarray
    motion_amount: float
    timeout: int

    def __str__(self):
        return self.guid
    def __repr__(self):
        return f'MotionMessageQueueItem(camera_name={self.camera_name}, priority={self.priority}, guid={self.guid})'

    def __lt__(self, other): # used for priority queue sorting
        # Compare based on the 'priority' attribute
        return self.priority < other.priority
    def __getitem__(self, index): # used for priority queue sorting
        # Allow accessing attributes by index
        if index == 0: return self.priority
        else: raise NotImplementedError(f"Index {index} not supported")

    @staticmethod
    def from_kafka_message(message: ConsumerRecord) -> 'MotionMessageQueueItem':
        # Create a BinaryDecoder from the Avro bytes

        avro_buffer = io.BytesIO(message)
        
        avro_buffer.seek(0)
        # Create a reader for Avro data using the specified schema
        reader: DataFileReader = DataFileReader(avro_buffer, DatumReader())

        try:
            avro_data: dict[str, Any] = next(reader) # type: ignore
        except StopIteration:
            logger.error("No data in Avro buffer")
            raise Exception("No data in Avro buffer")
        finally:
            reader.close()

        # Extract relevant fields from the Avro data
        try:
            camera_name = avro_data['camera_name']
            priority = avro_data['priority']
            guid = avro_data['guid']
            creation_timestamp = avro_data['creation_timestamp']
            # extract to numpy then to image
            frame_jpg = Image.open(io.BytesIO(avro_data['frame_jpg']))
            frame_ndarray = np.frombuffer(avro_data['frame_jpg'], np.uint8)
            motion_amount = avro_data['motion_amount']
            timeout = avro_data['timeout']
        except KeyError as e:
            logger.error(f"Missing key in Avro data: {e}")
            raise
        # Create and return an instance of PriorityQueueItem
        return MotionMessageQueueItem(
            camera_name=camera_name,
            priority=priority,
            guid=guid,
            creation_timestamp=creation_timestamp,
            frame_jpg=frame_jpg,
            frame_ndarray=frame_ndarray,
            motion_amount=motion_amount,
            timeout=timeout
        )

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
    jpg: Image.Image
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
            "frame_jpg": self.jpg_ndarray.tobytes(),
            "detections": [detection.to_dict() for detection in self.detections],
        }

# Kafka Consumer
class KafkaImageConsumer:
    queue: PriorityQueue
    # Environment variables with defaults
    topic: str = os.getenv("CONSUMER_TOPIC", "camera_motion_threshold_exceeded")
    
    def __init__(self, queue: PriorityQueue):
        self.KAFKA_CONNECTION_STRING: str = os.getenv("KAFKA_BOOTSTRAP_SERVER", "localhost:9092")
        if 'KAFKA_BOOTSTRAP_SERVER' not in os.environ:
            logger.warning(f"KAFKA_BOOTSTRAP_SERVER environment variable not set. Defaulting to {self.KAFKA_CONNECTION_STRING}")
        try:
            logger.info(f'Initializing Kafka Consumer. bootstrap server: {self.KAFKA_CONNECTION_STRING}')
            self.consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.KAFKA_CONNECTION_STRING])
            logger.info("Kafka Consumer initialized")
        except KafkaError as e:
            logger.error(f"Error initializing Kafka Consumer: {e}")
            raise

        self.queue = queue
        
        logger.debug("Starting Kafka Consumer thread")
        # start monitoring for kafka messages in a separate thread
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            for message in self.consumer:
                assert isinstance(message, ConsumerRecord), f"Unexpected message type: {type(message)}"
                item = MotionMessageQueueItem.from_kafka_message(message.value)
                self.queue.put(item)
                logger.debug(f"Message added to priority queue: {item}")
        except KafkaError as e:
            logger.error(f"Error in Kafka Consumer loop: {e}")

# Kafka Producer
class KafkaResultProducer:
    KAFKA_CONNECTION_STRING: str 
    topic: str = 'camera_object_detection_results'
    object_detection_result_schema: schema.Schema
    _producer: KafkaProducer
    def __init__(self):
        self.KAFKA_CONNECTION_STRING = os.getenv("KAFKA_BOOTSTRAP_SERVER", "localhost:9092")
        if 'KAFKA_BOOTSTRAP_SERVER' not in os.environ:
            logger.warning(f"KAFKA_BOOTSTRAP_SERVER environment variable not set. Defaulting to {self.KAFKA_CONNECTION_STRING}")

        _success = True
        for i in range(_max_tries:=10):
            try:
                _success = True
                logger.info(f'Initializing Kafka Producer. bootstrap server: {self.KAFKA_CONNECTION_STRING}')
                self._producer = KafkaProducer(bootstrap_servers=[self.KAFKA_CONNECTION_STRING])
                logger.info("Kafka Producer initialized")
                break
            except KafkaError as e:
                logger.error(f"Error initializing Kafka Producer. bootstrap server: {self.KAFKA_CONNECTION_STRING}. Exception: {e}")
                _success = False
                if i == _max_tries - 1:
                    raise e
        if not _success:
            # this should be impossible to reach as the for loop should raise an exception before this
            raise Exception("IT SHOULD BE IMPOSSIBLE TO REACH THIS CONDITION: Unable to initialize Kafka Producer")

        objectDetectionResultSchemaText = Path('ObjectDetectionResult.avsc').read_text()
        self.object_detection_result_schema = schema.parse(objectDetectionResultSchemaText)
        
    def send_warning(self, topic, message):
        raise NotImplementedError("send_warning not implemented")

    def send_result(self, detection_result: DetectionResult):
        # Serialize the DetectionResult object to Avro format
        bytes_writer = io.BytesIO()
        writer = DataFileWriter(bytes_writer, DatumWriter(), self.object_detection_result_schema)
        detection_result_dict = detection_result.to_dict()
        # for debug purposes serialize detection_result_dict to disk using pathlib
        # Path(f'output',f'detection_result_dict{detection_result.frame_id}.json').write_text(str(detection_result_dict))
        try:
            writer.append(detection_result_dict)
            writer.flush()
            bytes_writer.seek(0)
        except AvroTypeException as e:
            logger.error(f'Unable to serialize detection_result_dict : {replace_bytes_in_exception(e)}')
            raise e
            return None


        try:
            # Send the serialized Avro data to the Kafka topic
            future = self._producer.send(self.topic, bytes_writer.read())
            # also show unique detections
            logger.debug(f"Detection result for {detection_result.frame_id} sent to Kafka topic with {len(detection_result.detections)} detections and {[(d.classification, d.certainty) for d in detection_result.detections]}")
            return future
        except KafkaError as e:
            # logger.error(f'Unable to send {self.topic} event: {replace_bytes_in_exception(e)}')
            logger.error(f'Unable to send {self.topic} event: {e}')
# Object Detector
class ObjectDetector:
    queue: PriorityQueue
    image_processor: YolosImageProcessor
    model: YolosForObjectDetection
    def __init__(self):
        model_name = 'hustvl/yolos-tiny'
        model_path = Path('models') / model_name
        try:
            if not model_path.exists():
                pass # TODO: use cache if available
            logger.info(f"Model not found locally. Downloading and caching from {model_name}")
            model_path.mkdir(parents=True, exist_ok=True)
            # I'm using multiple lines with the try catch to get around the fact that the transformers library can return multiple types
            _model = YolosForObjectDetection.from_pretrained(model_name, cache_dir=model_path)
            if isinstance(_model, YolosForObjectDetection): self.model = _model
            else: raise TypeError(f"Unexpected type for model: {type(_model)}")
            _image_processor = YolosImageProcessor.from_pretrained(model_name, cache_dir=model_path)
            if isinstance(_image_processor, YolosImageProcessor): self.image_processor = _image_processor
            else: raise TypeError(f"Unexpected type for image_processor: {type(_image_processor)}")
            logger.debug(f"Model downloaded and cached at {model_path}")
            # else:
                # logger.info(f"Loading model from local cache at {model_path}")
                # # I'm using multiple lines with the try catch to get around the fact that the transformers library can return multiple types
                # _model = YolosForObjectDetection.from_pretrained(model_path)
                # if isinstance(_model, YolosForObjectDetection): self.model = _model
                # else: raise TypeError(f"Unexpected type for model: {type(_model)}")
                # _image_processor = YolosImageProcessor.from_pretrained(model_path)
                # if isinstance(_image_processor, YolosImageProcessor): self.image_processor = _image_processor
                # else: raise TypeError(f"Unexpected type for image_processor: {type(_image_processor)}")
                
        except OSError as e:
            logger.error(f"Error loading model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error loading model: {e}")
            raise

    def process_image(self, item:MotionMessageQueueItem) -> DetectionResult:
        try:
            image = item.frame_jpg
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            # Processing logic here...
            logger.debug("Image processed successfully")

            # model predicts bounding boxes and corresponding COCO classes
            logits = outputs.logits
            bboxes = outputs.pred_boxes

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

            detections: List[Detection] = []

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                # print(
                #     f"Detected {self.model.config.id2label[label.item()]} with confidence "
                #     f"{round(score.item(), 3)} at location {box}"
                # )
                detection = Detection(
                    bounding_box=BoundingBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3])),
                    classification=self.model.config.id2label[label.item()],
                    certainty=round(score.item(), 3)
                )
                detections.append(detection)
            return DetectionResult(
                frame_id=item.guid,
                camera_name=item.camera_name,
                detections=detections,
                jpg=item.frame_jpg,
                jpg_ndarray=np.array(item.frame_ndarray)
            )
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

# Main execution
def main():
    queue: PriorityQueue[MotionMessageQueueItem] = PriorityQueue()
    try:
        consumer = KafkaImageConsumer(queue)
        producer = KafkaResultProducer()
        detector = ObjectDetector()

        logger.info("Starting main loop")
        while True:
            # check if queue is full
            if queue.full():
                logger.warning(f"Queue is full!!!")
                try: producer.send_warning("queue_full", f"priority queue is full, {queue.qsize()} items in queue")
                except NotImplementedError as e: logger.warning(f"{e}") 
            item: MotionMessageQueueItem = consumer.queue.get()
            logger.debug(f"Processing item from queue: {item.camera_name} {item.guid} {item.creation_timestamp} {item.motion_amount} {item.timeout}")

            # if item age is greater than timeout, skip
            if item.creation_timestamp + item.timeout < time.time():
                logger.warning(f"Skipping item from queue due to timeout: {item.camera_name} {item.guid} {item.creation_timestamp} {item.motion_amount} {item.timeout}")
                continue
            detection_result:DetectionResult  = detector.process_image(item)
            logger.debug(f'processing time for frame {detection_result.frame_id=}: {time.time() - item.creation_timestamp=}')
            producer.send_result(detection_result)




    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.fatal(f"Unhandled exception in main: {e}")
        raise

