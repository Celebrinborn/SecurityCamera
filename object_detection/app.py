import dataclasses
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
from charset_normalizer import detect
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import KafkaError
from avro.io import DatumWriter, DatumReader, BinaryEncoder, BinaryDecoder
from avro.datafile import DataFileReader, DataFileWriter
from avro import schema

import json
from numpy import byte
from sympy import fu, im


from transformers import YolosImageProcessor, YolosForObjectDetection, PreTrainedModel
from PIL import Image
from pathlib import Path
import torch
from typing import List, Optional

# Set up logging
from log_config import configure_logging
configure_logging()
logger = logging.getLogger()

# mute logging from kafka to exception and above only
logging.getLogger("kafka").setLevel(logging.ERROR)


# Dataclass for Kafka message
@dataclass
class MotionMessageQueueItem:
    camera_name: str
    priority: float
    guid: str
    creation_timestamp: float
    frame_jpg: Image.Image
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
            frame_jpg = Image.open(io.BytesIO(avro_data['frame_jpg']))
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
            motion_amount=motion_amount,
            timeout=timeout
        )

# Dataclass for YOLOs output
@dataclass
class Detection:
    bounding_box: tuple[int, int, int, int]  # You can use a namedtuple if needed
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
        return asdict(self)
        

@dataclass
class DetectionResult:
    frame_id: str
    camera_name: str
    jpg: Image.Image
    detections: List[Detection]

    def __repr__(self) -> str:
        return f'detection_count: {self.frame_id}: {len(self.detections)})'

    def to_dict(self):
        assert isinstance(self.jpg, bytes), f"Unexpected type for jpg: {type(self.jpg)}"
        return {
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            "jpg": self.jpg,
            "detections": [detection.to_dict() for detection in self.detections],
        }

# Kafka Consumer
class KafkaImageConsumer:
    queue: PriorityQueue
    # Environment variables with defaults
    KAFKA_CONNECTION_STRING: str = os.getenv("KAFKA_CONNECTION_STRING", "localhost:9092")
    topic: str = os.getenv("CONSUMER_TOPIC", "camera_motion_threshold_exceeded")
    
    def __init__(self, queue: PriorityQueue):
        try:
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
    KAFKA_CONNECTION_STRING: str = os.getenv("KAFKA_CONNECTION_STRING", "localhost:9092")
    topic: str = os.getenv("PRODUCER_TOPIC", "camera_object_detection_results")
    object_detection_result_schema: schema.Schema
    _producer: KafkaProducer
    def __init__(self):
        try:
            self._producer = KafkaProducer(bootstrap_servers=[self.KAFKA_CONNECTION_STRING])
            logger.info("Kafka Producer initialized")
        except KafkaError as e:
            logger.error(f"Error initializing Kafka Producer: {e}")
            raise
        objectDetectionResultSchemaText = Path('ObjectDetectionResult.avsc').read_text()
        self.object_detection_result_schema = schema.parse(objectDetectionResultSchemaText)
        


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
        except Exception as e:
            logger.error(f'Unable to serialize detection_result_dict : {e}')
            logger.error(f'detection result dict is {detection_result_dict}')

        try:
            # Send the serialized Avro data to the Kafka topic
            future = self._producer.send(self.topic, bytes_writer.read())
            logger.debug(f"Detection result for {detection_result.frame_id} sent to Kafka topic")
            return future
        except KafkaError as e:
            logger.error(f'Unable to send camera_motion_threshold_exceeded event : {e}')
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
                    bounding_box=box,
                    classification=self.model.config.id2label[label.item()],
                    certainty=round(score.item(), 3)
                )
                detections.append(detection)
            return DetectionResult(
                frame_id=item.guid,
                camera_name=item.camera_name,
                detections=detections,
                jpg=item.frame_jpg
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

        logger.info("Kafka Consumer thread started")

        while True:
            item: MotionMessageQueueItem = consumer.queue.get()
            logger.debug(f"Processing item from queue: {item}")

            print(f"Processing item from queue: {item.camera_name} {item.guid} {item.creation_timestamp} {item.motion_amount} {item.timeout}")
            detection_result:DetectionResult  = detector.process_image(item)
            producer.send_result(detection_result)




    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        raise

if __name__ == "__main__":
    main()

