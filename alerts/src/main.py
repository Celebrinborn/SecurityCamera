import io
import logging
from log_config import configure_logging

from typing import List, Optional
from dataclasses import dataclass

import threading

from src.strategies.message_strategy import MessageStrategy
from src.strategies.person_detected_strategy import PersonDetectedStrategy

from kafka import KafkaConsumer, KafkaProducer
from avro.io import DatumReader, DataFileReader
import io

configure_logging()
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class ObjectDetection:
    bounding_box: BoundingBox
    classification: str
    certainty: float

@dataclass
class ObjectDetectionResult:
    frame_id: str
    camera_name: str
    detections: List[ObjectDetection]

class KafkaConsumerManager(threading.Thread):
    thread:threading.Thread.Thread
    def __init__(self, topic, group_id, schema):
        self.thread = threading.Thread(target=self.run, args=())
        self.topic = topic
        self.group_id = group_id
        self.schema = schema
        self.consumer = KafkaConsumer(
            self.topic,
            group_id=self.group_id
        )
        self.thread.start()


    def run(self):
        for message in self.consumer:
            message_data = io.BytesIO(message.value)
            message_data.seek(0)

            # Deserialize using Avro
            avro_reader = DataFileReader(message_data, DatumReader())
            for record in avro_reader:
                detection_result = ObjectDetectionResult(
                    frame_id=record['frame_id'],
                    camera_name=record['camera_name'],
                    detections=[
                        ObjectDetection(
                            bounding_box=BoundingBox(
                                x1=detection['bounding_box']['x1'],
                                y1=detection['bounding_box']['y1'],
                                x2=detection['bounding_box']['x2'],
                                y2=detection['bounding_box']['y2']
                            ),
                            classification=detection['classification'],
                            certainty=detection['certainty']
                        )
                        for detection in record['detections']
                    ]
                )

                # Process the ObjectDetectionResult instance
                # E.g., print, log, or perform some action
                print(detection_result)

            avro_reader.close()

    def stop(self):
        self.consumer.close()



class Main:
    def __init__(self):
        self.strategies:List[MessageStrategy] = [PersonDetectedStrategy()]

    def initialize_strategies(self):
        # Initialize and run strategies
        for strategy in self.strategies:
            strategy.start_processing()

    # Context manager methods
    def __enter__(self):
        self.initialize_strategies()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean-up code
        pass


if __name__ == '__main__':
    logger.info('Starting application')
    with Main() as main:
        logger.info('Application started')
        

        # Usage example
        topic = 'your_kafka_topic'
        group_id = 'your_consumer_group'
        schema = 'your_avro_schema'  # Replace with your actual Avro schema

        # Create and start the Kafka consumer thread
        consumer_thread = KafkaConsumerManager(topic, group_id, schema)


        