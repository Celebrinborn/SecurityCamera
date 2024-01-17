import io
import logging
import os
from log_config import configure_logging

from typing import List, Optional
from dataclasses import dataclass

import threading

from strategies.message_strategy import MessageStrategy
from strategies.person_detected_strategy import PersonDetectedStrategy

from kafka import KafkaConsumer, KafkaProducer
from avro.io import DatumReader
import io
from avro.datafile import DataFileReader

import dotenv
from pathlib import Path

configure_logging()
logger = logging.getLogger(__name__)

# Load environment variables from .env file
if Path('.env').exists():
    logger.info('Loading environment variables from .env file')
    dotenv.load_dotenv('.env')
    logger.info(f'Environment variables loaded from .env file: {", ".join([key for key, value in os.environ.items()])}')

# mute logging from kafka to exception and above only
logging.getLogger("kafka").setLevel(logging.ERROR)
# mute logging from avro to exception and above only
logging.getLogger("avro").setLevel(logging.ERROR)

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

class KafkaConsumerManager:
    thread:threading.Thread
    def __init__(self, topic:[str, list[str]], bootstrap_servers:str, schema):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True, name='KafkaConsumerManager')
        self.topic = topic
        self.schema = schema
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=bootstrap_servers
        )
        self.thread.start()


    def run(self):
        logger.debug('Starting Kafka consumer thread')
        for message in self.consumer:
            message_data = io.BytesIO(message.value)
            message_data.seek(0)

            logger.debug(f"Received message: {str(message_data)}")

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
        topic = 'camera_object_detection_results'
        schema = '''{
                "fields": [
                    {
                    "name": "frame_id",
                    "type": "string"
                    },
                    {
                    "name": "camera_name",
                    "type": "string"
                    },
                    {
                    "name": "jpg",
                    "type": "bytes"
                    },
                    {
                    "name": "detections",
                    "type": {
                        "items": {
                        "fields": [
                            {
                            "name": "bounding_box",
                            "type": {
                                "fields": [
                                {
                                    "name": "x1",
                                    "type": "int"
                                },
                                {
                                    "name": "y1",
                                    "type": "int"
                                },
                                {
                                    "name": "x2",
                                    "type": "int"
                                },
                                {
                                    "name": "y2",
                                    "type": "int"
                                }
                                ],
                                "name": "BoundingBox",
                                "type": "record"
                            }
                            },
                            {
                            "name": "classification",
                            "type": "string"
                            },
                            {
                            "name": "certainty",
                            "type": "float"
                            }
                        ],
                        "name": "ObjectDetection",
                        "type": "record"
                        },
                        "type": "array"
                    }
                    }
                ],
                "name": "ObjectDetectionResult",
                "namespace": "land.coleman.cameras",
                "type": "record"
                }'''
        bootstrap_servers = os.environ.get('BOOTSTRAP_SERVER', 'localhost:9092')

        # Create and start the Kafka consumer thread
        consumer_manager = KafkaConsumerManager(topic=topic, bootstrap_servers=bootstrap_servers, schema=schema)


        logger.info('waiting for consumer thread to finish. this should never happen')
        consumer_manager.thread.join()


        