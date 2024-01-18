import datetime
import io
import logging
import os

from log_config import configure_logging

from typing import List, Optional
from dataclasses import dataclass
from queue import Queue

import threading
from senders.message_sender import Sender
from senders.SMTPSender import SMTPSender

from strategies.message_strategy import MessageStrategy
from strategies.person_detected_strategy import PersonDetectedStrategy



from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord



from pathlib import Path

configure_logging()
logger = logging.getLogger(__name__)

# Load environment variables from .env file
if Path('.env').exists():
    import dotenv
    logger.info('Loading environment variables from .env file')
    dotenv.load_dotenv('.env')
    logger.info(f'Environment variables loaded from .env file: {", ".join([key for key, value in os.environ.items()])}')

# Load environment variables from Docker secrets if running in Docker
secrets_dir = Path('/run/secrets/')
if secrets_dir.exists():
    for secret_file in secrets_dir.iterdir():
        if secret_file.is_file():
            # Read the content of each secret file
            secret = secret_file.read_text().strip()
            # Set the content as an environment variable
            os.environ[secret_file.name] = secret

# mute logging from kafka to exception and above only
logging.getLogger("kafka").setLevel(logging.ERROR)
# mute logging from avro to exception and above only
logging.getLogger("avro").setLevel(logging.ERROR)


class KafkaConsumerManager:
    thread:threading.Thread
    message_queue:Queue
    def __init__(self, queue:Queue, topic:[str, list[str]], bootstrap_servers:str, schema):
        self.message_queue = queue
        self.thread = threading.Thread(target=self._run, args=(), daemon=True, name='KafkaConsumerManager')
        self.topic = topic
        self.schema = schema
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=bootstrap_servers
        )


    def _run(self):
        logger.debug('Starting Kafka consumer thread')
        for message in self.consumer:
            self.message_queue.put(message)
            
            

    def stop(self):
        self.consumer.close()


if __name__ == '__main__':
    logger.info('Starting application')


    logger.debug(f'list of env vars: {", ".join([key for key, value in os.environ.items()])}')

    message_queue:Queue[ConsumerRecord] = Queue()

    # configure strategies
    strategies:List[MessageStrategy] = [PersonDetectedStrategy(message_queue = message_queue, senders=[SMTPSender()])]
    for strategy in strategies:
        strategy.start_processing()   

    # Create and start the Kafka consumer thread
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
    consumer_manager = KafkaConsumerManager(queue=message_queue, topic=topic, bootstrap_servers=bootstrap_servers, schema=schema)
    consumer_manager.thread.start()

    logger.info('application fully running, press Ctrl+C to exit')
    consumer_manager.thread.join()


    