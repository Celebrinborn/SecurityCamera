import logging
import threading
from typing import Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json

from camera.frame import Frame

import os

logger = logging.getLogger(__name__)

# reduce the amount of logging from kafka
logging.getLogger('kafka').setLevel(logging.ERROR)

class KafkaManager:
    _instance = None  # Class-level variable to hold the singleton instance
    _lock = threading.Lock()
    _producer: KafkaProducer

    def __new__(cls, bootstrap_servers=None):
        # If an instance already exists, return it
        if cls._instance is not None:
            # logger.debug(f'SQLManager already exists, returning existing instance to caller {inspect.stack()[1].function}')
            return cls._instance
        
        # If no instance exists, create a new one and store it in _instance
        cls._instance = super().__new__(cls)
        logger.debug('SQLManager does not exist, creating new instance')
        return cls._instance

    def __init__(self, bootstrap_servers=None):
        # If an instance already exists, return it
        if hasattr(self, '_producer') and self._producer:
            return

        # get bootstrap_servers from env variable if not passed as argument
        if bootstrap_servers is None:
            bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVER', None)
            # if env variable is not set, log a warning and use default value
            if bootstrap_servers is None:
                logger.warning('KAFKA_BOOTSTRAP_SERVER environment variable not set, using default value: localhost:9092')
                bootstrap_servers = 'localhost:9092'

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers
            )
        except KafkaError as e:
            logger.error(f'Error connecting to Kafka: {e}')
            raise


    def send_message(self, topic, value:str):
        try:
            future = self._producer.send(topic, bytes(value, 'utf-8'))
            return future
        except KafkaError as e:
            logger.error(f'Error sending message to topic {topic}: {e}')
            #TODO: Need to account for this error more smartly, right now I just ignore it
    def send_frame(self, topic, frame: Frame):
        """
        Serializes a Frame object and sends it to the specified Kafka topic.

        :param topic: The Kafka topic to which the frame will be sent.
        :param frame: The Frame object to send.
        """
        try:
            # Serialize the Frame object. Choose either JSON or Avro based on your preference.
            serialized_frame:bytes = frame.serialize_avro()  # or frame.Save_To_JSON()
            
            # Send the serialized frame to Kafka
            future = self._producer.send(topic, serialized_frame)
            return future
        except KafkaError as e:
            logger.error(f'Error sending frame to topic {topic}: {e}')
            #TODO: Need to account for this error more smartly, right now I just ignore it

    def flush(self):
        self._producer.flush()

    def close(self):
        self._producer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
