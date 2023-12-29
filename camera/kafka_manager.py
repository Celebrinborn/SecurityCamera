import logging
import threading
from typing import Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json

from avro.io import DatumWriter, BinaryEncoder
from avro.datafile import DataFileReader, DataFileWriter
import avro.schema
import io
from io import BytesIO

from pathlib import Path
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


    @classmethod
    def _cache_avro_schema(cls) -> None:
        """
        Cache the Avro schema in memory.
        """
        schema_path = Path('avro_schemas', 'frame.avsc')
        if not schema_path.exists():
            logger.error(f"Schema file {schema_path} does not exist")
            raise Exception(f"Schema file {schema_path} does not exist")

        with open(schema_path, 'r') as file:
            schema_str = file.read()

        cls._avro_cached_schema = avro.schema.parse(schema_str)

    def send_message(self, topic, value:str):
        try:
            future = self._producer.send(topic, bytes(value, 'utf-8'))
            return future
        except KafkaError as e:
            logger.error(f'Error sending message to topic {topic}: {e}')
            #TODO: Need to account for this error more smartly, right now I just ignore it
    def send_motion_alert(self, frame: Frame, camera_name: str, priority: float, motion_amount: float, timeout: int = 60*5):
        topic = 'camera_motion_threshold_exceeded'
        frame_data = {
            "camera_name": camera_name,
            "priority": priority,
            "guid": str(frame.guid),
            "creation_timestamp": frame.creation_timestamp,
            "frame_jpg": frame.Export_To_JPG(),
            "motion_amount": motion_amount,
            "timeout": timeout
        }

        bytes_writer = BytesIO()
        if not hasattr(self, '_avro_cached_schema'):
            self._cache_avro_schema()
        writer = DataFileWriter(bytes_writer, DatumWriter(), self._avro_cached_schema)
        writer.append(frame_data)
        writer.flush()
        bytes_writer.seek(0)

        try:
            future = self._producer.send(topic, bytes_writer.read())
            return future
        except KafkaError as e:
            logger.error(f'Unable to send camera_motion_threshold_exceeded event : {e}')


        

    def flush(self):
        self._producer.flush()

    def close(self):
        self._producer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
