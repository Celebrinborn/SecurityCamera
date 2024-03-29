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

# reduce the amount of logging from avro
logging.getLogger('avro').setLevel(logging.ERROR)

class KafkaManager:
    _instance = None  # Class-level variable to hold the singleton instance
    _lock = threading.Lock()
    _producer: KafkaProducer
    bootstrap_servers: str

    def __new__(cls, bootstrap_servers=None):
        # If an instance already exists, return it
        if cls._instance is not None:
            # logger.debug(f'SQLManager already exists, returning existing instance to caller {inspect.stack()[1].function}')
            return cls._instance
        
        # If no instance exists, create a new one and store it in _instance
        cls._instance = super().__new__(cls)
        logger.debug('Kafka Manager does not exist, creating new instance')
        return cls._instance

    def connect(self) -> bool:
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers
            )
        except KafkaError as e:
            logger.error(f'Error connecting to Kafka: {e}')
            logger.error(f'bootstrap_servers: {self.bootstrap_servers}')
            logger.error(f'KAFKA_BOOTSTRAP_SERVER environment variable: {os.environ.get("KAFKA_BOOTSTRAP_SERVER", "not set")}')
            
            logger.critical('Unable to connect to Kafka, camera will be unable to send messages to the rest of the system.')
            # raise e #TODO: Need to account for this error more smartly, right now I just crash
            return False
        return True
    def __init__(self, bootstrap_servers=None):
        # If an instance already exists, return it
        if hasattr(self, '_producer') and self._producer:
            return

        if bootstrap_servers is None:
            bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVER', 'localhost:9092')
        self.bootstrap_servers = bootstrap_servers

        self.connect()

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
        if not hasattr(self, '_producer'):
            logger.error('Kafka producer not available')
            return
        try:
            future = self._producer.send(topic, bytes(value, 'utf-8'))
            return future
        except KafkaError as e:
            logger.error(f'Error sending message to topic {topic}: {e}')
            #TODO: Need to account for this error more smartly, right now I just ignore it
    def send_motion_alert(self, frame: Frame, camera_name: str, priority: float, motion_amount: float, timeout: int = 60*5):
        # check if self has producer
        if not hasattr(self, '_producer'):
            logger.error('Kafka producer not available')
            return
        topic = 'camera_motion_threshold_exceeded'
        frame_data = {
            "camera_name": camera_name,
            "priority": priority,
            "guid": str(frame.guid),
            "creation_timestamp": frame.creation_timestamp,
            "frame_jpg": frame.scale(height=480, width=640).Export_To_JPG(), #TODO: This is a hack, if filesize is too big avro fails silently
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
