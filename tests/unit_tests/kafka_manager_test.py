import warnings
import pytest
from unittest.mock import patch, MagicMock
from camera.frame import Frame  # Adjust the import according to your project structure
from camera.kafka_manager import KafkaManager  # Adjust the import according to your project structure
import json
import numpy as np

# Fixture to create a numpy array consumed by the Frame object
@pytest.fixture(scope='module')
def GetImage():
    image = np.random.randint(0, 256, (480, 640, 3), dtype = np.uint8)
    yield image

# Fixture to create a Frame object
@pytest.fixture(scope='module')
def GetFrame(GetImage):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        image = GetImage

        frame = Frame(image)
        yield frame

# Test for sending a simple message
@patch('camera.kafka_manager.KafkaProducer')  # Corrected path
def test_send_message(mock_kafka_producer):
    # Setup
    topic = 'test_topic'
    message = "hello world"

    # Mock the KafkaProducer's send method
    mock_send = MagicMock()
    mock_kafka_producer.return_value.send = mock_send

    # When
    kafka_manager = KafkaManager(bootstrap_servers='localhost:9092')
    kafka_manager.send_message(topic, message)

    # Then
    mock_send.assert_called_once_with(topic, bytes(message, 'utf-8')) # Check if send was called with the correct arguments. You can be more specific based on the serialization format.

# Test for sending a frame
@patch('camera.kafka_manager.KafkaProducer')
def test_send_frame(mock_kafka_producer, GetFrame):
    # Setup
    topic = 'frame_topic'
    frame = GetFrame # Create a Frame object with appropriate parameters

    # Mock the KafkaProducer's send method
    mock_send = MagicMock()
    mock_kafka_producer.return_value.send = mock_send

    # When
    kafka_manager = KafkaManager(bootstrap_servers='localhost:9092')
    kafka_manager.send_frame(topic, frame)

    # Then
    mock_send.assert_called_once()  # Check if send was called. You can be more specific based on the serialization format.

# Verify that creating kafka manager multiple times returns the same instance
@patch('camera.kafka_manager.KafkaProducer') # this mock is actually used - it prevents the KafkaProducer from being created which prevents it from checking if localhost:9092 is available
def test_kafka_manager_singleton(mock_kafka_producer):
    # When
    kafka_manager1 = KafkaManager(bootstrap_servers='localhost:9092')
    kafka_manager2 = KafkaManager(bootstrap_servers='localhost:9092')

    # Then
    assert kafka_manager1 is kafka_manager2, "KafkaManager is not a singleton"

    # verify cls._instance is set
    assert hasattr(KafkaManager, '_instance'), "KafkaManager does not have _instance attribute"
