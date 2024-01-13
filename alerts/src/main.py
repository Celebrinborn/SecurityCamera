import logging
from log_config import configure_logging

from typing import List, Optional

import threading

from src.strategies.message_strategy import MessageStrategy
from src.strategies.person_detected_strategy import PersonDetectedStrategy

from kafka import KafkaConsumer, KafkaProducer

configure_logging()
logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self.strategies:List[MessageStrategy] = [PersonDetectedStrategy()]

    def listen_to_kafka(self):
        # Kafka listening logic
        pass

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
        main.listen_to_kafka()
        