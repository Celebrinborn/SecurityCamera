import logging

from abc import ABC, abstractmethod
import queue
import threading
from typing import List

from senders.message_sender import Sender

from kafka.consumer.fetcher import ConsumerRecord

logger = logging.getLogger(__name__)

class MessageStrategy(ABC):
    _thread:threading.Thread
    _message_queue:queue.Queue[ConsumerRecord]
    _senders:List[Sender]
    def __init__(self, message_queue:queue.Queue, senders:List[Sender]):
        self._message_queue = queue.Queue()
        self._senders = senders
        self._message_queue = message_queue

    def get_thread(self) -> threading.Thread:
        return self._thread

    @abstractmethod
    def on_message(self, message):
        raise NotImplementedError

    def start_processing(self):
        logger.info(f'Starting {self.__class__.__name__} strategy')
        self._thread = threading.Thread(target=self._process_messages, name=f'{self.__class__.__name__}_strategy', daemon=True)
        self._thread.start()

    def _process_messages(self):
        while True:
            message = self._message_queue.get()
            if self._use_strategy(message):
                self.on_message(message)

    def _use_strategy(self, message) -> bool:
        return False  # Default implementation
