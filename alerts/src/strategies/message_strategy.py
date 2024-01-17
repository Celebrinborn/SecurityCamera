import logging

from abc import ABC, abstractmethod
import queue
import threading

logger = logging.getLogger(__name__)

class MessageStrategy(ABC):
    _thread:threading.Thread
    def __init__(self):
        self.message_queue = queue.Queue()

    def get_thread(self) -> threading.Thread:
        return self._thread

    @abstractmethod
    def on_message(self, message):
        raise NotImplementedError

    def start_processing(self):
        self._thread = threading.Thread(target=self._process_messages, name=f'{self.__class__.__name__}_strategy', daemon=True)
        self._thread.start()

    def _process_messages(self):
        while True:
            message = self.message_queue.get()
            if self._use_strategy(message):
                self.on_message(message)

    def _use_strategy(self, message):
        return False  # Default implementation
