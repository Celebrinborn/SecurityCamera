import logging

from src.strategies.message_strategy import MessageStrategy



logger = logging.getLogger(__name__)

class PersonDetectedStrategy(MessageStrategy):
    def on_message(self, message):
        if message['type'] == 'Person Detected':
            # Logic for Person Detected message
            pass

    def _use_strategy(self, message):
        return message['type'] == 'Person Detected'
