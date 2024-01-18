import io
import logging

from abc import ABC, abstractmethod
from typing import Optional, List


logger = logging.getLogger(__name__)

class Sender(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def Send(subject:str, content:str, attachments:Optional[List[io.BytesIO]]):
        raise NotImplementedError