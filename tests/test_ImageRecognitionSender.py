import asyncio
import datetime
import io
import logging
import numpy as np
import queue
import requests
import threading
import uuid
from queue import Queue
from unittest.mock import Mock, patch
from typing import List, NamedTuple, Tuple
import pytest

from camera.ImageRecognitionSender import ImageRecognitionSender

@pytest.fixture(scope="function")
def image_recognition_sender():
    with ImageRecognitionSender(ip_address='http://localhost:8000') as sender:
        yield sender

def test_Abort_Force():
    with ImageRecognitionSender(ip_address='http://localhost:8000') as sender:
        sender.Abort(force=True)
        assert sender._thread

