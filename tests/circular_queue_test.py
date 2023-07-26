import pytest
from camera.circular_queue import Circular_Queue
from dataclasses import dataclass


def test_require_dataclass():
    # create non-data class
    @dataclass
    class Data:
        make: str
        model: int

    class NotData:
        make: str
        model: str

    with pytest.raises(TypeError) as exc_info:
        notData = NotData('Jeep', 'Cherokee')
        queue = Circular_Queue(notData, 8)
    
    
    assert exc_info.type is TypeError, 'queue did not require a dataclass'

    data = Data('Jeep', 'Cherokee')
    queue = Circular_Queue(data, 8)

    assert isinstance(queue, Circular_Queue), 'failed to create a circular queue'

def test_single_put_get():
    @dataclass
    class Data:
        make: str
        model: int
    data = Data('jeep', 'cherokee')
    queue = Circular_Queue(data, 8)

    queue.put(data)
    d2 = queue.get()

    assert data == d2, 'single put get failed'
