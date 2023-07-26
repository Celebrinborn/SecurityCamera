import pandas as pd
import numpy as np
from dataclasses import dataclass, fields, is_dataclass
import time

class Circular_Queue:
    '''
    fifo queue
    head is where the queue is read from
    tail is where the queue is written to
    '''
    _max_size: int
    _queue:pd.DataFrame
    front:int # where 
    rear:int
    _item_type:type
    def __init__(self, item, max_size=1024):
        
        if not is_dataclass(item):
            raise TypeError(f'{item=} must be a dataclass')
        self._max_size=max_size
        self.front=-1
        self.rear=-1
        self._item_type = type(item)
        
        annotations = item.__annotations__
        # Create an empty DataFrame with columns and types corresponding to the dataclass fields
        column_names = list(annotations.keys())
        column_types = list(annotations.values())
        
        _df_constructor = {}

        for column_name in annotations.keys():
            _dtype = None
            # Check if the annotation is a recognized data type
            if annotations[column_name] == str:
                _dtype = 'object'
            elif annotations[column_name] == int:
                _dtype = 'Int64'  # Nullable integer
            elif annotations[column_name] == bool:
                _dtype = 'boolean'
            elif annotations[column_name] == float:
                _dtype = 'Float64'  # Nullable float
            else:
                _dtype = 'object'
            _df_constructor[column_name] = pd.Series(dtype=_dtype)
        
        self._queue = pd.DataFrame(_df_constructor)
        
        # initiate values as empty
        for i in range(self._max_size):
            self._queue.loc[i] = np.nan

    def put(self, item:dataclass):
        if type(item) != self._item_type:
            raise TypeError(f'attempted to inject {type(item)=} however {self._item_type} expected')
        self.rear += 1
        if self.rear == self._max_size:
            self.rear = 0
        if self.front == -1:
            self.front = 0
        
        annotations = item.__annotations__
        data = [getattr(test_class, k) for k in annotations.keys()]
        index = [k for k in annotations.keys()]

        row = pd.Series(data=data, index=index)

        self._queue.loc[self.rear] = row

    def get(self):
        if self.front == -1:
            raise NotImplementedError('dealing with an empty queue is not yet dealt with')
        row = self._queue.loc[self.front]
        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            front += 1
            if front == self._max_size:
                front = 0
        return row
    
    def peek(self):
        if self.front == -1:
            raise NotImplementedError('dealing with an empty queue is not yet dealt with')
        row = self._queue.loc[self.front]
        return row
    


if __name__ == '__main__':
    
    import uuid
    @dataclass
    class TestClass:
        guid:uuid.UUID
        creation_time:float
        filename:str
    test_class = TestClass(uuid.uuid4(), time.time(), 'test.avi')
    queue = Circular_Queue(test_class, 4)
    queue.put(test_class)

    print(queue._queue.head())
    print(queue._queue.tail())