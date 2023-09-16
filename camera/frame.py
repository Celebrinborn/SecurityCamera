import numpy as np
import warnings
import uuid
import time
import io
import json
import base64
from typing import Optional, Union

class Frame(np.ndarray):
    guid: uuid.UUID
    creation_timestamp: float
    def __new__(cls, input_array, *, GUID:Union[uuid.UUID, None] = None, creation_timestamp:Optional[float] = None):
        """
        Overriding this method allows us to provide additional attributes (GUID and timestamp) to our frame data, which are critical for tracking each frame individually across its life cycle and associating it with its creation moment. The expectation here is that the `input_array` argument is a suitable representation of the image data, otherwise the class may not behave as expected.
        """

        # Check the name of the module from which this method is being called
        warnings.warn("Frame objects should only be created in the camera module.")
        # raise Exception("Frame objects should only be created in the camera module.")

        obj = np.asarray(input_array).view(cls)
        obj.guid = uuid.uuid4() if GUID == None else GUID
        obj.creation_timestamp = time.time() if creation_timestamp == None else creation_timestamp
        return obj
    def __array_finalize__(self, obj):
        """
        This method is overridden to ensure the GUID and timestamp are preserved when using numpy operations that produce new objects, such as slicing. This is essential because the default ndarray behavior does not retain additional attributes during these operations. Keep in mind that if a new `Frame` object is created without going through `__new__` (e.g., when slicing), the `guid` and `creation_timestamp` attributes may not be properly preserved.
        """
        if obj is None: return
        self.guid = getattr(obj, 'guid', uuid.uuid4())
        self.creation_timestamp = getattr(obj, 'creation_timestamp', time.time())
        
    def __getitem__(self, index):
        """
        By overriding this method, numpy's indexing behavior is altered to retain the GUID and timestamp when a subset of the Frame object is accessed. This is beneficial for maintaining the identity of the frame even when it's manipulated or accessed as slices. Note that the slice will have the same GUID and timestamp as the original, so it should not be considered as a completely new frame.
        """

        result = super().__getitem__(index)
        if type(result) is Frame:
            result.guid = self.guid
            result.creation_timestamp = self.creation_timestamp
        return result
    
    def __eq__(self, other):
        """
        Overriding this method allows us to modify the equality comparison from the standard numpy array comparison (element-wise) to a comparison of the GUIDs. This is required because we want to treat two frames as equal if they originate from the same frame, not if their content is identical. This might lead to unintuitive results if you expect the `==` operator to perform an element-wise comparison.
        """

        if isinstance(other, Frame):
            return self.guid == other.guid
        else:
            return super().__eq__(other)
    def __ne__(self, other):
        if isinstance(other, Frame):
            return self.guid != other.guid
        else:
            return super().__ne__(other)
    def __hash__(self):
        """
        Overriding this method enables Frame objects to be used in sets and as dictionary keys by providing a hash method. The hash is based on the GUID, reinforcing the individual identity of each frame. However, be cautious when using Frame objects in a context where hash collisions could be problematic, as the probability of a collision, although extremely low, is not zero with UUIDs.
        """

        return hash(self.guid)
    
    def Save_To_JSON(self):
        # Create a buffer
        buffer = io.BytesIO()

        # Save the numpy array to this buffer
        np.save(buffer, self)

        # Create a base64 string from the buffer
        buffer.seek(0)
        ndarray = base64.b64encode(buffer.read()).decode('ascii')

        # Create a JSON object
        json_obj = {
            'ndarray': ndarray,
            'GUID': str(self.guid),
            'creation_timestamp': self.creation_timestamp
        }

        return json.dumps(json_obj)

    @staticmethod
    def Load_From_JSON(json_str: str):
        # Load the JSON object
        json_obj = json.loads(json_str)

        # Get the base64 string data
        data = base64.b64decode(json_obj['ndarray'])

        # Create a buffer from the string data
        buffer = io.BytesIO(data)

        # Load the numpy array from the buffer
        arr = np.load(buffer)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # Create a Frame object with the loaded array and the GUID and timestamp from the JSON
            frame = Frame(arr, GUID=uuid.UUID(json_obj['GUID']), creation_timestamp=float(json_obj['creation_timestamp']))

        return frame
    
    def preserve_identity_with(self, new_array: np.ndarray) -> 'Frame':
        """
        Returns a new Frame object from the provided numpy array with the same guid and creation_timestamp
        as the calling frame.
        :param new_array: The modified numpy array.
        :return: A new Frame object.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Frame(new_array, GUID=self.guid, creation_timestamp=self.creation_timestamp)




