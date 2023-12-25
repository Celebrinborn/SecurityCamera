from pathlib import Path
import numpy as np
import warnings
import uuid
import time
import io
import json
import base64
from typing import Optional, Union, Any

import avro.schema
from avro.io import DatumWriter
from avro.io import DatumReader
from avro.datafile import DataFileReader, DataFileWriter
from io import BytesIO

import os

import cv2

import logging
logger = logging.getLogger(__name__)


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
    
    def Export_To_JSON(self):
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

    def Export_To_JPG(self) -> bytes:
        """
        Convert a Frame object to a JPEG byte string.

        :param frame: Frame object to convert.
        :return: JPEG byte string.
        """
        # Encode the frame as JPEG
        success, encoded_image = cv2.imencode('.jpg', self)
        if not success:
            raise ValueError("Could not encode image to JPEG")
        return encoded_image.tobytes()

    @staticmethod
    def _load_From_JPG(jpg_bytes: bytes) -> np.ndarray:
        """
        Convert a JPEG byte string to an image array.

        :param jpg_bytes: JPEG byte string.
        :return: Image array.
        """
        # Decode the JPEG byte string to an image array
        image_array = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image_array

    @staticmethod
    def Load_From_JPG(jpg_bytes: bytes, guid: uuid.UUID, creation_timestamp: float) -> 'Frame':
        """
        Convert a JPEG byte string back to a Frame object.

        :param jpg_bytes: JPEG byte string.
        :param guid: Optional UUID for the frame.
        :param creation_timestamp: Optional timestamp for the frame.
        :return: Frame object.
        """
        # Decode the JPEG byte string to an image array
        image_array = Frame._load_From_JPG(jpg_bytes)

        if guid is None:
            guid = uuid.uuid4()
        if creation_timestamp is None:
            creation_timestamp = time.time()

        return Frame(image_array, GUID=guid, creation_timestamp=creation_timestamp)

    @DeprecationWarning
    @classmethod
    def _cache_avro_schema(cls) -> None:
        """
        Cache the Avro schema in memory.
        """
        schema_path = Path('data', 'avro_schemas', 'frame.avsc')
        if not schema_path.exists():
            logger.error(f"Schema file {schema_path} does not exist")
            raise Exception(f"Schema file {schema_path} does not exist")
        
        with open(schema_path, 'r') as file:
            schema_str = file.read()
        
        schema = avro.schema.parse(schema_str)

        
        cls._avro_cached_schema = schema
    
    @DeprecationWarning
    def serialize_avro(self) -> bytes:
        # raise NotImplementedError("This method is not implemented yet")
        """
        Serialize the Frame object to Avro format in memory.
        :return: BytesIO object containing Avro-encoded data.
        """

        if not hasattr(self, '_avro_cached_schema'):
            Frame._cache_avro_schema()
        schema = self._avro_cached_schema

        frame_data = {
            "guid": str(self.guid),
            "creation_timestamp": self.creation_timestamp,
            "data": self.Export_To_JPG()
        }

        bytes_writer = BytesIO()
        writer = DataFileWriter(bytes_writer, DatumWriter(), schema)
        writer.append(frame_data)
        writer.flush()
        bytes_writer.seek(0)

        return bytes_writer.getvalue()

    @staticmethod
    def deserialize_avro(avro_bytes: bytes) -> 'Frame':
        """
        Deserialize Avro data from an in-memory buffer.
        :param avro_buffer: BytesIO object containing Avro-encoded data.
        :return: Frame object.
        """
        avro_buffer = io.BytesIO(avro_bytes)
        avro_buffer.seek(0)
        reader: DataFileReader = DataFileReader(avro_buffer, DatumReader())
        try:
            frame_data: dict[str, Any] = next(reader) # type: ignore
        except StopIteration:
            logger.error("No data in Avro buffer")
            raise Exception("No data in Avro buffer")
        finally:
            reader.close()

        # verify that the needed attributes are present
        if 'guid' not in frame_data:
            logger.error("GUID not present in Avro buffer")
        if 'creation_timestamp' not in frame_data:
            logger.error("creation_timestamp not present in Avro buffer")
        if 'data' not in frame_data:
            logger.error("data not present in Avro buffer")
        return Frame.Load_From_JPG(frame_data['data'], guid=uuid.UUID(frame_data['guid']), creation_timestamp=frame_data['creation_timestamp'])