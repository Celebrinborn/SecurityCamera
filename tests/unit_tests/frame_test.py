# helper: run this to test: python -m pytest tests/unit_tests/frame_test.py

from camera.frame import Frame
import pytest
import numpy as np
import uuid
# import torch
import time
import io
import warnings
import cv2

@pytest.fixture(scope='module')
def GetImage():
    image = np.random.randint(0, 256, (480, 640, 3), dtype = np.uint8)
    yield image
@pytest.fixture(scope='module')
def GetWhiteImage():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    yield image

@pytest.fixture(scope='module')
def GetWhiteFrame(GetWhiteImage):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        image = GetWhiteImage
        frame = Frame(image)
    yield frame

@pytest.fixture(scope='module')
def GetFrame(GetImage):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        image = GetImage

        frame = Frame(image)
        yield frame

@pytest.fixture(scope='module')
def GetFrame2():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        image = np.random.randint(0, 256, (480, 640, 3), dtype = np.uint8)
        frame = Frame(image)
        yield frame


def test_frame_creation(GetFrame):
    frame: Frame = GetFrame
    assert isinstance(frame, Frame), "Object is not an instance of Frame"
    assert isinstance(frame.guid, uuid.UUID), "GUID is not a UUID instance"

def test_frame_slicing(GetFrame, GetImage):
    frame: Frame = GetFrame
    image = GetImage
    slice = frame[100:200, 100:200]
    assert isinstance(slice, Frame), "Sliced object is not an instance of Frame"
    assert slice.guid == frame.guid, "GUIDs do not match after slicing"
    assert slice.creation_timestamp == frame.creation_timestamp, "Timestamps do not match after slicing"

def test_frame_equality(GetFrame, GetFrame2):
    frame1 = GetFrame
    frame2 = GetFrame2
    assert frame1 != frame2, "Frames with different GUIDs are being considered equal"
    assert not frame1 == frame2, "Frames with different GUIDs are being considered equal"
    assert frame1 == frame1, "Frame is not equal to itself"

def test_frame_numpy_equality(GetImage, GetFrame):

    image = GetImage
    frame: Frame = GetFrame
    assert np.array_equal(frame, image), f'verify np.array_equal failed: {np.array_equal(frame, image)=}'
    assert (frame == image).all(), f'verify frame == ndarray failed: {(frame == image).all()=}'
    assert not (frame != image).all(), f'verify frame != ndarray failed: {(frame != image).all()=}'

def test_frame_hash(GetFrame):
    frame: Frame = GetFrame
    assert hash(frame) == hash(frame.guid), "Hash of Frame object and its GUID do not match"

def test_frame_warning():
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    with pytest.warns(UserWarning) as record:
        frame = Frame(image)
    assert len(record) == 1, "More than one warning was raised"
    assert str(record[0].message) == "Frame objects should only be created in the camera module.", "Warning message does not match expected"

def test_frame_manual_guid_creation_timestamp(GetImage):
    image = GetImage

    # Create a custom guid and timestamp
    custom_guid = uuid.uuid4()
    custom_timestamp = time.time()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # Create a Frame object with the custom guid and timestamp
        frame = Frame(image, GUID=custom_guid, creation_timestamp=custom_timestamp)

    # Test that the Frame object was created successfully
    assert isinstance(frame, Frame), "Object is not an instance of Frame"

    # Test that the guid and timestamp are correct
    assert frame.guid == custom_guid, f"GUID does not match: {frame.guid} != {custom_guid}"
    assert frame.creation_timestamp == custom_timestamp, f"Timestamp does not match: {frame.creation_timestamp} != {custom_timestamp}"

# I don't know why this test exists, I don't think I use torch anywhere
# def test_frame_to_pytorch(GetImage, GetFrame):
#     image = GetImage
#     frame: Frame = GetFrame
#     tensor_from_image = torch.from_numpy(image)
#     tensor_from_frame = torch.from_numpy(frame)
#     assert type(tensor_from_image) == type(tensor_from_frame), "Type mismatch between tensors"
#     assert tensor_from_image.shape == tensor_from_frame.shape, "Shape mismatch between tensors"
#     assert torch.all(tensor_from_image.eq(tensor_from_frame)), "Data mismatch between tensors"

def test_save_and_load_frame(GetFrame, GetImage):
    frame:Frame = GetFrame
    
    json_frame = frame.Export_To_JSON()
    
    loaded_frame = Frame.Load_From_JSON(json_frame)

    assert frame.guid == loaded_frame.guid
    assert frame.creation_timestamp == loaded_frame.creation_timestamp
    assert frame == loaded_frame, 'frame lost after saving and loading'

def test_preserve_identity_with(GetFrame):
    import cv2  # Assuming OpenCV is available for this test

    # Get the original frame from the fixture
    frame: Frame = GetFrame

    # Apply some transformation using OpenCV (in this case, converting the frame to grayscale)
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the preserve_identity_with method to obtain a new Frame object with the grayscale image
    with pytest.warns(None) as record:  # This expects no warnings
        new_frame = frame.preserve_identity_with(grayscale_image)

    # Check that no warnings were raised
    assert len(record) == 0, "Unexpected warnings raised during transformation"

    # Check that the new_frame is an instance of Frame
    assert isinstance(new_frame, Frame), "Transformed object is not an instance of Frame"

    # Check that the guid and creation_timestamp are preserved
    assert new_frame.guid == frame.guid, "GUIDs do not match after transformation"
    assert new_frame.creation_timestamp == frame.creation_timestamp, "Timestamps do not match after transformation"

    # Optionally, verify the data (to ensure the grayscale conversion worked as expected)
    assert np.array_equal(new_frame, grayscale_image), "Data mismatch after transformation"

def test_frame_avro_serialization_deserialization(GetWhiteFrame): # use GetWhiteFrame to avoid jpeg compression artifacts
    original_frame: Frame = GetWhiteFrame

    # Serialize the frame to Avro format
    avro_data = original_frame.serialize_avro()

    # Deserialize the Avro data back into a Frame object
    deserialized_frame = Frame.deserialize_avro(avro_data)

    # Test that the GUID and creation_timestamp are preserved
    assert original_frame.guid == deserialized_frame.guid, "GUID mismatch after deserialization"
    assert original_frame.creation_timestamp == deserialized_frame.creation_timestamp, "Creation timestamp mismatch after deserialization"

    # Test that the numpy data is preserved
    assert original_frame == deserialized_frame, "Frame mismatch after deserialization"

    # NOTE: THIS TEST WILL FAIL IF THE FRAME IS NOT PURE COLOR DUE TO THE JPEG COMPRESSION
    assert np.array_equal(original_frame, deserialized_frame), "Frame data mismatch after deserialization"


def test_Export_To_JPG(GetWhiteImage, GetWhiteFrame):
    # Create a pure white Frame object

    image = GetWhiteImage
    frame:Frame = GetWhiteFrame

    # Export to JPG
    jpg_bytes = frame.Export_To_JPG()

    # Check if the output is a non-empty bytes object
    assert isinstance(jpg_bytes, bytes), "Output is not a bytes object"
    assert len(jpg_bytes) > 0, "Output is an empty bytes object"

    # Check if the output is a valid JPEG image
    jpg_buffer = np.frombuffer(jpg_bytes, dtype=np.uint8)

    decoded_image:np.ndarray = cv2.imdecode(jpg_buffer, cv2.IMREAD_COLOR)
    # verify image and decoded_image are similar
    np.testing.assert_array_almost_equal(image, decoded_image, err_msg="Loaded array does not match original")


def test_Load_From_JPG(GetWhiteImage, GetWhiteFrame):
    image = GetWhiteImage
    original_frame:Frame = GetWhiteFrame

    jpg_bytes = original_frame.Export_To_JPG()

    # Load from JPG
    loaded_frame = Frame.Load_From_JPG(jpg_bytes, original_frame.guid, original_frame.creation_timestamp)

    # Check if the loaded frame has the same GUID and timestamp
    assert loaded_frame.guid == original_frame.guid
    assert loaded_frame.creation_timestamp == original_frame.creation_timestamp
    # Optionally, check if the image data is similar (JPEG should handle pure white without artifacts)
    np.testing.assert_array_almost_equal(loaded_frame, original_frame, err_msg="Loaded array does not match original")

    assert original_frame == loaded_frame, "Frame mismatch after loading from JPEG"

def test_private_load_from_jpg(GetWhiteImage, GetWhiteFrame):
    # Create a dummy JPEG image (pure white image for simplicity)
    frame:Frame = GetWhiteFrame
    image = GetWhiteImage

    jpg_bytes = frame.Export_To_JPG()

    # verify jpg is valid 
    jpg_buffer = np.frombuffer(jpg_bytes, dtype=np.uint8)

    decoded_image:np.ndarray = cv2.imdecode(jpg_buffer, cv2.IMREAD_COLOR)

    # verify image and decoded_image are similar
    np.testing.assert_array_almost_equal(image, decoded_image, err_msg="Loaded array does not match original")

