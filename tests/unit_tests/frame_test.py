from camera.frame import Frame
import pytest
import numpy as np
import uuid
import torch
import time
import io
import warnings

@pytest.fixture(scope='module')
def GetImage():
    image = np.random.randint(0, 256, (480, 640, 3), dtype = np.uint8)
    yield image

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
    frame = GetFrame
    assert isinstance(frame, Frame), "Object is not an instance of Frame"
    assert isinstance(frame.guid, uuid.UUID), "GUID is not a UUID instance"

def test_frame_slicing(GetFrame, GetImage):
    frame = GetFrame
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
    frame = GetFrame
    assert np.array_equal(frame, image), f'verify np.array_equal failed: {np.array_equal(frame, image)=}'
    assert (frame == image).all(), f'verify frame == ndarray failed: {(frame == image).all()=}'
    assert not (frame != image).all(), f'verify frame != ndarray failed: {(frame != image).all()=}'

def test_frame_hash(GetFrame):
    frame = GetFrame
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

def test_frame_to_pytorch(GetImage, GetFrame):
    image = GetImage
    frame = GetFrame
    tensor_from_image = torch.from_numpy(image)
    tensor_from_frame = torch.from_numpy(frame)
    assert type(tensor_from_image) == type(tensor_from_frame), "Type mismatch between tensors"
    assert tensor_from_image.shape == tensor_from_frame.shape, "Shape mismatch between tensors"
    assert torch.all(tensor_from_image.eq(tensor_from_frame)), "Data mismatch between tensors"

def test_save_and_load_frame(GetFrame, GetImage):
    frame = GetFrame
    
    json_frame = frame.Save_To_JSON()
    
    loaded_frame = Frame.Load_From_JSON(json_frame)

    assert frame.guid == loaded_frame.guid
    assert frame.creation_timestamp == loaded_frame.creation_timestamp
    assert frame == loaded_frame, 'frame lost after saving and loading'

def test_preserve_identity_with(GetFrame):
    import cv2  # Assuming OpenCV is available for this test

    # Get the original frame from the fixture
    frame = GetFrame

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
