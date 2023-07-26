import os
import tempfile
import pytest
from fastapi.testclient import TestClient

# Import your FastAPI application
from alerts.app import app

# Create a temporary database for testing
@pytest.fixture(scope="session")
def test_database():
    # Create a temporary file for the SQLite database
    temp_file = tempfile.NamedTemporaryFile(delete=True)

    # Get the path to the temporary file
    database_location = temp_file.name

    # Set the environment variable for the test database location
    os.environ["DATABASE_LOCATION"] = database_location

    # Yield the database location
    yield database_location

    # Clean up the temporary database file after the tests
    del os.environ["DATABASE_LOCATION"]

# Set up the test client
@pytest.fixture(scope="session")
def client():
    # Use the TestClient with the FastAPI application
    with TestClient(app) as client:
        yield client

# Test case for the /event_motion endpoint
def test_event_motion(client, test_database):
    response = client.post("/event_motion", json={
        "frame_id": "123",
        "clip_name": "video.mp4",
        "camera_name": "camera",
        "event_magnitude": 1.23,
        "image_guid": "image123"
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Event motion logged successfully"}

# Test case for the /event_object_detected endpoint
def test_event_object_detected(client, test_database):
    response = client.post("/event_object_detected", json={
        "frame_id": "456",
        "clip_name": "video.mp4",
        "camera_name": "camera",
        "object_detected_json": "{}",
        "image_guid": "image456"
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Event object detected logged successfully"}

# Test case for the /image_filepath endpoint
def test_image_filepath(client, test_database):
    response = client.post("/image_filepath", json={
        "image_guid": "image123",
        "filepath": "/path/to/image.png"
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Image filepath linked successfully"}
