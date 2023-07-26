from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import logging.config
import yaml
import os

from log_config import configure_logging

if __name__ == '__main__': configure_logging()

logger = logging.getLogger(__name__)

_alert_table_path = os.environ.get("DATABASE_LOCATION", "alerts.db")

app = FastAPI()


logger.info('creating database if it does not already exist')
# Create SQLite3 database and tables if they don't exist
conn = sqlite3.connect(_alert_table_path)
cursor = conn.cursor()

# Table for motion events
cursor.execute('''CREATE TABLE IF NOT EXISTS motion_events
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  frame_id TEXT,
                  clip_name TEXT,
                  camera_name TEXT,
                  event_magnitude REAL,
                  image_guid TEXT,
                  UNIQUE(frame_id, clip_name, camera_name, event_magnitude))''')

# Table for object detected events
cursor.execute('''CREATE TABLE IF NOT EXISTS object_detected_events
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  frame_id TEXT,
                  clip_name TEXT,
                  camera_name TEXT,
                  object_detected_json TEXT,
                  image_guid TEXT,
                  UNIQUE(frame_id, clip_name, camera_name, object_detected_json))''')

# Table for image GUIDs and filepaths
cursor.execute('''CREATE TABLE IF NOT EXISTS image_files
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_guid TEXT,
                  filepath TEXT,
                  UNIQUE(image_guid))''')

conn.commit()

logger.info('succesfully created database')

class EventMotionPayload(BaseModel):
    frame_id: str
    clip_name: str
    camera_name: str
    event_magnitude: float
    image_guid: str

@app.post("/event_motion")
def log_event_motion(payload: EventMotionPayload):
    frame_id = payload.frame_id
    clip_name = payload.clip_name
    camera_name = payload.camera_name
    event_magnitude = payload.event_magnitude
    image_guid = payload.image_guid

    conn = sqlite3.connect(_alert_table_path)
    cursor = conn.cursor()
    cursor.execute('''SELECT id FROM motion_events WHERE frame_id = ? AND clip_name = ? AND camera_name = ? AND event_magnitude = ?''',
                   (frame_id, clip_name, camera_name, event_magnitude))
    existing_event = cursor.fetchone()
    if existing_event:
        logger = logging.getLogger("event_motion")
        logger.warning("Event motion already exists")
        return {"message": "Event motion already exists"}
    else:
        cursor.execute('''INSERT INTO motion_events (frame_id, clip_name, camera_name, event_magnitude, image_guid)
                          VALUES (?, ?, ?, ?, ?)''', (frame_id, clip_name, camera_name, event_magnitude, image_guid))
        conn.commit()
        logger = logging.getLogger("event_motion")
        logger.info("Event motion logged successfully")
        return {"message": "Event motion logged successfully"}

class EventObjectDetectedPayload(BaseModel):
    frame_id: str
    clip_name: str
    camera_name: str
    object_detected_json: str
    image_guid: str

@app.post("/event_object_detected")
def log_event_object_detected(payload: EventObjectDetectedPayload):
    frame_id = payload.frame_id
    clip_name = payload.clip_name
    camera_name = payload.camera_name
    object_detected_json = payload.object_detected_json
    image_guid = payload.image_guid

    conn = sqlite3.connect(_alert_table_path)
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT id FROM object_detected_events WHERE frame_id = ? AND clip_name = ? AND camera_name = ? AND object_detected_json = ?''',
        (frame_id, clip_name, camera_name, object_detected_json))
    existing_event = cursor.fetchone()
    if existing_event:
        logger = logging.getLogger("event_object_detected")
        logger.warning("Event object detected already exists")
        return {"message": "Event object detected already exists"}
    else:
        cursor.execute(
            '''INSERT INTO object_detected_events (frame_id, clip_name, camera_name, object_detected_json, image_guid)
              VALUES (?, ?, ?, ?, ?)''',
            (frame_id, clip_name, camera_name, object_detected_json, image_guid))
        conn.commit()
        logger = logging.getLogger("event_object_detected")
        logger.info("Event object detected logged successfully")
        return {"message": "Event object detected logged successfully"}


@app.post("/image_filepath")
def link_image_filepath(image_guid: str, filepath: str):
    conn = sqlite3.connect(_alert_table_path)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO image_files (image_guid, filepath)
                      VALUES (?, ?)''', (image_guid, filepath))
    conn.commit()
    logger = logging.getLogger("image_filepath")
    logger.info("Image filepath linked successfully")
    return {"message": "Image filepath linked successfully"}
class ImageFilepathPayload(BaseModel):
    image_guid: str
    filepath: str


@app.post("/image_filepath")
def link_image_filepath(payload: ImageFilepathPayload):
    image_guid = payload.image_guid
    filepath = payload.filepath

    conn = sqlite3.connect(_alert_table_path)
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO image_files (image_guid, filepath) VALUES (?, ?)''',
        (image_guid, filepath))
    conn.commit()
    logger = logging.getLogger("image_filepath")
    logger.info("Image filepath linked successfully")
    return {"message": "Image filepath linked successfully"}