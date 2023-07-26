from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import threading
import queue
import numpy as np
import requests
from typing import Optional
from detect import ObjectDetector
from uuid import UUID
import base64
import cv2

import json

import logging
from log_config import configure_logging

# from database_manager import DatabaseManager

# Configure logging
configure_logging()
logger = logging.getLogger()

# Define the priority queue
priority_queue = queue.PriorityQueue()

# Create an instance of the FastAPI class
app = FastAPI()

# Load the ObjectDetector instance into memory
object_detector = ObjectDetector()

# Define a model for the request body
class DetectionRequest(BaseModel):
    priority: int
    camera_name: str
    image_guid: str
    timestamp: int
    timeout: int
    frame: str  # Frame will be received as a base64 encoded string

@app.on_event("startup")
async def startup_event():
    """Start a daemon thread that will consume items from the priority queue"""
    logger.info("Starting daemon thread to consume items from priority queue")
    thread = threading.Thread(target=consume_queue, daemon=True)
    thread.start()
@app.get("/recorded_guids")
async def get_recorded_guids():
    """Endpoint to retrieve a comma-delimited list of recorded GUIDs"""
    db_name = os.path.join('data', 'object_detection.sqlite3')
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)

        # Create a cursor
        cur = conn.cursor()

        # Execute the query to fetch all recorded GUIDs
        cur.execute("SELECT image_id FROM detection_results")

        # Fetch all records
        results = cur.fetchall()

        # Extract the GUIDs and join them into a comma-delimited string
        recorded_guids = ",".join([record[0] for record in results])

        return recorded_guids

    except sqlite3.Error as e:
        logger.error(f"An error occurred while fetching recorded GUIDs: {e}")

    finally:
        conn.close()


@app.post("/detect_objects", status_code=202)
async def detect_objects(request: DetectionRequest):
    """Add detection requests to the priority queue"""
    # Validate input
    try:
        # Try to decode the frame
        jpg_original = base64.b64decode(request.frame)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid frame. Must be a valid base64 encoded image.")
    try:
        UUID(request.image_guid, version=4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_guid. Must be a valid UUID.")

    db_name = os.path.join('data', 'object_detection.sqlite3')
    _directory = os.path.dirname(db_name)
    if not os.path.exists(_directory):
        logger.info(f'creating directory: {_directory}')
        os.makedirs(_directory)
        
    # check if image has been checked yet
    try:
        # connect
        conn = sqlite3.connect(db_name)

        # check if table exists
        if not _table_exists(conn, 'detection_results'):
            logger.info(f'detection_results directory does not exist, creating...')
            _create_object_detection_table(conn)
        if not image_id_exists(conn, request.image_guid):
            logger.warning(f'duplicate request, please see {request.image_guid}')
            raise HTTPException(status_code=409, detail='duplicate request. Image already processed')
        
    finally:
        conn.close()


    request.frame = frame
    priority_queue.put((request.priority, request))
    logger.info(f"Request {request.image_guid} added to the priority queue")
    return {"message": "Request added to the priority queue."}

def consume_queue():
    """Consume items from the priority queue and run them through the ObjectDetector's detect function"""
    while True:
        if not priority_queue.empty():
            priority, request = priority_queue.get()
            try:
                # Perform the object detection
                logger.info(f"Processing request {request.image_guid}")
                result = object_detector.detect(request.frame, request.image_guid)

                # Send the results
                send_results(result)
                logger.info(f"Successfully processed and sent results for request {request.image_guid}")
            except ValueError as ve:
                # Handle cases where the object detection fails because of a bad input
                logger.error(f"Bad input for request {request.image_guid}: {str(ve)}")
            except requests.exceptions.RequestException as re:
                # Handle cases where sending the results fails because of a network issue
                logger.error(f"Failed to send results for request {request.image_guid}: {str(re)}")
            except Exception as e:
                # Log any other failures
                logger.exception(f"Unexpected error for request {request.image_guid}: {str(e)}", stack_info=True, exc_info=True)
                raise e




import sqlite3
import os

import logging
logger = logging.getLogger(__name__)

def image_id_exists(conn, image_id):
    # Create a cursor
    cur = conn.cursor()

    # Execute the query
    cur.execute("SELECT 1 FROM detection_results WHERE image_id = ?", (image_id,))

    # Fetch one record
    result = cur.fetchone()
    
    logger.debug(f'result is: {result}')

    # If a record is found, return True, else return False
    _is_in_db = result is None
    logger.debug(f'was record found {_is_in_db}')
    return _is_in_db


def send_results(results):
    """Send the results out via a network request using the requests library"""
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary")
    if 'annotations' not in results:
        raise ValueError("results must have a key 'annotations'")
    if not isinstance(results['annotations'], list):
        raise TypeError("results['annotations'] must be a list")
    for annotation in results['annotations']:
        if not isinstance(annotation, dict):
            raise TypeError("Each annotation must be a dictionary")
        required_keys = ['id', 'image_id', 'category_id', 'category_name', 'bbox', 'area', 'confidence']
        if not all(key in annotation for key in required_keys):
            raise ValueError("Each annotation must have keys 'id', 'image_id', 'category_id', 'category_name', 'bbox', 'area', and 'confidence'")
        if not isinstance(annotation['bbox'], list):
            raise TypeError("bbox must be a list")
        # fix me later. this should be true, instead its returning tensors
        # if not all(isinstance(b, (float, int)) for b in annotation['bbox']): #TODO: I need to fix this. it returns tensors instead of floats
        #     raise TypeError("bbox must contain only numbers")
        if not isinstance(annotation['image_id'], str):
            raise TypeError("image_id must be a string")
        if not isinstance(annotation['category_id'], int):
            raise TypeError("category_id must be an integer")
        if not isinstance(annotation['category_name'], str):
            raise TypeError("category_name must be a string")
        # if not isinstance(annotation['area'], (float, int)): #TODO: I need to fix this logic
            # raise TypeError("area must be a number")
        if not isinstance(annotation['confidence'], float):
            raise TypeError("confidence must be a float")


    db_name = os.path.join('data', 'object_detection.sqlite3')
    _directory = os.path.dirname(db_name)
    if not os.path.exists(_directory):
        logger.info(f'creating directory: {_directory}')
        os.makedirs(_directory)
        
    try:
        # connect
        conn = sqlite3.connect(db_name)

        # check if table exists
        if not _table_exists(conn, 'detection_results'):
            logger.info(f'detection_results directory does not exist, creating...')
            _create_object_detection_table(conn)
        
        # save results
        logger.debug(f'saving results to db: id: {str(results)}')
        save_results_to_db(conn, results)
    except sqlite3.Error as e:
        logger.error(f'an error occured with sqlite3 while attempting to save detection_results {e}', exc_info=True, stack_info=True)

    finally:
        conn.close()

def _table_exists(conn, table_name):
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    return result is not None

def save_results_to_db(conn, results):
    sql = '''INSERT INTO detection_results(image_id, category_id, category_name, bbox, area, confidence)
                VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()

    # Iterate over each annotation and insert into the database
    for annotation in results['annotations']:
        try:
            bbox = ",".join(str(b.item()) for b in annotation['bbox'])  # Convert tensor to string
            data = (annotation['image_id'], annotation['category_id'], annotation['category_name'], bbox, annotation['area'].item(), annotation['confidence'])
            cur.execute(sql, data)
        except sqlite3.Error as e:
            logger.error(f"Failed to insert data into the database: {e}")
            continue  # Skip this annotation and try the next one

    try:
        conn.commit()
    except sqlite3.Error as e:
        logger.exception(f"Failed to commit changes to the database: {e}")


def _create_object_detection_table(conn):
    try:
        sql = '''CREATE TABLE IF NOT EXISTS detection_results (
                    id INTEGER PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    category_name TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    area REAL NOT NULL,
                    confidence REAL NOT NULL
                    );'''
        c = conn.cursor()
        c.execute(sql)
    except sqlite3.Error as e:
        logger.exception(e)