from flask import Flask, request, jsonify
import numpy as np
import io
import uuid
import pandas as pd
import yaml
import logging
from typing import List, Optional


app = Flask(__name__)

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

allowed_events: List[str] = config["allowed_events"]

def is_valid_datetime(datetime: str) -> bool:
    try:
        pd.to_datetime(datetime)
        return True
    except ValueError:
        return False

@app.route("/api/new_event", methods=["POST"])
def add_event():
    event_type: str = request.form.get("event_type")
    if event_type not in allowed_events:
        logger.error(f"Invalid event type from {request.remote_addr}, payload size: {request.content_length}")
        return "Invalid event type", 400
    
    event_id: str = str(uuid.uuid4())
    datetime: str = request.form.get("datetime")
    if not is_valid_datetime(datetime):
        logger.error(f"Invalid datetime from {request.remote_addr}, payload size: {request.content_length}")
        return "Invalid datetime", 400
    datetime = pd.to_datetime(datetime)
    
    screenshot: Optional[np.ndarray] = None
    if "screenshot" in request.files:
        file = request.files["screenshot"]
        binary_data: bytes = file.read()
        file = io.BytesIO(binary_data)
        screenshot = np.load(file)
        if not isinstance(screenshot, np.ndarray):
            logger.error(f"Invalid file format from {request.remote_addr}, payload size: {request.content_length}")
            return "Invalid file format, must be numpy ndarray", 400
    
    with pd.HDFStore("events.h5") as store:
        if "events" in store:
            events_df = store["events"]
        else:
            events_df = pd.DataFrame(columns=["event_id", "datetime", "event_type", "screenshot"])
        
        events_df = events_df.append({"event_id": event_id, "datetime": datetime, "event_type": event_type, "screenshot": screenshot}, ignore_index=True)
        store["events"] = events_df
    
    logger.info(f"Successful event added from {request.remote_addr}, payload size: {request.content_length}, event type: {event_type}")
    return jsonify({"event_id": event_id})

if __name__ == '__main__':
    app.run(debug=True)
