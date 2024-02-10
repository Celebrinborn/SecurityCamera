from dataclasses import dataclass
import datetime
import io
import logging
from typing import List

from strategies.message_strategy import MessageStrategy

from avro.io import DatumReader
import io
from avro.datafile import DataFileReader

from kafka.consumer.fetcher import ConsumerRecord

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class ObjectDetection:
    bounding_box: BoundingBox
    classification: str
    certainty: float

@dataclass
class ObjectDetectionResult:
    frame_id: str
    camera_name: str
    detections: List[ObjectDetection]
    frame_jpg: bytes



class PersonDetectedStrategy(MessageStrategy):
    
    @staticmethod
    def get_html_template()->str:
        return f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        color: #333;
    }}
    .container {{
        width: 80%;
        margin: 20px auto;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 7px;
        background-color: #f9f9f9;
    }}
    h1 {{
        color: #444;
    }}
    .alert-image {{
        width: 100%;
        max-width: 600px;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>Security Alert: Person Detected</h1>
    <p>A person has been detected by your security camera. Please review the image and take appropriate action.</p>
    <img src="cid:image0" alt="Security Alert Image" class="alert-image">
    <p>Time of Detection: {datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%A, %B %d, %Y at %I:%M %p')}</p>
    <p>If you recognize this activity as normal, no further action is needed. If not, please check your premises or contact authorities.</p>
</div>
</body>
</html>
"""
    
    

    
    last_email_sent:dict[str, datetime.datetime] = {}

    @staticmethod
    def _deserialize_object_detection_message(message:ConsumerRecord) -> ObjectDetectionResult:
        if not message.topic == 'camera_object_detection_results':
            raise ValueError(f"Message name must be 'camera_object_detection_results', not {message.name}")
        message_data = io.BytesIO(message.value)
        message_data.seek(0)

        # logger.debug(f"Received message: {str(message_data)}")

        # Deserialize using Avro
        avro_reader = DataFileReader(message_data, DatumReader())
        for record in avro_reader:
            detection_result = ObjectDetectionResult(
                frame_id=record['frame_id'],
                camera_name=record['camera_name'],
                frame_jpg=io.BytesIO(record['frame_jpg']),
                detections=[
                    ObjectDetection(
                        bounding_box=BoundingBox(
                            x1=detection['bounding_box']['x1'],
                            y1=detection['bounding_box']['y1'],
                            x2=detection['bounding_box']['x2'],
                            y2=detection['bounding_box']['y2']
                        ),
                        classification=detection['classification'],
                        certainty=detection['certainty']
                    )
                    for detection in record['detections']
                ]
            )
        avro_reader.close()
        return detection_result

    def on_message(self, message):
        logger.info(f"Person Detected: {message}")
        objectDetectionResult = self._deserialize_object_detection_message(message)

        # Check if we've already sent an email for this camera in the last 5 minutes
        camera = objectDetectionResult.camera_name
        if camera in self.last_email_sent:
            if (datetime.datetime.now() - self.last_email_sent[camera]) < datetime.timedelta(minutes=5):
                logger.info(f"Already sent email for {camera} in last 5 minutes. Skipping.")
                return
        self.last_email_sent[camera] = datetime.datetime.now()

        # Send email
        for sender in self._senders:
            sender.Send(
                subject=f"Security Alert: Person Detected by {objectDetectionResult.camera_name}",
                content=self.get_html_template(),
                attachments=[objectDetectionResult.frame_jpg]
            )


    def _use_strategy(self, message:ConsumerRecord) -> bool:
        assert isinstance(message, ConsumerRecord), f"message must be of type ConsumerRecord, not {type(message)}"
        if not message.topic == 'camera_object_detection_results':
            return False
        objectDetectionResult = self._deserialize_object_detection_message(message)
        if len(objectDetectionResult.detections) == 0:
            return False
        
        person_detections = [detection for detection in objectDetectionResult.detections if detection.classification == 'person']
        
        return len(person_detections) > 0
        
        
