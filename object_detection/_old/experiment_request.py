import cv2
import base64
import json
import requests
from camera.frame import Frame

# Path to the image you want to send
image_path = "test_image.jpg"

# Read the image data using OpenCV
cap = cv2.VideoCapture(0)
ret, image = cap.read()
frame = Frame(image)

# Convert the image to a JPEG file in memory
success, encoded_image = cv2.imencode('.jpg', image)
if not success:
    raise ValueError("Could not encode image")

# Encode this memory file in base64
image_base64 = base64.b64encode(encoded_image).decode('utf-8')

# Construct the JSON body
json_body = {
    "priority": 1,
    "camera_name": "Camera1",
    "image_guid": str(frame.guid),
    "timestamp": 1620000000,
    "timeout": 3000,
    "frame": image_base64
}

# Convert the Python dictionary to a JSON string
json_body = json.dumps(json_body)

# Send the request
response = requests.post("http://127.0.0.1:8000/detect_objects", data=json_body, headers={'Content-Type': 'application/json'})

# Print the response
print(response.json())

print(f'guid is {frame.guid}')

cv2.imshow('Captured Frame', image)

cv2.waitKey(0)
cv2.destroyAllWindows()