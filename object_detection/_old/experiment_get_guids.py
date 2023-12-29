import cv2
import base64
import json
import requests

response = requests.get("http://127.0.0.1:6666/recorded_guids", headers={'Content-Type': 'application/json'})

# Print the response
print(response.text)
