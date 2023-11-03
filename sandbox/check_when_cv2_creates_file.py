import cv2
import pandas as pd
from pathlib import Path
import numpy as np

# Define the output file path and name
output_file = Path('data', 'sandbox', 'test.mp4')

# delete file if it exists
output_file.unlink(missing_ok=True)

# Create a cv2 VideoWriter object with the output file path, codec, frame rate, and frame size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
_fps, _resolution = (10, (640, 480))
video_filename_str = str(output_file)
print(Path(video_filename_str).exists(), 'before video creation')


videowriter = cv2.VideoWriter(video_filename_str, fourcc, _fps, _resolution) # type: ignore


# Check if the file exists using pathlib.Path
print(Path(video_filename_str).exists(), 'after video creation')

#  create a random np array of the frame size
frame = np.random.randint(0, 255, (_resolution[1], _resolution[0], 3), dtype=np.uint8)


# Write the DataFrame to the output file using the cv2 VideoWriter object
videowriter.write(frame)

# Check if the file exists using pathlib.Path
print(Path(output_file).exists(), 'after writing first frame')
    
# Close the cv2 VideoWriter object
videowriter.release()

# Check if the file exists using pathlib.Path
if Path(output_file).exists():
    print("File was closed successfully")
