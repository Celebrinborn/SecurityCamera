import cv2

def display_motion_text_on_webcam():
    # Open a connection to the first webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    motion_string = "Motion Detected"

    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            # If the frame was read successfully
            if ret:
                # Get the dimensions of the frame
                (height, width) = frame.shape[:2]

                # Get the size of the text box
                (text_width, text_height), _ = cv2.getTextSize(motion_string, font, font_scale, font_thickness)

                # Calculate the starting position of the text
                start_x = width - text_width - 10  # 10 pixels from the right edge
                start_y = height - 10  # 10 pixels from the bottom edge

                # Draw black outline
                cv2.putText(frame, motion_string, (start_x, start_y), font, font_scale, (0, 0, 0), font_thickness + 2)

                # Draw white text
                cv2.putText(frame, motion_string, (start_x, start_y), font, font_scale, (255, 255, 255), font_thickness)

                # Show the processed frame
                cv2.imshow('Webcam Feed', frame)

                # Wait for the user to press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Could not read frame.")
                break
    finally:
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Run the function to display motion text on webcam feed
display_motion_text_on_webcam()
