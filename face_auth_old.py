import cv2
import numpy as np
import face_recognition
from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(2)  # Allow camera to warm up

# Load known face encodings (example)
known_face_encodings = []  # Add your known encodings here
known_face_names = []      # Add corresponding names here

# Main loop
while True:
    # Capture frame (assuming it’s in RGB format despite BGR888 config)
    frame_rgb = picam2.capture_array()
    
    # Convert to BGR for OpenCV drawing and display
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Use the RGB frame directly for face recognition
    face_locations = face_recognition.face_locations(frame_rgb, model='hog')
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    
    # Process each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Check if there’s a match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Draw rectangle and label (using BGR colors)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    
    # Display the frame
    cv2.imshow("Face Authentication", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
