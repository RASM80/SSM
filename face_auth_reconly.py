import cv2
import face_recognition
import pickle
import subprocess
import logging
import picamera2
import numpy as np
import time
from collections import deque
import os
import logging
from scipy.spatial import distance as dist

# Logging configuration
logging.basicConfig(filename='face_auth.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable parameters
FULL_HD_RESOLUTION = (1920, 1080)
DETECTION_RESOLUTION = (320, 180)
RECOGNITION_RESOLUTION = (480, 270)
FPS = 10
SESSION_FRAMES = 50  # 5 seconds at 10 FPS
REQUIRED_RECOGNITIONS = 5
SKIP_FRAMES = 5
EAR_THRESHOLD = 0.185
CONSECUTIVE_FRAMES = 3
MIN_AUTH_INTERVAL = 60 # seconds

def load_encodings(models_dir="models"):
    """Load face encodings and names from pickle files in the models directory."""
    encodings = []
    names = []
    if not os.path.exists(models_dir):
        logging.error(f"Models directory {models_dir} does not exist.")
        return encodings, names
    
    for file in os.listdir(models_dir):
        if file.endswith(".pickle"):
            # Derive the person's name from the filename (e.g., "john.pickle" -> "john")
            name = os.path.splitext(file)[0]
            try:
                with open(os.path.join(models_dir, file), 'rb') as f:
                    encodings_list = pickle.load(f)
                    # Extend the encodings list with the loaded encodings
                    encodings.extend(encodings_list)
                    # Extend the names list with the person's name repeated for each encoding
                    names.extend([name] * len(encodings_list))
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
    return encodings, names

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) for a given eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def has_blink(ear_data, consecutive_frames, ear_threshold):
    """Check if there is at least one blink in the EAR data."""
    if len(ear_data) < consecutive_frames:
        return False
    for i in range(len(ear_data) - consecutive_frames + 1):
        if all(ear_data[j][1] < ear_threshold for j in range(i, i + consecutive_frames)) and \
           all(ear_data[j+1][0] == ear_data[j][0] + 1 for j in range(i, i + consecutive_frames - 1)):
            return True
    return False

def main():
    """Main function for face authentication with session-based recognition."""
     # Load known face encodings
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        print("No known face encodings found. Exiting.")
        logging.error("No known face encodings loaded.")
        return


    # Initialize camera with Full HD resolution and 10 FPS
    camera = picamera2.Picamera2()
    config = camera.create_preview_configuration(main={"size": FULL_HD_RESOLUTION}, 
                                                 controls={"FrameDurationLimits": (1000000 // FPS, 1000000 // FPS)})
    camera.configure(config)
    camera.start()


    # Scale factors
    detection_scale_factor = FULL_HD_RESOLUTION[0] / DETECTION_RESOLUTION[0]  # 3
    rec_scale_factor = RECOGNITION_RESOLUTION[0] / FULL_HD_RESOLUTION[0]      # 0.5

    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("debug.log"),
                            logging.StreamHandler()
                            ]   
                        )

    quit_flag = False
    while not quit_flag:
        # Start a new session
        session_data = {}
        for frame_number in range(SESSION_FRAMES):
            # Capture frame
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Resize for detection
            detection_frame = cv2.resize(frame, DETECTION_RESOLUTION)
            face_locations = face_recognition.face_locations(detection_frame)

            # Scale locations to full HD for display
            full_hd_locations = [(int(top * detection_scale_factor), int(right * detection_scale_factor),
                                  int(bottom * detection_scale_factor), int(left * detection_scale_factor))
                                 for (top, right, bottom, left) in face_locations]

            # Resize for recognition
            recognition_frame = cv2.resize(frame, RECOGNITION_RESOLUTION)
            rec_locations = [(int(top * rec_scale_factor), int(right * rec_scale_factor),
                              int(bottom * rec_scale_factor), int(left * rec_scale_factor))
                             for (top, right, bottom, left) in full_hd_locations]

            # Compute face encodings
            face_encodings = face_recognition.face_encodings(recognition_frame, rec_locations)

            for face_index, ((top, right, bottom, left), face_encoding) in enumerate(zip(full_hd_locations, face_encodings)):
                # Match face to known encodings
                #matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"


                logging.info('frame')

                # Draw bounding box and label on full HD frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{name} EAR: {ear:.2f}" if name != "Unknown" and 'ear' in locals() else name
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display frame
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_flag = True
                break

        if quit_flag:
            break


    # Clean up
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
