import cv2
import face_recognition
import pickle
import subprocess
import logging
import numpy as np
import time
from collections import deque
import os
from scipy.spatial import distance as dist

# Logging configuration
logging.basicConfig(filename='face_auth.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable parameters
EAR_THRESHOLD = 0.16  # Threshold for blink detection
CONSECUTIVE_FRAMES = 3  # Number of consecutive frames for a blink
DETECTION_WINDOW = 5  # Seconds to consider for detection frequency
MIN_AUTH_INTERVAL = 300  # Minimum time between authorizations (5 minutes)

def load_encodings(models_dir="models"):
    encodings = []
    names = []
    for file in os.listdir(models_dir):
        if file.endswith(".pickle"):
            try:
                with open(os.path.join(models_dir, file), 'rb') as f:
                    person_encodings = pickle.load(f)
                    person_name = os.path.splitext(file)[0]  # Extract name from filename
                    for encoding in person_encodings:
                        encodings.append(encoding)
                        names.append(person_name)
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

def has_blink(detections, frame_counter):
    """Check for blink patterns in detection history."""
    if len(detections) < CONSECUTIVE_FRAMES:
        return False
    recent_detections = list(detections)[-CONSECUTIVE_FRAMES:]
    blink_detected = all(d[2] < EAR_THRESHOLD for d in recent_detections)
    return blink_detected

def main():
    """Main function for face authentication with blink detection."""
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        print("No known face encodings found. Exiting.")
        logging.error("No known face encodings loaded.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        logging.error("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    person_tracker = {}
    last_auth_time = {}
    frame_counter = 0

    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                logging.error("Failed to capture frame.")
                break
            frame_counter += 1


            # Detect faces on the smaller frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            print(f"Frame {frame_counter}: Detected {len(face_locations)} faces")

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Match face to known encodings
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Get facial landmarks for EAR calculation
                face_landmarks = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])
                if face_landmarks:
                    landmarks = face_landmarks[0]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    print(f"Detected {name} with EAR: {ear:.3f}")

                    # Update person tracker
                    if name not in person_tracker:
                        person_tracker[name] = deque(maxlen=100)
                    person_tracker[name].append((time.time(), frame_counter, ear))

                    # Remove old detections
                    current_time = time.time()
                    while person_tracker[name] and current_time - person_tracker[name][0][0] > DETECTION_WINDOW:
                        person_tracker[name].popleft()

                    # Check authorization conditions
                    if name != "Unknown":
                        detections = person_tracker[name]
                        detection_freq = len(detections) >= 5  # At least 5 detections in 5 seconds
                        blink_detected = has_blink(detections, frame_counter)
                        last_auth = last_auth_time.get(name, 0)
                        time_since_last_auth = current_time - last_auth > MIN_AUTH_INTERVAL

                        print(f"{name} - Detections: {len(detections)}, Blink: {blink_detected}, "
                              f"Time since last auth: {current_time - last_auth:.1f}s")

                        if detection_freq and blink_detected and time_since_last_auth:
                            print(f"Authorizing {name}")
                            logging.info(f"Authorizing {name}")
                            subprocess.run(["python", "io.py", name])
                            last_auth_time[name] = current_time

                # Draw bounding box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{name} EAR: {ear:.2f}" if name != "Unknown" else name
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display frame
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()