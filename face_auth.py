import cv2
import face_recognition
import pickle
import subprocess
import logging
from picamera2 import Picamera2
import numpy as np
import time
from collections import deque

# Configure logging
logging.basicConfig(filename='face_auth.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_encodings(file_path):
    """Load known face encodings and names from a pickle file."""
    print(f"Loading encodings from {file_path}...")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print("Encodings loaded successfully.")
        return data["encodings"], data["names"]
    except Exception as e:
        print(f"Failed to load encodings: {e}")
        logging.error(f"Failed to load encodings from {file_path}: {e}")
        raise

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    print("Calculating Euclidean distance between two points...")
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    print(f"Distance calculated: {dist}")
    return dist

def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR) for a given eye."""
    print("Calculating Eye Aspect Ratio (EAR)...")
    A = distance(eye[1], eye[5])  # Vertical distance 1
    B = distance(eye[2], eye[4])  # Vertical distance 2
    C = distance(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    print(f"EAR calculated: {ear}")
    return ear

def main(encodings_file="encodings.pickle", io_script="io_control.py", tolerance=0.5):
    """Main function for face authentication with blink detection."""
    print("Starting main function...")
    # Load known encodings
    try:
        known_encodings, known_names = load_encodings(encodings_file)
        logging.info("Successfully loaded face encodings")
        print("Face encodings loaded successfully.")
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        print(f"Initialization failed: {e}")
        return
    
    # Initialize camera
    print("Initializing camera...")
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        logging.info("Camera initialized successfully")
        print("Camera initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        print(f"Failed to initialize camera: {e}")
        return
    
    # Dictionary to track persons
    person_tracker = {}
    print("Person tracker initialized.")
    
    try:
        while True:
            # Capture frame in BGR
            print("Capturing frame...")
            frame = picam2.capture_array()
            print("Frame captured successfully.")
            
            # Use BGR frame for face recognition and display
            print("Detecting face locations...")
            face_locations = face_recognition.face_locations(frame, model='hog')
            print(f"Detected {len(face_locations)} face(s).")
            print("Encoding faces...")
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            print("Extracting face landmarks...")
            face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
            print(f"Processed {len(face_encodings)} face encodings and landmarks.")
            
            for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                # Find the best match
                print("Calculating face distances...")
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance_index = np.argmin(distances)
                    if distances[min_distance_index] < tolerance:
                        name = known_names[min_distance_index]
                        color = (0, 255, 0)  # Green for known person
                        logging.info(f"Detected {name} with distance {distances[min_distance_index]:.4f}")
                        print(f"Detected {name} with distance {distances[min_distance_index]:.4f}")
                        
                        # Calculate EAR
                        if 'left_eye' in landmarks and 'right_eye' in landmarks:
                            print(f"Processing eye landmarks for {name}...")
                            left_eye = landmarks['left_eye']
                            right_eye = landmarks['right_eye']
                            ear_left = calculate_ear(left_eye)
                            ear_right = calculate_ear(right_eye)
                            ear_avg = (ear_left + ear_right) / 2.0
                            print(f"Calculated EAR for {name}: {ear_avg:.4f}")
                        else:
                            ear_avg = None
                            logging.warning(f"Could not detect eyes for {name}")
                            print(f"Could not detect eyes for {name}")
                        
                        # Initialize tracker for new person
                        current_time = time.time()
                        if name not in person_tracker:
                            person_tracker[name] = {"detections": deque(), "last_auth": 0}
                            print(f"Initialized tracker for {name}")
                        
                        # Add detection timestamp and EAR
                        print(f"Adding detection for {name} at {current_time}...")
                        person_tracker[name]["detections"].append((current_time, ear_avg))
                        print(f"Detection added for {name}.")
                        
                        # Remove detections older than 5 seconds
                        print(f"Cleaning old detections for {name}...")
                        while person_tracker[name]["detections"] and person_tracker[name]["detections"][0][0] < current_time - 5:
                            person_tracker[name]["detections"].popleft()
                            print(f"Removed old detection for {name}")
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                        logging.info("Detected unknown person")
                        print("Detected unknown person")
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    logging.info("Detected unknown person")
                    print("Detected unknown person")
                
                # Draw rectangle and label on frame
                print(f"Drawing rectangle and label for {name}...")
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                print("Rectangle and label drawn.")
            
            # Authorization logic with blink detection
            print("Checking authorization for tracked persons...")
            for person in list(person_tracker.keys()):
                recent_detections = [d for d in person_tracker[person]['detections'] if time.time() - d[0] <= 5]
                print(f"Checking authorization for {person}: {len(recent_detections)} recent detections")
                if len(recent_detections) >= 5:
                    ears = [d[1] for d in recent_detections if d[1] is not None]
                    if ears and min(ears) < 0.2 and max(ears) > 0.25:  # Check for blink
                        if time.time() - person_tracker[person]['last_auth'] > 300:  # 5 minutes
                            logging.info(f"Authorizing {person} after detecting blink")
                            print(f"Authorizing {person} after detecting blink")
                            try:
                                print(f"Triggering io_control.py for {person}...")
                                subprocess.Popen(["python3", io_script, person])
                                person_tracker[person]['last_auth'] = time.time()
                                logging.info(f"Successfully triggered io_control.py for {person}")
                                print(f"Successfully triggered io_control.py for {person}")
                            except Exception as e:
                                logging.error(f"Failed to execute I/O script for {person}: {e}")
                                print(f"Failed to execute I/O script for {person}: {e}")
                        else:
                            logging.info(f"{person} already authorized recently")
                            print(f"{person} already authorized recently")
                    else:
                        logging.info(f"No blink detected for {person}")
                        print(f"No blink detected for {person}")
                # Update detections
                person_tracker[person]['detections'] = deque(recent_detections)
                print(f"Updated detections for {person}.")
            
            # Display the frame
            print("Displaying frame...")
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting loop...")
                break
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        print(f"Error in main loop: {e}")
    finally:
        print("Stopping camera and closing windows...")
        picam2.stop()
        cv2.destroyAllWindows()
        logging.info("Camera stopped and windows closed")
        print("Camera stopped and windows closed successfully.")

if __name__ == "__main__":
    print("Executing main function...")
    main()