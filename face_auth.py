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
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    except Exception as e:
        logging.error(f"Failed to load encodings from {file_path}: {e}")
        raise

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR) for a given eye."""
    A = distance(eye[1], eye[5])  # Vertical distance 1
    B = distance(eye[2], eye[4])  # Vertical distance 2
    C = distance(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

def main(encodings_file="encodings.pickle", io_script="io_control.py", tolerance=0.5):
    """Main function for face authentication with blink detection."""
    # Load known encodings
    try:
        known_encodings, known_names = load_encodings(encodings_file)
        logging.info("Successfully loaded face encodings")
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return
    
    # Initialize camera
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        logging.info("Camera initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        return
    
    # Dictionary to track persons
    person_tracker = {}
    
    try:
        while True:
            # Capture frame in BGR
            frame = picam2.capture_array()
            
            # Use BGR frame for face recognition and display
            face_locations = face_recognition.face_locations(frame, model='hog')
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
            
            for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                # Find the best match
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance_index = np.argmin(distances)
                    if distances[min_distance_index] < tolerance:
                        name = known_names[min_distance_index]
                        color = (0, 255, 0)  # Green for known person
                        logging.info(f"Detected {name} with distance {distances[min_distance_index]:.4f}")
                        
                        # Calculate EAR
                        if 'left_eye' in landmarks and 'right_eye' in landmarks:
                            left_eye = landmarks['left_eye']
                            right_eye = landmarks['right_eye']
                            ear_left = calculate_ear(left_eye)
                            ear_right = calculate_ear(right_eye)
                            ear_avg = (ear_left + ear_right) / 2.0
                        else:
                            ear_avg = None
                            logging.warning(f"Could not detect eyes for {name}")
                        
                        # Initialize tracker for new person
                        current_time = time.time()
                        if name not in person_tracker:
                            person_tracker[name] = {"detections": deque(), "last_auth": 0}
                        
                        # Add detection timestamp and EAR
                        person_tracker[name]["detections"].append((current_time, ear_avg))
                        
                        # Remove detections older than 5 seconds
                        while person_tracker[name]["detections"] and person_tracker[name]["detections"][0][0] < current_time - 5:
                            person_tracker[name]["detections"].popleft()
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                        logging.info("Detected unknown person")
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    logging.info("Detected unknown person")
                
                # Draw rectangle and label on frame
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Authorization logic with blink detection
            for person in list(person_tracker.keys()):
                recent_detections = [d for d in person_tracker[person]['detections'] if time.time() - d[0] <= 5]
                if len(recent_detections) >= 5:
                    ears = [d[1] for d in recent_detections if d[1] is not None]
                    if ears and min(ears) < 0.2 and max(ears) > 0.25:  # Check for blink
                        if time.time() - person_tracker[person]['last_auth'] > 300:  # 5 minutes
                            logging.info(f"Authorizing {person} after detecting blink")
                            try:
                                subprocess.Popen(["python3", io_script, person])
                                person_tracker[person]['last_auth'] = time.time()
                                logging.info(f"Successfully triggered io_control.py for {person}")
                            except Exception as e:
                                logging.error(f"Failed to execute I/O script for {person}: {e}")
                        else:
                            logging.info(f"{person} already authorized recently")
                    else:
                        logging.info(f"No blink detected for {person}")
                # Update detections
                person_tracker[person]['detections'] = deque(recent_detections)
            
            # Display the frame
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        logging.info("Camera stopped and windows closed")

if __name__ == "__main__":
    main()