import cv2
import face_recognition
import pickle
import subprocess
import logging
from picamera2 import Picamera2
import numpy as np
import time
from collections import deque
import os  # Added for directory operations

# Configure logging
logging.basicConfig(filename='face_auth.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_encodings(models_dir):
    """Load known face encodings and names from pickle files in the models directory."""
    known_encodings = []
    known_names = []
    if not os.path.exists(models_dir):
        logging.error(f"Models directory {models_dir} does not exist")
        raise FileNotFoundError(f"Models directory {models_dir} does not exist")
    for filename in os.listdir(models_dir):
        if filename.endswith(".pickle"):
            name = os.path.splitext(filename)[0]  # Extract name, e.g., "alice" from "alice.pickle"
            file_path = os.path.join(models_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    encodings = pickle.load(f)  # Load list of encodings
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)  # Associate each encoding with the person’s name
            except Exception as e:
                logging.error(f"Failed to load encodings from {file_path}: {e}")
    if not known_encodings:
        logging.warning("No encodings loaded. Check if the models directory contains valid pickle files.")
    return known_encodings, known_names

def main(models_dir="models", io_script="io_control.py", tolerance=0.5):
    """Main function for face authentication."""
    # Load known encodings
    try:
        known_encodings, known_names = load_encodings(models_dir)
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
            # Capture frame in RGB
            frame_rgb = picam2.capture_array()
            
            # Convert to BGR for OpenCV drawing and display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Use RGB frame for face recognition
            face_locations = face_recognition.face_locations(frame_rgb, model='hog')
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Find the best match
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance_index = np.argmin(distances)
                    if distances[min_distance_index] < tolerance:
                        name = known_names[min_distance_index]
                        color = (0, 255, 0)  # Green for known person
                        logging.info(f"Detected {name} with distance {distances[min_distance_index]:.4f}")
                        
                        # Initialize tracker for new person
                        current_time = time.time()
                        if name not in person_tracker:
                            person_tracker[name] = {"detections": deque(), "last_auth": 0}
                        
                        # Add detection timestamp
                        person_tracker[name]["detections"].append(current_time)
                        
                        # Remove detections older than 5 seconds
                        while person_tracker[name]["detections"] and person_tracker[name]["detections"][0] < current_time - 5:
                            person_tracker[name]["detections"].popleft()
                        
                        # Check authorization conditions
                        if len(person_tracker[name]["detections"]) >= 5:
                            if current_time - person_tracker[name]["last_auth"] > 300:  # 5 minutes
                                logging.info(f"Authorizing {name}")
                                try:
                                    subprocess.Popen(["python3", io_script, name])
                                    person_tracker[name]["last_auth"] = current_time
                                    logging.info(f"Successfully triggered io_control.py for {name}")
                                except Exception as e:
                                    logging.error(f"Failed to execute I/O script for {name}: {e}")
                            else:
                                logging.info(f"{name} already authorized recently")
                        else:
                            logging.info(f"Not enough detections for {name}: {len(person_tracker[name]['detections'])}")
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                        logging.info("Detected unknown person")
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    logging.info("Detected unknown person")
                
                # Draw rectangle and label on BGR frame
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                cv2.putText(frame_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Display the frame
            cv2.imshow("Face Authentication", frame_bgr)
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