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
import threading
from queue import Queue

# Logging configuration
logging.basicConfig(filename='face_auth.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable parameters
EAR_THRESHOLD = 0.16
CONSECUTIVE_FRAMES = 3
DETECTION_WINDOW = 5
MIN_AUTH_INTERVAL = 300
SCALE_FACTOR = 0.5
NUM_WORKERS = max(1, os.cpu_count() - 1)

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
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def has_blink(detections):
    if len(detections) < CONSECUTIVE_FRAMES:
        return False
    recent_detections = list(detections)[-CONSECUTIVE_FRAMES:]
    blink_detected = all(d[1] < EAR_THRESHOLD for d in recent_detections)
    return blink_detected

def capture_frames(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
        else:
            logging.error("Failed to capture frame.")
            break
    cap.release()

def process_frame(frame, known_encodings, known_names, person_tracker, last_auth_time, data_lock, scale_factor=SCALE_FACTOR):
    start_time = time.time()
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    try:
        face_locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample=0)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    except Exception as e:
        logging.error(f"Error in face detection/encoding: {e}")
        return frame
    logging.info(f"Processed frame with {len(face_locations)} faces detected")
    
    processed_frame = frame.copy()
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        logging.info(f"Detected {name}")
        
        try:
            face_landmarks = face_recognition.face_landmarks(small_frame, [(top, right, bottom, left)])
        except Exception as e:
            logging.error(f"Error in landmark detection: {e}")
            face_landmarks = []
        
        ear = None
        if face_landmarks:
            landmarks = face_landmarks[0]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            logging.info(f"EAR for {name}: {ear:.3f}")
        
        with data_lock:
            if name not in person_tracker:
                person_tracker[name] = deque(maxlen=100)
            if ear is not None:
                person_tracker[name].append((time.time(), ear))
            current_time = time.time()
            while person_tracker[name] and current_time - person_tracker[name][0][0] > DETECTION_WINDOW:
                person_tracker[name].popleft()
            if name != "Unknown" and ear is not None:
                detections = list(person_tracker[name])
                detection_freq = len(detections) >= 5
                blink_detected = has_blink(detections)
                last_auth = last_auth_time.get(name, 0)
                time_since_last_auth = current_time - last_auth > MIN_AUTH_INTERVAL
                if detection_freq and blink_detected and time_since_last_auth:
                    logging.info(f"Authorizing {name}")
                    subprocess.Popen(["python", "io.py", name])
                    last_auth_time[name] = current_time
        
        top_scaled = int(top / scale_factor)
        right_scaled = int(right / scale_factor)
        bottom_scaled = int(bottom / scale_factor)
        left_scaled = int(left / scale_factor)
        cv2.rectangle(processed_frame, (left_scaled, top_scaled), (right_scaled, bottom_scaled), (0, 255, 0), 2)
        label = f"{name} EAR: {ear:.2f}" if name != "Unknown" and ear is not None else name
        cv2.putText(processed_frame, label, (left_scaled, top_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    processing_time = time.time() - start_time
    logging.debug(f"Frame processing time: {processing_time:.3f} seconds")
    return processed_frame

def worker(frame_queue, result_queue, stop_event, known_encodings, known_names, person_tracker, last_auth_time, data_lock):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            processed_frame = process_frame(frame, known_encodings, known_names, person_tracker, last_auth_time, data_lock)
            result_queue.put(processed_frame)
        except queue.Empty:
            continue

def main():
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)
    stop_event = threading.Event()
    data_lock = threading.Lock()
    
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        logging.error("No known face encodings loaded.")
        return
    
    person_tracker = {}
    last_auth_time = {}
    
    capture_thread = threading.Thread(target=capture_frames, args=(frame_queue, stop_event))
    capture_thread.start()
    
    workers = []
    for _ in range(NUM_WORKERS):
        worker_thread = threading.Thread(target=worker, args=(frame_queue, result_queue, stop_event, known_encodings, known_names, person_tracker, last_auth_time, data_lock))
        worker_thread.start()
        workers.append(worker_thread)
    
    while not stop_event.is_set():
        frame = None
        while not result_queue.empty():
            frame = result_queue.get()
        if frame is not None:
            cv2.imshow("Face Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
    
    capture_thread.join()
    for worker_thread in workers:
        worker_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()