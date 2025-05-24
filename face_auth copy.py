import threading
import queue
import time
import cv2
import numpy as np
import face_recognition
from picamera2 import Picamera2

# Constants
FPS = 10
SESSION_FRAMES = 50
MIN_AUTH_INTERVAL = 5  # seconds
REQUIRED_RECOGNITIONS = 5
SKIP_FRAMES = 2
CONSECUTIVE_FRAMES = 3
EAR_THRESHOLD = 0.2
DETECTION_RESOLUTION = (0, 0, 640, 480)
RECOGNITION_RESOLUTION = 0.25  # 1/4 of full HD

# Shared variables
frame_queue = queue.Queue(maxsize=SESSION_FRAMES)
start_capture_event = threading.Event()
session_complete_event = threading.Event()

def load_encodings():
    # Placeholder: Replace with actual known encodings and names
    known_encodings = []
    known_names = []
    return known_encodings, known_names

def calculate_ear(eye):
    # Simplified EAR calculation
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def has_blink(ear_data, consecutive_frames=CONSECUTIVE_FRAMES, ear_threshold=EAR_THRESHOLD):
    ear_data = sorted(ear_data, key=lambda x: x[0])  # Sort by frame number
    for i in range(len(ear_data) - consecutive_frames + 1):
        if all(ear_data[j][1] < ear_threshold for j in range(i, i + consecutive_frames)):
            if ear_data[i + consecutive_frames - 1][0] - ear_data[i][0] == consecutive_frames - 1:
                return True
    return False

def capture_task():
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (1920, 1080)},
        controls={"FrameDurationLimits": (100000, 100000)}  # 10 FPS
    )
    camera.configure(config)
    camera.start()
    try:
        while True:
            start_capture_event.wait()
            start_capture_event.clear()
            for _ in range(SESSION_FRAMES):
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_queue.put(frame)
            session_complete_event.set()
    finally:
        camera.stop()
        camera.close()

def processing_task():
    known_encodings, known_names = load_encodings()
    last_auth_time = {}
    session_data = {}
    frame_counter = 0

    while True:
        # Wait for the first frame to start processing
        frame = frame_queue.get()
        frame_counter += 1

        # Resize for detection
        detection_frame = cv2.resize(frame, (DETECTION_RESOLUTION[2], DETECTION_RESOLUTION[3]))
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        scale_x = 1920 / DETECTION_RESOLUTION[2]
        scale_y = 1080 / DETECTION_RESOLUTION[3]

        # Process each face
        for top, right, bottom, left in face_locations:
            # Scale to recognition resolution
            face_frame = frame[int(top * scale_y):int(bottom * scale_y), int(left * scale_x):int(right * scale_x)]
            face_frame = cv2.resize(face_frame, (0, 0), fx=RECOGNITION_RESOLUTION, fy=RECOGNITION_RESOLUTION)
            rgb_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            if encodings:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Initialize session data for this person if not present
                if name not in session_data:
                    session_data[name] = {'recognition_frames': [], 'ear_data': []}

                # Recognition with skip frames
                last_rec_frame = session_data[name]['recognition_frames'][-1] if session_data[name]['recognition_frames'] else -SKIP_FRAMES - 1
                if frame_counter - 1 > last_rec_frame + SKIP_FRAMES:
                    session_data[name]['recognition_frames'].append(frame_counter - 1)

                # Blink detection
                landmarks = face_recognition.face_landmarks(rgb_face)
                if landmarks:
                    left_eye = landmarks[0]['left_eye']
                    right_eye = landmarks[0]['right_eye']
                    ear_left = calculate_ear(np.array(left_eye))
                    ear_right = calculate_ear(np.array(right_eye))
                    ear = (ear_left + ear_right) / 2.0
                    session_data[name]['ear_data'].append((frame_counter - 1, ear))

                # Draw on original frame
                top, right, bottom, left = int(top * scale_y), int(right * scale_x), int(bottom * scale_y), int(left * scale_x)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # End of session
        if frame_counter == SESSION_FRAMES:
            session_complete_event.wait()
            session_complete_event.clear()
            current_time = time.time()
            authorized = False

            for name in session_data:
                recognitions = len(session_data[name]['recognition_frames'])
                if (recognitions >= REQUIRED_RECOGNITIONS and 
                    has_blink(session_data[name]['ear_data']) and 
                    current_time - last_auth_time.get(name, 0) > MIN_AUTH_INTERVAL):
                    print(f"Authorized: {name}")
                    last_auth_time[name] = current_time
                    authorized = True

            # Reset for next session
            session_data = {}
            frame_counter = 0
            if authorized:
                time.sleep(MIN_AUTH_INTERVAL)
            start_capture_event.set()

# Start threads
capture_thread = threading.Thread(target=capture_task, daemon=True)
processing_thread = threading.Thread(target=processing_task, daemon=True)
capture_thread.start()
processing_thread.start()

# Initiate first capture
start_capture_event.set()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()