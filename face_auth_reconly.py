import cv2
import face_recognition
import picamera2
import numpy as np
from scipy.spatial import distance as dist

# Configurable parameters
FULL_HD_RESOLUTION = (1920, 1080)
DETECTION_RESOLUTION = (320, 180)
LANDMARK_RESOLUTION = (480, 270)
FPS = 10

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) for a given eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_faces(frame, detection_resolution, full_hd_resolution):
    """Detect faces in the frame and return their locations in full HD resolution."""
    detection_frame = cv2.resize(frame, detection_resolution)
    face_locations = face_recognition.face_locations(detection_frame)
    detection_scale_factor = full_hd_resolution[0] / detection_resolution[0]
    full_hd_locations = [(int(top * detection_scale_factor), int(right * detection_scale_factor),
                          int(bottom * detection_scale_factor), int(left * detection_scale_factor))
                         for (top, right, bottom, left) in face_locations]
    return full_hd_locations

def main():
    """Main function for face detection with EAR computation."""
    # Initialize camera with Full HD resolution and 10 FPS
    camera = picamera2.Picamera2()
    config = camera.create_preview_configuration(main={"size": FULL_HD_RESOLUTION}, 
                                                 controls={"FrameDurationLimits": (1000000 // FPS, 1000000 // FPS)})
    camera.configure(config)
    camera.start()

    # Scale factor for landmark detection
    landmark_scale_factor = LANDMARK_RESOLUTION[0] / FULL_HD_RESOLUTION[0]

    quit_flag = False
    while not quit_flag:
        # Capture frame
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect faces
        full_hd_locations = detect_faces(frame, DETECTION_RESOLUTION, FULL_HD_RESOLUTION)

        # Prepare for landmark detection
        landmark_frame = cv2.resize(frame, LANDMARK_RESOLUTION)
        landmark_locations = [(int(top * landmark_scale_factor), int(right * landmark_scale_factor),
                               int(bottom * landmark_scale_factor), int(left * landmark_scale_factor))
                              for (top, right, bottom, left) in full_hd_locations]

        # Compute landmarks for all faces
        all_face_landmarks = face_recognition.face_landmarks(landmark_frame, landmark_locations)

        # Process each detected face
        for (top, right, bottom, left), face_landmarks in zip(full_hd_locations, all_face_landmarks):
            if face_landmarks and "left_eye" in face_landmarks and "right_eye" in face_landmarks:
                left_eye = face_landmarks["left_eye"]
                right_eye = face_landmarks["right_eye"]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                label = f"Face EAR: {ear:.2f}"
            else:
                label = "Face"

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frame
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_flag = True

    # Clean up
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()