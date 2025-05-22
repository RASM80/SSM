import os
import pickle
import face_recognition
import cv2

def load_all_encodings(models_dir):
    """Load all pickle files into a dictionary."""
    known_encodings = {}
    for pickle_file in os.listdir(models_dir):
        if pickle_file.endswith(".pickle"):
            person_name = os.path.splitext(pickle_file)[0]
            with open(os.path.join(models_dir, pickle_file), "rb") as f:
                encodings = pickle.load(f)
            known_encodings[person_name] = encodings
    return known_encodings

def main(models_dir="models", tolerance=0.6):
    # Load all encodings
    known_encodings = load_all_encodings(models_dir)
    
    # Initialize camera (example with OpenCV)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Recognize faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            for person, encodings in known_encodings.items():
                matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=tolerance)
                if True in matches:
                    name = person
                    break
            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display frame
        cv2.imshow("Face Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()