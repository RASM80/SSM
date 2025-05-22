import os
import face_recognition
import pickle

def preprocess_faces(main_folder, output_file):
    known_encodings = []
    known_names = []
    for person_name in os.listdir(main_folder):
        person_dir = os.path.join(main_folder, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, image_file)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image, model='cnn')
                if len(face_locations) == 1:
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                else:
                    print(f"Warning: {len(face_locations)} faces found in {image_path}. Expected 1.")
    data = {"encodings": known_encodings, "names": known_names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Preprocessing complete. Data saved to {output_file}")

if __name__ == "__main__":
    main_folder = "authorized_faces"
    output_file = "encodings.pickle"
    preprocess_faces(main_folder, output_file)
