import os
import face_recognition
import pickle

def preprocess_faces(main_folder, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each person's folder
    for person_name in os.listdir(main_folder):
        person_dir = os.path.join(main_folder, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        # Collect face encodings from images
        encodings = []
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, image_file)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image, model='hog')
                if len(face_locations) == 1:
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    encodings.append(encoding)
                else:
                    print(f"Warning: {len(face_locations)} faces found in {image_path}. Expected 1.")
        
        # Save encodings if there are any
        if encodings:
            output_file = os.path.join(output_dir, f"{person_name.lower()}.pickle")
            # Remove existing model file if it exists
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"Removed existing model file: {output_file}")
            # Save new model file
            with open(output_file, "wb") as f:
                pickle.dump(encodings, f)
            print(f"Saved encodings for {person_name} to {output_file}")

if __name__ == "__main__":
    main_folder = "authorized_faces"
    output_dir = "models"
    preprocess_faces(main_folder, output_dir)