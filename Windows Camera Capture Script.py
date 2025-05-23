import os
import cv2
import re

def get_folder_name(parent_dir):
    while True:
        subfolder_name = input("Enter folder name: ")
        full_path = os.path.join(parent_dir, subfolder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        else:
            print("Folder already exists.")
            while True:
                choice = input("Do you want to (1) change the name or (2) add pictures to the existing folder? Enter 1 or 2: ")
                if choice == '1':
                    break  # go back to ask for a new name
                elif choice == '2':
                    return full_path
                else:
                    print("Invalid choice. Please enter 1 or 2.")

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera detected")
        return None
    return cap

def get_next_counter(folder_path):
    existing_files = os.listdir(folder_path)
    pattern = r'image_(\d{3})\.jpg'
    numbers = []
    for file in existing_files:
        match = re.match(pattern, file)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        return max(numbers) + 1
    else:
        return 0

def capture_images(folder_path):
    cap = initialize_camera()
    if cap is None:
        return
    counter = get_next_counter(folder_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 13:  # Enter key
            filename = os.path.join(folder_path, f"image_{counter:03d}.jpg")
            cv2.imwrite(filename, frame)
            counter += 1
            print(f"Saved {filename}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    authorized_faces_dir = os.path.join(script_dir, "authorized_faces")
    if not os.path.exists(authorized_faces_dir):
        os.makedirs(authorized_faces_dir)
    folder_path = get_folder_name(authorized_faces_dir)
    capture_images(folder_path)