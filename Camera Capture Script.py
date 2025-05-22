import os
import cv2

def get_folder_name(parent_dir):
    while True:
        subfolder_name = input("Enter folder name: ")
        full_path = os.path.join(parent_dir, subfolder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        else:
            print("Folder already exists. Please choose a different name.")

def initialize_camera():
    use_picamera = False
    picam2 = None
    cap = None
    try:
        from picamera2 import Picamera2
        camera_info = Picamera2.global_camera_info()
        if camera_info:
            use_picamera = True
            picam2 = Picamera2()
            picam2.start()
    except ImportError:
        pass
    if not use_picamera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No camera detected")
            return None, None, False
    return picam2, cap, use_picamera

def capture_images(folder_path):
    picam2, cap, use_picamera = initialize_camera()
    if picam2 is None and cap is None:
        return
    counter = 0
    while True:
        if use_picamera:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
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
    if use_picamera:
        picam2.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    authorized_faces_dir = os.path.join(script_dir, "authorized_faces")
    if not os.path.exists(authorized_faces_dir):
        os.makedirs(authorized_faces_dir)
    folder_path = get_folder_name(authorized_faces_dir)
    capture_images(folder_path)