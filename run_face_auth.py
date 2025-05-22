import subprocess

def run_face_auth():
    try:
        subprocess.run(["python3", "face_auth.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running face_auth.py: {e}")

if __name__ == "__main__":
    run_face_auth()