import requests
import bz2
import os
import sys

def download_file(url, path, description):
    """
    Downloads a file from a URL to a specified path, showing progress.
    """
    if os.path.exists(path):
        print(f"'{os.path.basename(path)}' already exists. Skipping download.")
        return True
    
    print(f"Downloading {description}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
        
        print(f"Successfully downloaded '{os.path.basename(path)}'.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {description}: {e}")
        print("Please check your internet connection and try again.")
        return False
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        return False


def download_dlib_model():
    """
    Downloads and extracts the dlib shape predictor model.
    """
    model_name = "shape_predictor_68_face_landmarks.dat"
    model_folder = "shape_predictor_model"
    model_path = os.path.join(model_folder, model_name)
    
    if os.path.exists(model_path):
        print(f"'{model_name}' already exists. Skipping download.")
        return True

    print(f"Downloading '{model_name}'...")
    url = f"http://dlib.net/files/{model_name}.bz2"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Decompress and save the file
        with open(model_path, "wb") as f_out:
            decompressor = bz2.BZ2Decompressor()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(decompressor.decompress(chunk))
        
        print(f"Successfully downloaded and saved to '{model_path}'")
        return True
    except Exception as e:
        print(f"Error downloading or extracting dlib model: {e}")
        print("Please check your internet connection or download the file manually from dlib.net.")
        return False

def preload_all_models():
    """
    Downloads all necessary models for the proctoring application.
    """
    # --- Create Directories ---
    os.makedirs("shape_predictor_model", exist_ok=True)
    os.makedirs("object_detection_model/weights", exist_ok=True)
    os.makedirs("object_detection_model/config", exist_ok=True)
    os.makedirs("object_detection_model/objectLabels", exist_ok=True)

    # --- Dlib Model ---
    dlib_success = download_dlib_model()
    if not dlib_success:
        return # Stop if dlib model fails

    # --- YOLO Models ---
    print("\nDownloading YOLOv3-tiny models...")
    
    # 1. Weights file
    weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    weights_path = "object_detection_model/weights/yolov3-tiny.weights"
    weights_success = download_file(weights_url, weights_path, "YOLOv3-tiny weights")
    
    # 2. Config file
    cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
    cfg_path = "object_detection_model/config/yolov3-tiny.cfg"
    cfg_success = download_file(cfg_url, cfg_path, "YOLOv3-tiny config")

    # 3. Coco Names file
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    names_path = "object_detection_model/objectLabels/coco.names"
    names_success = download_file(names_url, names_path, "COCO class names")

    if dlib_success and weights_success and cfg_success and names_success:
        print("\nAll models have been successfully downloaded and are ready.")
    else:
        print("\nSome models failed to download. Please check the errors above.")

if __name__ == "__main__":
    preload_all_models()
