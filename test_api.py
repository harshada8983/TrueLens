import requests
import json
import os
import glob

url = "http://localhost:8000/api/predict"

# Intelligently find ANY image in your processed data folder
search_path = os.path.join("backend", "data", "processed_train", "real", "*.jpg")
images = glob.glob(search_path)

if not images:
    print("Error: Could not find any images in the processed_train/real folder!")
else:
    # Grab the very first image the system finds
    image_path = images[0] 
    print(f"Sending {image_path} to TrueLens API...")
    
    # Open the file and send the POST request
    with open(image_path, "rb") as image_file:
        # We dynamically pass the actual filename
        files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
        response = requests.post(url, files=files)
    
    # Print the beautifully formatted JSON response
    print("\n--- API RESPONSE ---")
    print(json.dumps(response.json(), indent=4))