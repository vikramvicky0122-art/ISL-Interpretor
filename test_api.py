import requests
import cv2
import numpy as np
from io import BytesIO

# Test if server is running
try:
    response = requests.get("http://localhost:5000/health")
    print("Server Status:", response.json())
except Exception as e:
    print("ERROR: Server not running:", e)
    exit(1)

# Test with a simple webcam frame
print("\nTesting prediction with webcam frame...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

print("Webcam opened. Capturing 5 test frames...")
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read frame {i}")
        continue
    
    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Send to Flask server
    files = {'frame': ('frame.jpg', BytesIO(buffer), 'image/jpeg')}
    try:
        response = requests.post("http://localhost:5000/predict", files=files)
        if response.status_code == 200:
            data = response.json()
            print(f"Frame {i+1}: {data['letter']} (confidence: {data['confidence']:.2%})")
        else:
            print(f"Frame {i+1}: ERROR {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Frame {i+1}: Network error - {e}")

cap.release()
print("\nTest complete!")
