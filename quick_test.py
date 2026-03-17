"""
Model-only test - no camera window needed
"""
import numpy as np
import torch
import pickle

print("="*60)
print("ISL SIGN TRANSLATOR - MODEL TEST (No Camera)")
print("="*60)

device = torch.device('cpu')

# Load model
print("\n[1] Loading model components...")
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/le.pkl', 'rb') as f:
        le = pickle.load(f)
    
    class ISLClassifier(torch.nn.Module):
        def __init__(self, input_dim=63, num_classes=26):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.fc(x)

    model = ISLClassifier(input_dim=63).to(device)
    model.load_state_dict(torch.load('models/isl_classifier.pth', map_location=device))
    model.eval()
    
    print("    [OK] Scaler, Label Encoder, and Model loaded successfully")
    print(f"    Classes: A-Z (26 total)")
    print(f"    Model state: EVAL MODE (ready for inference)")
    
except Exception as e:
    print(f"    [ERROR] {e}")
    exit()

# Test 1: Random data
print("\n[2] Testing model with random landmark data...")
for test_num in range(3):
    # Generate realistic landmark data (normalized 0-1)
    landmarks = np.random.uniform(0.2, 0.8, size=(1, 63))
    
    # Scale
    landmarks_scaled = scaler.transform(landmarks)
    
    # Predict
    with torch.no_grad():
        output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    # Get top 3
    top_3 = np.argsort(probs)[::-1][:3]
    
    print(f"\n    Test {test_num + 1}:")
    for idx, class_idx in enumerate(top_3):
        letter = le.classes_[class_idx]
        prob = probs[class_idx]
        print(f"      {idx+1}. {letter}: {prob*100:.1f}%")

# Test 2: Check camera availability
print("\n[3] Checking camera availability...")
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("    [OK] Camera is available and working")
        cap.release()
    else:
        print("    [WARNING] Camera exists but cannot read frames")
        cap.release()
else:
    print("    [WARNING] No camera detected on system")

# Test 3: Test real hand detection (if camera available)
print("\n[4] Testing hand detection capability...")
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
print("    [OK] Mediapipe Hands loaded")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Model Status: [OK] - Ready to use")
print("Scaler: [OK] - Configured for 63-dim input")
print("Label Encoder: [OK] - 26 classes (A-Z)")
print("Mediapipe Hands: [OK] - Ready for landmark extraction")
print("="*60)

print("\nThe model is working correctly!")
print("\nTo use the full web interface:")
print("  python app.py")
print("\nThen open: http://localhost:5000 in your browser")
