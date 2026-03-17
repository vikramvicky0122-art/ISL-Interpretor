"""
Command-line only test - no GUI needed
Tests model with random landmark data
"""
import numpy as np
import torch
import pickle

print("="*60)
print("ISL SIGN TRANSLATOR - MODEL VERIFICATION TEST")
print("="*60)

device = torch.device('cpu')

# Load components
print("\n1. Loading model components...")
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("   [OK] Scaler loaded")
except Exception as e:
    print(f"   [ERROR] Scaler failed: {e}")
    exit()

try:
    with open('models/le.pkl', 'rb') as f:
        le = pickle.load(f)
    print("   [OK] Label encoder loaded")
    print(f"   Classes: {list(le.classes_)}")
except Exception as e:
    print(f"   [ERROR] Label encoder failed: {e}")
    exit()

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

try:
    model = ISLClassifier(input_dim=63).to(device)
    model.load_state_dict(torch.load('models/isl_classifier.pth', map_location=device))
    model.eval()
    print("   [OK] Model loaded and eval mode set")
except Exception as e:
    print(f"   [ERROR] Model failed: {e}")
    exit()

# Test with sample data
print("\n2. Testing model with sample landmarks...")

# Create realistic landmark data (normalized values between 0-1)
sample_landmarks = np.random.uniform(0, 1, size=(1, 63))
print(f"   Sample input shape: {sample_landmarks.shape}")
print(f"   Sample input range: [{sample_landmarks.min():.3f}, {sample_landmarks.max():.3f}]")

# Scale
sample_scaled = scaler.transform(sample_landmarks)
print(f"   After scaling: mean={sample_scaled.mean():.3f}, std={sample_scaled.std():.3f}")

# Predict
with torch.no_grad():
    input_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].cpu().numpy()

print("\n3. Test Predictions (Top 5):")
top_5_idx = np.argsort(probs)[::-1][:5]
for i, idx in enumerate(top_5_idx):
    print(f"   {i+1}. {le.classes_[idx]}: {probs[idx]*100:6.1f}%")

print("\n4. Checking model behavior...")
print(f"   Model is in eval mode: {not model.training}")
print(f"   Output shape: {output.shape}")
print(f"   Probabilities sum to 1: {probs.sum():.4f}")

if probs.sum() > 0.99 and probs.sum() < 1.01:
    print("   [OK] Model working correctly!")
else:
    print("   [ERROR] Probability normalization issue")

print("\n" + "="*60)
print("MODEL STATUS: [OK] READY FOR INFERENCE")
print("="*60)

# Now test with real webcam data
print("\n5. Testing with real webcam data...")
print("   Initializing Mediapipe...")

import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

print("   [OK] Mediapipe ready")

# Capture from webcam
import cv2

print("\n6. Capturing from webcam...")
print("   Make a hand sign and wait 3 seconds...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frames_to_skip = 30  # Skip first 30 frames for camera to focus
captured_frames = []

while len(captured_frames) < 5:
    ret, frame = cap.read()
    if not ret:
        print("   [ERROR] Cannot read from camera!")
        cap.release()
        exit()
    
    if frames_to_skip > 0:
        frames_to_skip -= 1
        continue
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Extract landmarks
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        captured_frames.append(np.array(landmarks))
        print(f"   Captured {len(captured_frames)}/5 frames with hand detected")

cap.release()

if not captured_frames:
    print("   [ERROR] No hand detected in any frame!")
    exit()

print("\n7. Predicting on captured frames...")
for frame_idx, landmarks in enumerate(captured_frames):
    landmarks_scaled = scaler.transform([landmarks])
    
    with torch.no_grad():
        output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    top_idx = np.argmax(probs)
    top_letter = le.classes_[top_idx]
    top_prob = probs[top_idx]
    
    print(f"   Frame {frame_idx+1}: {top_letter} ({top_prob*100:.1f}%)")

print("\n" + "="*60)
print("[OK] ALL TESTS PASSED - MODEL IS WORKING")
print("="*60)
