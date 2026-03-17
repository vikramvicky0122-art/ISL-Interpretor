import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp
from io import BytesIO
from PIL import Image

# Load everything
device = torch.device('cpu')

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
model.eval()  # IMPORTANT: Set to eval mode to disable dropout

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Test 1: Direct webcam (like inference.py)
print("=" * 60)
print("TEST 1: Direct Webcam (inference.py method)")
print("=" * 60)

cap = cv2.VideoCapture(0)
frame_count = 0
predictions = []

for _ in range(30):  # Capture 30 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        landmarks = np.array(landmarks)
        landmarks_scaled = scaler.transform([landmarks])
        
        with torch.no_grad():
            output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
        
        letter = le.classes_[pred_idx]
        predictions.append((letter, confidence))
        
        cv2.putText(frame, f'{letter} ({confidence:.2f})', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('TEST 1: Direct Webcam', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()

print(f"\nCaptured {frame_count} frames with hand detected")
if predictions:
    print(f"Sample predictions: {predictions[:5]}")
    # Most common prediction
    from collections import Counter
    most_common = Counter([p[0] for p in predictions]).most_common(1)
    if most_common:
        print(f"Most common prediction: {most_common[0][0]} ({len(most_common[0])} frames)")
else:
    print("ERROR: No hands detected!")

# Test 2: Simulated web request (JPEG compression)
print("\n" + "=" * 60)
print("TEST 2: Web Method (JPEG compression simulation)")
print("=" * 60)

cap = cv2.VideoCapture(0)
web_predictions = []

for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Simulate web request: Convert to PIL, then compress as JPEG, then back
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Simulate JPEG compression
    jpeg_buffer = BytesIO()
    pil_img.save(jpeg_buffer, format='JPEG', quality=80)
    jpeg_buffer.seek(0)
    
    # Read back
    compressed_img = Image.open(jpeg_buffer)
    frame_from_web = cv2.cvtColor(np.array(compressed_img), cv2.COLOR_RGB2BGR)
    
    # Process
    rgb_frame = cv2.cvtColor(frame_from_web, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        landmarks = np.array(landmarks)
        landmarks_scaled = scaler.transform([landmarks])
        
        with torch.no_grad():
            output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
        
        letter = le.classes_[pred_idx]
        web_predictions.append((letter, confidence))
        
        cv2.putText(frame_from_web, f'{letter} ({confidence:.2f})', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('TEST 2: Web Method (JPEG)', frame_from_web)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nCaptured {len(web_predictions)} frames with hand detected")
if web_predictions:
    print(f"Sample predictions: {web_predictions[:5]}")
    from collections import Counter
    most_common = Counter([p[0] for p in web_predictions]).most_common(1)
    if most_common:
        print(f"Most common prediction: {most_common[0][0]} ({len(most_common[0])} frames)")
else:
    print("ERROR: No hands detected in web method!")

# Comparison
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
if predictions and web_predictions:
    direct_acc = np.mean([p[1] for p in predictions])
    web_acc = np.mean([p[1] for p in web_predictions])
    print(f"Direct acc: {direct_acc:.3f}")
    print(f"Web acc: {web_acc:.3f}")
    print(f"Difference: {abs(direct_acc - web_acc):.3f}")
    
    if abs(direct_acc - web_acc) > 0.1:
        print("\n⚠️  JPEG compression is affecting accuracy!")
    else:
        print("\n✓ JPEG compression is not the issue")
else:
    print("Cannot compare - hand detection failed in one or both methods")
