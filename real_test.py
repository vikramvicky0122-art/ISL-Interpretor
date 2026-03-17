"""
Real hand landmark extraction and model prediction test
Simplified version - console-based output
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
from pathlib import Path

print("\n" + "="*70)
print("REAL HAND LANDMARK TEST")
print("="*70)

# Load model components
print("\n[1] Loading model components...")
device = torch.device("cpu")

# Load model
checkpoint = torch.load("models/isl_classifier.pth", map_location=device)

class ISLClassifier(torch.nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
        super(ISLClassifier, self).__init__()
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
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

model = ISLClassifier(input_dim=63, num_classes=26)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

# Load scaler and encoder
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/le.pkl", "rb") as f:
    le = pickle.load(f)

print("    [OK] Model loaded")
print("    [OK] Scaler loaded")
print("    [OK] Label encoder loaded (26 classes)")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

print("\n[2] Starting camera...")
print("    Position your hand in front of camera")
print("    Press SPACE to capture")
print("    Press Q to quit\n")

cap = cv2.VideoCapture(0)
frame_count = 0
captured_landmarks = None
hand_detected_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Simple display
    display_frame = frame.copy()
    hand_detected = False
    
    if results.multi_hand_landmarks:
        hand_detected = True
        hand_detected_count += 1
        status = "HAND DETECTED - Press SPACE to capture"
        color = (0, 255, 0)
    else:
        hand_detected_count = 0
        status = "NO HAND - Move hand to camera"
        color = (0, 0, 255)
    
    cv2.putText(display_frame, status, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(display_frame, f"Frame: {frame_count}  Detected: {hand_detected_count}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(display_frame, "Q to quit", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.imshow("Camera - Show Hand Gesture", display_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord(" "):  # SPACE - capture
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])

            captured_landmarks = np.array(landmarks_list, dtype=np.float32)
            print(f"\n[3] Hand captured successfully!")
            print(f"    Frame: {frame_count}")
            print(f"    Landmarks shape: {captured_landmarks.shape}")
            print(f"    Landmarks range: {captured_landmarks.min():.6f} to {captured_landmarks.max():.6f}")
            break
        else:
            print(f"\n!!! No hand detected on frame {frame_count}")
            print("    Keep your hand visible and try again...")

    if key == ord("q"):
        print("\nQuitting...")
        break

cap.release()
cv2.destroyAllWindows()

if captured_landmarks is not None:
    print("\n[4] Processing landmarks...")
    
    # Scale landmarks
    scaled_landmarks = scaler.transform(captured_landmarks.reshape(1, -1))

    # Model prediction
    with torch.no_grad():
        landmarks_tensor = torch.FloatTensor(scaled_landmarks).to(device)
        logits = model(landmarks_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1)

    pred_letter = le.inverse_transform(predicted_idx.cpu().numpy())[0]
    pred_confidence = probabilities[0, predicted_idx[0]].item() * 100

    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print(f"\nMain Prediction:  {pred_letter}  ({pred_confidence:.1f}%)\n")

    # Top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    print("TOP 3 PREDICTIONS:")
    for i in range(3):
        letter = le.inverse_transform([top3_indices[i].item()])[0]
        confidence = top3_probs[i].item() * 100
        bar = "*" * int(confidence / 5)
        print(f"  {i+1}. {letter}  {confidence:6.1f}%  {bar}")

    print("\n" + "="*70)
else:
    print("\nNo hand captured.")
