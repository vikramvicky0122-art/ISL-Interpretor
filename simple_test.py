"""
Simple test to verify model accuracy on a single image
"""
import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp

print("Loading model and dependencies...")

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
model.eval()  # CRITICAL

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

print("✓ Model loaded\n")

# Capture one frame
print("Press SPACE to capture a frame, ESCAPE to quit")
cap = cv2.VideoCapture(0)
frame_captured = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    cv2.imshow('Capture Test Image', frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == 32:  # SPACE
        frame_captured = frame
        print("✓ Frame captured")
        break
    elif key == 27:  # ESCAPE
        print("Cancelled")
        exit()

cap.release()
cv2.destroyAllWindows()

if frame_captured is None:
    print("No frame captured!")
    exit()

# Process frame
print("\nProcessing frame...")
rgb_frame = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)
results = hands.process(rgb_frame)

if not results.multi_hand_landmarks:
    print("❌ No hand detected in captured frame!")
    cv2.imshow('Failed', frame_captured)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    exit()

print("✓ Hand detected")

# Extract landmarks
landmarks = []
for hand_landmarks in results.multi_hand_landmarks:
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

landmarks = np.array(landmarks)
print(f"✓ Landmarks extracted: {len(landmarks)} values")

# Scale
landmarks_scaled = scaler.transform([landmarks])
print(f"✓ Landmarks scaled")
print(f"  Mean: {landmarks_scaled[0].mean():.4f}")
print(f"  Std: {landmarks_scaled[0].std():.4f}")

# Predict
with torch.no_grad():
    input_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32).to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].cpu().numpy()

# Show top 5
top_5_idx = np.argsort(probs)[::-1][:5]

print(f"\n✓ Prediction complete!\n")
print("Top 5 predictions:")
for i, idx in enumerate(top_5_idx):
    print(f"  {i+1}. {le.classes_[idx]:2s}: {probs[idx]*100:6.1f}%")

print("\n" + "="*50)
pred_letter = le.classes_[top_5_idx[0]]
print(f"PREDICTED: {pred_letter}")
print("="*50)
