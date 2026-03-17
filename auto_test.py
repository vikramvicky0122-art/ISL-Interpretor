"""
Automatic prediction test - shows continuous predictions
"""
import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp

print("Loading model...")

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
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

print("✓ Model loaded\n")
print("Starting webcam... Press 'q' to quit")
print("Make a hand sign and watch predictions below\n")

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Extract landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        landmarks = np.array(landmarks)
        landmarks_scaled = scaler.transform([landmarks])
        
        # Predict
        with torch.no_grad():
            output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        # Get top 3
        top_3_idx = np.argsort(probs)[::-1][:3]
        
        # Display on frame
        y_pos = 40
        for i, idx in enumerate(top_3_idx):
            letter = le.classes_[idx]
            prob = probs[idx]
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.putText(frame, f'{i+1}. {letter}: {prob*100:.1f}%', (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            y_pos += 50
        
        # Print to console
        if frame_count % 10 == 0:
            print(f"\nFrame {frame_count}:")
            for i, idx in enumerate(top_3_idx):
                print(f"  {i+1}. {le.classes_[idx]}: {probs[idx]*100:.1f}%")
            
    else:
        cv2.putText(frame, 'No hand detected', (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('ISL Prediction Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nTest completed")
