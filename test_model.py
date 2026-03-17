import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# Load model, scaler, label encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(frame):
    """Extract hand landmarks - same as inference.py"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None

# Test with webcam
print("Testing ISL Model Accuracy...")
print("=" * 50)
print("Showing predictions for hand signs")
print("Press 'q' to quit")
print("=" * 50)

cap = cv2.VideoCapture(0)

frame_count = 0
correct_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Extract landmarks
    landmarks = extract_landmarks(frame)
    
    if landmarks is not None and len(landmarks) == 63:
        # Scale and predict
        landmarks_scaled = scaler.transform([landmarks])
        with torch.no_grad():
            output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            top_3_idx = np.argsort(probabilities)[::-1][:3]
            
            pred_idx = top_3_idx[0]
            confidence = probabilities[pred_idx]
            predicted_letter = le.classes_[pred_idx]
            
            # Display
            cv2.putText(frame, f'Prediction: {predicted_letter} ({confidence:.2f})', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show top 3
            for i, idx in enumerate(top_3_idx):
                cv2.putText(frame, f'{i+1}. {le.classes_[idx]}: {probabilities[idx]:.2f}', 
                           (50, 100 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    cv2.imshow('ISL Sign Interpreter - Test Mode', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTest completed. Tested {frame_count} frames.")
print("If predictions look wrong, the model may have been trained on different data.")
