import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
import pyttsx3
import mediapipe as mp
import os

# Model class from train
class ISLClassifier(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

def extract_landmarks(frame, hands):
    """Extract hand landmarks from frame using Mediapipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None

def main():
    # Load model, scaler, label encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/le.pkl', 'rb') as f:
        le = pickle.load(f)
    
    model = ISLClassifier(input_dim=63).to(device)
    model.load_state_dict(torch.load('models/isl_classifier.pth', map_location=device))
    model.eval()
    
    # Initialize Mediapipe and TTS
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Extract landmarks
        landmarks = extract_landmarks(frame, hands)
        
        if landmarks is not None and len(landmarks) == 63:
            # Scale and predict
            landmarks_scaled = scaler.transform([landmarks])
            with torch.no_grad():
                output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
                pred_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
                confidence = torch.softmax(output, dim=1)[0, pred_idx].cpu().numpy()
            
            predicted_letter = le.classes_[pred_idx]
            
            # Display on frame
            cv2.putText(frame, f'{predicted_letter} ({confidence:.2f})', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            # Speak the letter
            if confidence > 0.7:
                engine.say(predicted_letter)
                engine.runAndWait()
        
        cv2.imshow('ISL Sign Translator', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
