from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
import mediapipe as mp
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load model, scaler, label encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/le.pkl', 'rb') as f:
    le = pickle.load(f)

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

model = ISLClassifier(input_dim=63).to(device)
model.load_state_dict(torch.load('models/isl_classifier.pth', map_location=device))
model.eval()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(frame):
    """Extract hand landmarks from frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get frame from request
        file = request.files['frame']
        img = Image.open(BytesIO(file.read()))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks = extract_landmarks(frame)
        
        if landmarks is not None and len(landmarks) == 63:
            # Scale and predict
            landmarks_scaled = scaler.transform([landmarks])
            with torch.no_grad():
                output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
                pred_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
                confidence = torch.softmax(output, dim=1)[0, pred_idx].cpu().item()
            
            predicted_letter = le.classes_[pred_idx]
            
            return jsonify({
                'letter': predicted_letter,
                'confidence': confidence
            })
        
        return jsonify({
            'letter': 'No hand detected',
            'confidence': 0.0
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting ISL Sign Translator Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
