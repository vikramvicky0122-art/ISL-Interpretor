from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
import mediapipe as mp
from io import BytesIO
from PIL import Image
import logging
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model, scaler, label encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    scaler = None

try:
    with open('models/le.pkl', 'rb') as f:
        le = pickle.load(f)
    print("✓ Label Encoder loaded")
except Exception as e:
    logger.error(f"Failed to load label encoder: {e}")
    le = None

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

try:
    model = ISLClassifier(input_dim=63).to(device)
    model.load_state_dict(torch.load('models/isl_classifier.pth', map_location=device))
    model.eval()
    print("✓ Model loaded and ready")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Initialize Mediapipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    print("✓ Mediapipe initialized")
except Exception as e:
    logger.error(f"Failed to initialize Mediapipe: {e}")
    hands = None

def extract_landmarks(frame):
    """Extract hand landmarks from frame"""
    if hands is None:
        return None
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks)
        return None
    except Exception as e:
        logger.error(f"Error extracting landmarks: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'mediapipe_loaded': hands is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None or le is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get frame from request
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        file = request.files['frame']
        if file.filename == '':
            return jsonify({'error': 'No frame selected'}), 400
        
        # Read and convert image
        try:
            img = Image.open(BytesIO(file.read()))
            img = img.convert('RGB')
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Extract landmarks
        landmarks = extract_landmarks(frame)
        
        if landmarks is not None and len(landmarks) == 63:
            try:
                # Scale and predict
                landmarks_scaled = scaler.transform([landmarks])
                with torch.no_grad():
                    output = model(torch.tensor(landmarks_scaled, dtype=torch.float32).to(device))
                    pred_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
                    confidence = torch.softmax(output, dim=1)[0, pred_idx].cpu().item()
                
                predicted_letter = le.classes_[pred_idx]
                
                return jsonify({
                    'letter': predicted_letter,
                    'confidence': float(confidence)
                })
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'letter': 'No hand detected',
            'confidence': 0.0
        })
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ISL Sign Translator Web App")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
