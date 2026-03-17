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
        logger.error("Hands detector not initialized")
        return None
    
    try:
        # Validate frame
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame received")
            return None
        
        logger.debug(f"Processing frame: shape={frame.shape}, dtype={frame.dtype}")
        
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with Mediapipe
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            logger.info(f"Hands detected: {len(results.multi_hand_landmarks)}")
            landmarks = []
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                logger.debug(f"Hand {hand_idx}: {len(hand_landmarks.landmark)} landmarks")
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            
            return np.array(landmarks)
        else:
            logger.debug("No hands detected in frame")
            return None
    except Exception as e:
        logger.error(f"Error extracting landmarks: {e}", exc_info=True)
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
            img_data = file.read()
            img = Image.open(BytesIO(img_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL Image to numpy array
            img_array = np.array(img)
            
            # PIL uses RGB, OpenCV uses BGR, so convert
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            
        except Exception as e:
            logger.error(f"Error converting image: {e}")
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        # Extract landmarks
        landmarks = extract_landmarks(frame)
        
        if landmarks is not None and len(landmarks) == 63:
            try:
                logger.info(f"Landmarks extracted: {len(landmarks)} values")
                
                # Scale and predict
                landmarks_scaled = scaler.transform([landmarks])
                logger.info(f"Scaled landmarks: {landmarks_scaled.shape}")
                
                with torch.no_grad():
                    landmarks_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32).to(device)
                    output = model(landmarks_tensor)
                    pred_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
                    
                    # Get confidence for predicted class
                    probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
                    confidence = float(probabilities[pred_idx])
                
                predicted_letter = le.classes_[pred_idx]
                logger.info(f"Prediction: {predicted_letter}, Confidence: {confidence:.4f}")
                
                return jsonify({
                    'letter': predicted_letter,
                    'confidence': confidence
                })
            except Exception as e:
                logger.error(f"Error in prediction: {e}", exc_info=True)
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        else:
            logger.info(f"No hand detected or invalid landmarks: {landmarks}")
            return jsonify({
                'letter': 'No hand detected',
                'confidence': 0.0
            })
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ISL Sign Translator Web App")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
