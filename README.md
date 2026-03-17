# ISL Sign Translater

Indian Sign Language (English Alphabet) image dataset to text/voice using Mediapipe landmarks, PyTorch.

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Option 1: Web UI (Recommended)
```
python app.py
```
Open http://localhost:5000 in your browser and watch the live predictions!

### Option 2: Webcam Console App
```
python inference.py
```
Real-time sign recognition with text-to-speech output in console window.

### Training & Data Processing
1. Extract landmarks: `python 01_extract_landmarks.py` (processes all kids/teen/adult full/half sleeves → data/landmarks.csv)
2. Train model: `python train.py` (→ models/isl_classifier.pth, scaler/le.pkl, plots)

## Dataset
D:/ISL dataset/.../ISL Images/[1-3].* Images/[Full|Half] Sleeves/English Alphabet/[A-Z]/*.jpg

Labels A-Z (E1/E2 → E).

## Features
- Real-time hand landmark detection using Mediapipe
- PyTorch-based classification model (98.58% accuracy)
- Beautiful web interface with live video feed
- Confidence scoring for predictions
- Text-to-speech output

Enjoy signing!

