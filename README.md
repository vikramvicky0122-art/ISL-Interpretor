# ISL Sign Translater

Indian Sign Language (English Alphabet) image dataset to text/voice using Mediapipe landmarks, PyTorch.

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. Extract landmarks: `python 01_extract_landmarks.py` (processes all kids/teen/adult full/half sleeves → data/landmarks.csv)
2. Train model: `python train.py` (→ models/isl_classifier.pth, scaler/le.pkl, plots)
3. Inference: `python inference.py` (webcam: sign letter → text + voice)

## Dataset
D:/ISL dataset/.../ISL Images/[1-3].* Images/[Full|Half] Sleeves/English Alphabet/[A-Z]/*.jpg

Labels A-Z (E1/E2 → E).

Enjoy signing!
