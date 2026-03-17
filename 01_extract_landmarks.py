import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from tqdm import tqdm

def extract_landmarks(image_path, hands):
    """Extract hand landmarks from an image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    return None

def main():
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    # Base dataset path
    base_path = r'D:\ISL dataset\Static gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals\ISL Images'
    # Extract only Kids dataset for faster processing
    dataset_folders = [
        r'1. Kids ISL images\Kids ISL images in Full Sleeves\English Alphabet',
        r'1. Kids ISL images\Kids ISL images in Half Sleeves\English Alphabet',
    ]
    
    all_landmarks = []
    all_labels = []
    
    # Iterate through dataset
    for folder in dataset_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found")
            continue
        
        # Each letter has its own folder
        for letter in sorted(os.listdir(folder_path)):
            letter_path = os.path.join(folder_path, letter)
            if not os.path.isdir(letter_path):
                continue
            
            images = [f for f in os.listdir(letter_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in tqdm(images, desc=f'{folder}/{letter}', leave=False):
                img_path = os.path.join(letter_path, img_file)
                landmarks = extract_landmarks(img_path, hands)
                
                if landmarks and len(landmarks) == 63:  # 21 landmarks * 3 (x,y,z)
                    all_landmarks.append(str(landmarks))  # Convert to string for CSV
                    all_labels.append(letter)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame({
        'landmarks': all_landmarks,
        'label': all_labels
    })
    df.to_csv('data/landmarks.csv', index=False)
    print(f"Saved {len(df)} landmark samples to data/landmarks.csv")

if __name__ == '__main__':
    main()
