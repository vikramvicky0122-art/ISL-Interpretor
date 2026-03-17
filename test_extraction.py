import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

test_image_path = r'D:\ISL dataset\Static gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals\ISL Images\1. Kids ISL images\Kids ISL images in Full Sleeves\English Alphabet\A\A (1).jpg'

img = cv2.imread(test_image_path)
print(f"Image loaded: {img is not None}")
print(f"Image shape: {img.shape if img is not None else 'None'}")

if img is not None:
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    print(f"Hands detected: {results.multi_hand_landmarks is not None}")
    if results.multi_hand_landmarks:
        print(f"Number of hands: {len(results.multi_hand_landmarks)}")
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        print(f"Landmarks extracted: {len(landmarks)}")
        print(f"First 5 landmarks: {landmarks[:5]}")
