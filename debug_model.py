import pickle
import numpy as np

# Check what the scaler expects
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Scaler input shape:", scaler.n_features_in_)
print("Scaler mean shape:", scaler.mean_.shape)
print("Scaler scale shape:", scaler.scale_.shape)

# Also check the CSV
import pandas as pd
import ast

df = pd.read_csv('data/landmarks.csv')
if len(df) > 0:
    first_landmark = ast.literal_eval(df['landmarks'].iloc[0])
    print(f"CSV first row landmark shape: {len(first_landmark)}")
    print(f"Total samples in CSV: {len(df)}")
