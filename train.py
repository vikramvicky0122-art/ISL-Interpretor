import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import ast

class ISLDataset(Dataset):
    def __init__(self, landmarks, labels):
        self.landmarks = torch.tensor(landmarks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]

class ISLClassifier(nn.Module):
    def __init__(self, input_dim=225, num_classes=26):
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

def main():
    df = pd.read_csv('data/landmarks.csv')
    df['landmarks'] = df['landmarks'].apply(ast.literal_eval)
    landmarks = np.array(df['landmarks'].tolist())
    
    # Map E1/E2 to E
    labels = df['label'].tolist()
    labels = ['E' if l in ['E1', 'E2'] else l for l in labels]
    
    le = LabelEncoder()
    le.fit(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    labels_encoded = le.transform(labels)
    
    scaler = StandardScaler()
    landmarks_scaled = scaler.fit_transform(landmarks)
    
    X_train, X_val, y_train, y_val = train_test_split(landmarks_scaled, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)
    
    train_dataset = ISLDataset(X_train, y_train)
    val_dataset = ISLDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ISLClassifier(input_dim=landmarks.shape[1]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for landmarks_b, labels_b in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            landmarks_b, labels_b = landmarks_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            outputs = model(landmarks_b)
            loss = criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for landmarks_b, labels_b in val_loader:
                landmarks_b, labels_b = landmarks_b.to(device), labels_b.to(device)
                outputs = model(landmarks_b)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels_b.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/isl_classifier.pth')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/le.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.title('Train Loss')
    plt.subplot(1,2,2)
    plt.plot(val_accuracies)
    plt.title('Val Accuracy')
    plt.savefig('models/training_plots.png')
    print('Model, scaler, le saved to models/')

if __name__ == '__main__':
    main()
