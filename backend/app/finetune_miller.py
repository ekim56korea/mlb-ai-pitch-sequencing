import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import duckdb
import joblib
import os
import sys

# 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model_attention import PitchLSTMAttention

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")
GLOBAL_MODEL_PATH = os.path.join(DATA_DIR, "pitch_lstm_attention.pth")
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")
SAVE_PATH = os.path.join(DATA_DIR, "models", "lstm_attention_695243.pth") # Miller ID

MILLER_ID = 695243
EPOCHS = 30
BATCH_SIZE = 32
SEQ_LENGTH = 5
LR = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MillerDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def get_miller_data():
    print(f"🔍 Fetching data for Mason Miller ({MILLER_ID})...")
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Mason Miller의 2023-2024 데이터 조회
    query = f"""
        SELECT *
        FROM pitches
        WHERE pitcher = {MILLER_ID}
        AND pitch_type IS NOT NULL
        ORDER BY game_date, inning, at_bat_number, pitch_number
    """
    df = con.execute(query).df()
    con.close()
    
    print(f"📊 Found {len(df)} pitches for Mason Miller.")
    return df

def preprocess_data(df, encoders):
    # 인코더 로드
    le_pitch = encoders['le_pitch']
    le_stand = encoders['le_stand']
    le_p_throws = encoders['le_p_throws']
    
    features = []
    labels = []
    
    # 데이터 순회하며 피처 벡터 생성
    # (주의: main.py의 predict_pitch와 순서가 100% 일치해야 함)
    
    # 투수 고유 Z-Score (Miller의 2024 평균값 하드코딩 - 학습용)
    # Vel, Spin, HB, IVB, Ext, RelH, RelS, Whiff
    z_scores = [1.8, 0.5, -0.2, 1.5, 0.8, -0.3, 0.2, 2.1] 
    
    # 타자 성향 (평균값 가정)
    batter_whiff = 0.25
    batter_k = 0.20

    for i, row in df.iterrows():
        try:
            # Target Label
            if row['pitch_type'] not in le_pitch.classes_:
                continue
            label = le_pitch.transform([row['pitch_type']])[0]
            
            # Categorical Encoding
            stand = 0 if row['stand'] == 'R' else 1
            p_throws = 0 if row['p_throws'] == 'R' else 1
            
            # Runners
            on_1b = 1.0 if row['on_1b'] > 0 else 0.0
            on_2b = 1.0 if row['on_2b'] > 0 else 0.0
            on_3b = 1.0 if row['on_3b'] > 0 else 0.0
            
            # Score Diff
            score_diff = (row['bat_score'] - row['fld_score'])
            
            # Vector Assembly (25 Features)
            vec = [
                row['inning'] / 9.0,
                row['balls'] / 4.0,
                row['strikes'] / 3.0,
                row['outs_when_up'] / 3.0,
                (score_diff + 10) / 20.0,
                on_1b, on_2b, on_3b,
                float(stand),
                float(p_throws),
                0.0, # pitch_number (normalized roughly) - simplified
                0.25, # TTO simplified
                0.0, # pitcher_count simplified
                batter_whiff,
                batter_k,
                *z_scores, # 8개
                row['sz_top'] if pd.notnull(row['sz_top']) else 3.5,
                row['sz_bot'] if pd.notnull(row['sz_bot']) else 1.6
            ]
            
            # 길이 보정 (25개 맞추기)
            if len(vec) < 25:
                vec.extend([0.0] * (25 - len(vec)))
                
            features.append(vec)
            labels.append(label)
            
        except Exception as e:
            continue

    return np.array(features), np.array(labels)

def create_sequences(features, labels, seq_length=5):
    X, y = [], []
    # 데이터가 끊기지 않는다고 가정하고 슬라이딩 윈도우 적용
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(labels[i + seq_length]) # 다음 공을 예측
    return np.array(X), np.array(y)

def finetune():
    print("\n⚔️  Fine-tuning Attention Model for Mason Miller  ⚔️\n")
    
    # 1. Load Resources
    encoders = joblib.load(ENCODER_PATH)
    input_size = encoders.get('input_size', 25)
    num_classes = encoders.get('num_classes', 10)
    
    # 2. Prepare Data
    df = get_miller_data()
    feats, targets = preprocess_data(df, encoders)
    X, y = create_sequences(feats, targets, SEQ_LENGTH)
    
    print(f"📚 Training Data Shape: {X.shape}")
    
    dataset = MillerDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Load Global Model
    print(f"📥 Loading Global Model from {GLOBAL_MODEL_PATH}")
    model = PitchLSTMAttention(input_size, 128, 2, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=DEVICE))
    model.train() # Set to training mode
    
    # 4. Training Setup
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("\n🚀 Starting Fine-tuning Loop...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs) # Attention returns (output, weights)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
            
    # 6. Save Fine-tuned Model
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Fine-tuning Complete!")
    print(f"💾 Model Saved to: {SAVE_PATH}")

if __name__ == "__main__":
    finetune()