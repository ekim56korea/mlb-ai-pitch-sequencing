import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys

# 🌟 [핵심 수정 1] 상위 폴더(프로젝트 루트)를 파이썬 경로에 추가
# 이렇게 해야 'app' 폴더 안에 있는 model.py를 불러올 수 있습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.model import PitchLSTM  # ✅ 이제 app 패키지 안에서 찾습니다.

# ─── 설정 및 경로 수정 ───
# 🌟 [핵심 수정 2] 데이터가 'data' 폴더에 있으므로 경로를 명확히 지정
DATA_DIR = os.path.join(parent_dir, "data")
DB_FILE = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "pitch_lstm.pth")
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")

SEQ_LENGTH = 5  
BATCH_SIZE = 64
EPOCHS = 10     # 학습 횟수 조금 늘림
LEARNING_RATE = 0.001

# ─── 1. 데이터 로드 및 전처리 ───
def load_data():
    if not os.path.exists(DB_FILE):
        print(f"❌ Database not found at: {DB_FILE}")
        print("   Please run 'python scripts/setup_db.py' first.")
        sys.exit(1)

    print(f"🦆 Connecting to DuckDB at {DB_FILE}...")
    con = duckdb.connect(DB_FILE, read_only=True)
    
    # 최근 3년치 데이터 조회
    query = """
        SELECT game_pk, at_bat_number, pitch_number, 
               pitch_type, release_speed, plate_x, plate_z, 
               balls, strikes, stand
        FROM pitches
        WHERE game_date >= '2022-01-01'
          AND pitch_type IS NOT NULL 
          AND release_speed IS NOT NULL
          AND plate_x IS NOT NULL
        ORDER BY game_pk, at_bat_number, pitch_number
    """
    print("📊 Executing Query...")
    df = con.execute(query).df()
    con.close()
    
    print(f"✅ Loaded {len(df):,} pitches.")
    return df

# ─── 2. 시퀀스 데이터 생성 (Dataset) ───
class PitchDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_sequences(df, le_pitch, le_stand):
    # 인코딩
    df['pitch_code'] = le_pitch.fit_transform(df['pitch_type'])
    df['stand_code'] = le_stand.fit_transform(df['stand'])
    
    # 정규화
    scaler = StandardScaler()
    features = ['release_speed', 'plate_x', 'plate_z', 'balls', 'strikes', 'stand_code']
    df[features] = scaler.fit_transform(df[features])
    
    data = df[features + ['pitch_code']].values
    
    X, y = [], []
    
    print("✂️ Creating sequences...")
    # 간단한 시퀀스 생성 (연속된 데이터 가정)
    # 데이터가 너무 많으면 메모리 부족할 수 있으므로 최대 10만개 샘플링하거나 끊어서 처리
    limit = min(len(data), 200000) # 데모용으로 20만개만 사용 (속도 향상)
    
    for i in range(limit - SEQ_LENGTH):
        seq_features = data[i : i + SEQ_LENGTH, :-1] # (5, 6)
        target = data[i + SEQ_LENGTH, -1] # 다음 공
        
        X.append(seq_features)
        y.append(target)
        
    return np.array(X), np.array(y), scaler

# ─── 3. 메인 학습 루프 ───
def train():
    df = load_data()
    
    if df.empty:
        print("❌ No data found. Check your database.")
        return

    le_pitch = LabelEncoder()
    le_stand = LabelEncoder()
    
    X, y, scaler = create_sequences(df, le_pitch, le_stand)
    
    if len(X) == 0:
        print("❌ Not enough data to create sequences.")
        return

    dataset = PitchDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 모델 초기화
    input_size = X.shape[2]
    num_classes = len(le_pitch.classes_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Training on {device}")
    
    model = PitchLSTM(input_size, 128, 2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("🔥 Start Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (seqs, labels) in enumerate(dataloader):
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")
        
    # 저장
    print(f"💾 Saving Model to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)
    
    meta_data = {
        'le_pitch': le_pitch,
        'le_stand': le_stand,
        'scaler': scaler,
        'input_size': input_size,
        'num_classes': num_classes
    }
    joblib.dump(meta_data, ENCODER_PATH)
    print("🎉 Training Complete!")

if __name__ == "__main__":
    train()