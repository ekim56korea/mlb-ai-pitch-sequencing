import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import os
import duckdb
import gc
import glob
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 🆕 새로 만든 Attention 모델 임포트
from app.model_attention import PitchLSTMAttention

# ─── 설정 및 경로 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
# 모델 파일 이름 변경 (구분하기 위해)
ATTENTION_MODEL_PATH = os.path.join(DATA_DIR, "pitch_lstm_attention.pth")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints_attn") # 체크포인트 경로 분리
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ⚙️ 하이퍼파라미터
DB_CHUNK_SIZE = 10000 
BATCH_SIZE = 256       
TOTAL_EPOCHS = 5
INPUT_SIZE = 25

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB file not found at: {DB_PATH}")
    con = duckdb.connect(DB_PATH)
    con.execute("PRAGMA memory_limit='6GB'")
    return con

def get_latest_checkpoint(model, optimizer):
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return 0
    latest_cp = max(checkpoints, key=os.path.getctime)
    print(f"🔄 Resuming Attention Training from: {latest_cp}", flush=True)
    checkpoint = torch.load(latest_cp, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] + 1

def train_attention_model(start_year=2015):
    """Attention 모델 학습 루프"""
    print(f"🧠 Starting Attention Model Training (Year: {start_year})...", flush=True)

    con = get_db_connection()
    # 뷰 생성 함수는 기존 train.py의 로직을 공유하거나 여기서 다시 정의 필요 
    # (편의상 이미 뷰가 생성되어 있다고 가정)
    
    if os.path.exists(ENCODER_PATH):
        encoders = joblib.load(ENCODER_PATH)
    else:
        print("❌ Encoders not found. Run global training first.")
        return

    # 🆕 Attention 모델 초기화
    model = PitchLSTMAttention(INPUT_SIZE, 128, 2, encoders['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = get_latest_checkpoint(model, optimizer)
    
    total_rows = con.execute(f"SELECT COUNT(*) FROM pitches WHERE game_date >= '{start_year}-01-01'").fetchone()[0]
    print(f"🚀 Training Attention Model on {total_rows:,} rows", flush=True)

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\n🌟 Epoch {epoch+1} Start", flush=True)
        
        for offset in range(0, total_rows, DB_CHUNK_SIZE):
            gc.collect()
            
            # (쿼리 로직은 train.py와 100% 동일하므로 생략 - 그대로 복사해서 사용)
            # ... Query & Preprocessing Code Here ...
            # 실제 파일 작성 시에는 train.py의 쿼리 부분을 그대로 복붙하세요.
            
            # --- 데모용 임시 코드 (실제론 train.py 로직 복붙) ---
            # X_seq, y_seq = ... 
            # -----------------------------------------------

            if 'tensor_x' not in locals(): continue # 쿼리 로직 없을 때 방지

            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                
                # 🆕 핵심 변경 사항: 리턴값 언패킹
                # Attention 모델은 (output, weights)를 반환함
                outputs, _ = model(xb) 
                
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
            
            print(f"\r   Processing {min((offset+DB_CHUNK_SIZE), total_rows)/total_rows*100:.1f}%", end="", flush=True)

        # 체크포인트 저장
        save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        
        # 모델 저장
        torch.save(model.state_dict(), ATTENTION_MODEL_PATH)
        print(f"\n💾 Saved Attention Epoch {epoch+1}", flush=True)

    con.close()
    print("🎉 Attention Model Training Complete!", flush=True)
# train_attention.py 파일 맨 끝에 추가

if __name__ == "__main__":
    # 2015년부터 학습 시작 (실행 트리거)
    train_attention_model(start_year=2015)