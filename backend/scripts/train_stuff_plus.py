import duckdb
import pandas as pd
import xgboost as xgb
import joblib
import os
import sys

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(current_dir), "data")
DB_FILE = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "stuff_plus_model.pkl")

def train_stuff_model():
    print("🦆 Loading Data for Stuff+ Training...")
    con = duckdb.connect(DB_FILE, read_only=True)
    
    # 1. 학습 데이터 조회
    # 결과(delta_run_exp)가 있는 데이터만 가져옵니다.
    # Stuff+는 '공의 물리적 특성'만 봅니다. (위치 정보 plate_x, z 제외!)
    query = """
        SELECT 
            pitch_type, 
            release_speed, 
            pfx_x, 
            pfx_z, 
            release_extension, 
            release_spin_rate,
            delta_run_exp as target
        FROM pitches
        WHERE game_date >= '2022-01-01'
          AND delta_run_exp IS NOT NULL
          AND release_speed IS NOT NULL
          AND pfx_x IS NOT NULL
    """
    df = con.execute(query).df()
    con.close()
    
    # 2. 전처리
    # 구종(pitch_type)은 범주형이므로 One-Hot Encoding 또는 Label Encoding 필요
    # XGBoost는 숫자만 받으므로 변환
    df['pitch_type'] = df['pitch_type'].astype('category')
    
    # Features (X) & Target (y)
    X = df[['pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'release_extension', 'release_spin_rate']]
    y = df['target']
    
    # 3. 모델 학습 (XGBoost Regressor)
    # 목표: 물리적 제원(X)을 넣으면 -> 기대 득점 가치(y)를 예측
    print("🧠 Training XGBoost Model (This represents 'Stuff')...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        enable_categorical=True, # 카테고리형 데이터 자동 처리 (XGB 최신버전 기능)
        tree_method='hist'       # 속도 최적화
    )
    
    model.fit(X, y)
    
    # 4. 저장
    print(f"💾 Saving Stuff+ Model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("✅ Stuff+ Model Training Complete!")

if __name__ == "__main__":
    train_stuff_model()