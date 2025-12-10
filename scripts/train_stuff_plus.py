import sys
import os
import sqlite3
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join("data", "mlb_statcast.db")
MODEL_PATH = os.path.join("api", "engine", "stuff_plus_model.pkl")

def train_stuff_model():
    print("⚾️ Stuff+ 모델 학습 시작 (XGBoost)...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. 학습 데이터 로드 (투구 물리량 + 결과)
    # delta_run_exp: 해당 투구로 인한 득점 기대치 변화량 (xRV의 라벨)
    print("📥 데이터 로드 중...")
    query = """
    SELECT release_speed, release_spin_rate, pfx_x, pfx_z, release_extension, delta_run_exp
    FROM statcast
    WHERE release_speed IS NOT NULL 
      AND delta_run_exp IS NOT NULL
      AND game_date >= '2024-01-01'
    LIMIT 100000
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    finally:
        conn.close()
        
    print(f"📦 학습 데이터: {len(df):,} 건")
    
    # 2. Feature & Target
    X = df[['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_extension']]
    y = df['delta_run_exp'] # 목표: 투구의 물리적 스펙으로 xRV를 예측하는 것
    
    # 3. 모델 학습
    print("🧠 모델 학습 중...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1)
    model.fit(X, y)
    
    # 4. 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    train_stuff_model()