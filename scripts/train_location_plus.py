import sys
import os
import sqlite3
import pandas as pd
import joblib
from xgboost import XGBRegressor

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join("data", "mlb_statcast.db")
MODEL_DIR = os.path.join("api", "engine")
MODEL_PATH = os.path.join(MODEL_DIR, "location_plus_model.pkl")

def train_location_model():
    print("🎯 Location+ 모델 학습 시작 (Target: xRV)...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. 학습 데이터 로드
    # 로케이션(plate_x, z)과 상황(balls, strikes, stand)이 핵심 Feature입니다.
    query = """
    SELECT plate_x, plate_z, balls, strikes, stand, delta_run_exp
    FROM statcast
    WHERE plate_x IS NOT NULL 
      AND plate_z IS NOT NULL
      AND delta_run_exp IS NOT NULL
      AND game_date >= '2023-01-01'
    LIMIT 100000
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    finally:
        conn.close()
    
    if df.empty:
        print("❌ 학습 데이터가 없습니다. DB를 확인하세요.")
        return

    print(f"📦 학습 데이터: {len(df):,} 건")
    
    # 2. 전처리
    # 좌타/우타(stand)를 숫자로 변환 (R=0, L=1)
    df['stand_code'] = df['stand'].apply(lambda x: 1 if x == 'L' else 0)
    
    # Feature & Target
    # 구위(속도, 무브먼트)는 제외하고 오직 '위치'와 '상황'만 봅니다.
    features = ['plate_x', 'plate_z', 'balls', 'strikes', 'stand_code']
    X = df[features]
    y = df['delta_run_exp']
    
    # 3. 모델 학습
    print("🧠 XGBoost 학습 중...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1)
    model.fit(X, y)
    
    # 4. 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    train_location_model()