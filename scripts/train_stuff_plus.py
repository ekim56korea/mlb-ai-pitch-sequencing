import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api.engine.preprocessor import DataPreprocessor

DB_PATH = os.path.join("data", "mlb_statcast.db")
MODEL_PATH = os.path.join("api", "engine", "stuff_plus_model.pkl")

def train_advanced_stuff_model():
    print("⚾️ Advanced Stuff+ Model Training (v5.0)...")
    
    preprocessor = DataPreprocessor()
    conn = sqlite3.connect(DB_PATH)
    
    # 1. 데이터 로드
    print("📥 Loading Data...")
    query = """
    SELECT release_speed, release_spin_rate, pfx_x, pfx_z, release_extension, delta_run_exp
    FROM statcast
    WHERE release_speed IS NOT NULL 
      AND delta_run_exp IS NOT NULL
      AND game_date >= '2023-01-01'
    LIMIT 200000
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return
    finally:
        conn.close()
    
    # 2. 전처리 및 특성 공학 (Process & Feature Engineering)
    print(f"   Raw Data: {len(df):,} rows")
    df = preprocessor.clean_data(df)
    df = preprocessor.engineer_features(df)
    print(f"   Clean & Engineered Data: {len(df):,} rows")
    
    # 학습에 사용할 Feature 정의 (파생 변수 포함)
    features = [
        'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_extension',
        'effective_velo', 'velo_pfx_z_interaction', 'movement_per_spin'
    ]
    target = 'delta_run_exp'
    
    X = df[features]
    y = df[target]
    
    # 3. K-Fold 교차 검증 (Validation)
    print("\n🧐 5-Fold Cross Validation 진행 중...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
    
    # MAE (Mean Absolute Error)로 평가 (낮을수록 좋음)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kfold)
    
    mae_scores = -scores
    print(f"   📊 CV MAE Scores: {mae_scores}")
    print(f"   ✅ Average MAE: {np.mean(mae_scores):.4f} (Standard Deviation: {np.std(mae_scores):.4f})")
    
    # 4. 최종 모델 학습 (전체 데이터)
    print("\n🚀 Final Model Training...")
    model.fit(X, y)
    
    # 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 모델 저장 완료: {MODEL_PATH}")
    print("   -> 이제 모델은 '체감 구속'과 '회전 효율'을 이해합니다.")

if __name__ == "__main__":
    train_advanced_stuff_model()