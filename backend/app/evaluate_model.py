import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_pitch_classifier.json")
ENCODER_PATH = os.path.join(DATA_DIR, "xgb_encoders.pkl")

# 평가 대상 연도 (가장 최신 데이터로 검증)
TARGET_YEAR = 2025

def evaluate():
    print(f"🧪 Starting Evaluation on {TARGET_YEAR} Season Data...")
    
    # 1. 모델 및 아티팩트 로드
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("❌ Model or Artifacts not found.")
        return

    print("   📥 Loading Model & Artifacts...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    artifacts = joblib.load(ENCODER_PATH)
    pitch_map = artifacts['pitch_map']         # {'FF': 0, ...}
    feature_map = artifacts['feature_map']     # {'FF': 0, ..., 'None': 9}
    era_trends = artifacts['era_trends']       # {2015: {...}, ...}
    
    # Reverse Map (인덱스 -> 구종 이름)
    inv_pitch_map = {v: k for k, v in pitch_map.items()}
    
    # 2. 검증 데이터 로드 (2025년)
    if not os.path.exists(DB_PATH):
        print("❌ Database not found.")
        return

    con = duckdb.connect(DB_PATH, read_only=True)
    query = f"""
        SELECT 
            game_pk, pitch_number, at_bat_number,
            pitch_type, type as result_type,
            balls, strikes, outs_when_up, inning,
            (fld_score - bat_score) as score_diff,
            on_1b, on_2b, on_3b,
            stand, p_throws,
            release_speed,
            {TARGET_YEAR} as season
        FROM pitches
        WHERE pitch_type IS NOT NULL 
          AND balls IS NOT NULL
          AND CAST(STRFTIME(game_date, '%Y') AS INTEGER) = {TARGET_YEAR}
    """
    df = con.execute(query).df()
    con.close()
    
    # 유효 구종만 필터링 (학습 때와 동일하게)
    valid_pitches = list(pitch_map.keys())
    df = df[df['pitch_type'].isin(valid_pitches)]
    
    # 결측치 처리
    fill_cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
    df[fill_cols] = df[fill_cols].fillna(0).astype(int)
    
    print(f"   📊 Loaded {len(df)} rows for evaluation.")

    # 3. Feature Engineering (학습/Main과 동일 로직)
    
    # Era Context (저장된 트렌드 사용)
    trend = era_trends.get(TARGET_YEAR)
    if not trend:
        print(f"⚠️ No era trend found for {TARGET_YEAR}, using fallback.")
        trend = era_trends[max(era_trends.keys())]
        
    for k, v in trend.items():
        df[k] = v

    # Lag Features
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    df['prev_pitch_type'] = df.groupby('game_pk')['pitch_type'].shift(1).fillna('None')
    df['prev_result'] = df.groupby('game_pk')['result_type'].shift(1).fillna('None')
    
    df['count_advantage'] = df['strikes'] - df['balls']
    df['is_scoring_pos'] = ((df['on_2b'] == 1) | (df['on_3b'] == 1)).astype(int)

    # Encoding
    # Target
    y_true = df['pitch_type'].map(pitch_map).astype(int)
    
    # Features
    df['stand'] = df['stand'].apply(lambda x: 1 if x == 'L' else 0)
    df['p_throws'] = df['p_throws'].apply(lambda x: 1 if x == 'L' else 0)
    
    df['prev_pitch_type'] = df['prev_pitch_type'].map(feature_map).fillna(9).astype(int)
    
    res_map = {'S': 0, 'B': 1, 'X': 2, 'None': 3}
    df['prev_result'] = df['prev_result'].map(res_map).fillna(3).astype(int)

    # Feature List (순서 중요!)
    features = [
        'balls', 'strikes', 'outs_when_up', 'score_diff', 'inning',
        'on_1b', 'on_2b', 'on_3b', 'is_scoring_pos', 'count_advantage',
        'stand', 'p_throws',
        'prev_pitch_type', 'prev_result',
        'era_ff_rate', 'era_sweeper_rate', 'era_swing_rate', 'era_avg_velo'
    ]
    
    X = df[features]
    
    # 4. 예측 수행
    print(f"   🔥 Predicting...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # 5. 성능 지표 계산
    acc = accuracy_score(y_true, y_pred)
    
    # Top-2 Accuracy Calculation
    # 예측 확률 중 상위 2개 인덱스 추출
    top2_indices = np.argsort(y_proba, axis=1)[:, -2:]
    # 실제 정답이 상위 2개 안에 포함되는지 확인
    top2_hits = [y_true.iloc[i] in top2_indices[i] for i in range(len(y_true))]
    top2_acc = np.mean(top2_hits)

    print("\n" + "="*40)
    print(f"   🏆 EVALUATION REPORT ({TARGET_YEAR})")
    print("="*40)
    print(f"   ✅ Top-1 Accuracy: {acc:.4f} (Expected: ~0.55+)")
    print(f"   ✅ Top-2 Accuracy: {top2_acc:.4f} (Expected: ~0.80+)")
    print("-" * 40)
    
    # 상세 리포트
    target_names = [inv_pitch_map[i] for i in range(len(pitch_map))]
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

if __name__ == "__main__":
    evaluate()