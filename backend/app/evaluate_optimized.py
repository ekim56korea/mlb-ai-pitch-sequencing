import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import gc
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_pitch_classifier.json")
ENCODER_PATH = os.path.join(DATA_DIR, "xgb_encoders.pkl")

TARGET_YEAR = 2025
BATCH_SIZE = 10000  # 🌟 핵심: 1만 개씩 끊어서 처리 (메모리 보호)

def evaluate_optimized():
    print(f"🧪 Starting Optimized Evaluation on {TARGET_YEAR}...")
    
    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found.")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    artifacts = joblib.load(ENCODER_PATH)
    pitch_map = artifacts['pitch_map']
    feature_map = artifacts['feature_map']
    era_trends = artifacts['era_trends']
    
    inv_pitch_map = {v: k for k, v in pitch_map.items()}
    num_classes = len(pitch_map)

    # 2. 데이터 로드
    print("   📥 Loading Test Data...")
    con = duckdb.connect(DB_PATH, read_only=True)
    query = f"""
        SELECT 
            pitcher,
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
    
    valid_keys = list(pitch_map.keys())
    df = df[df['pitch_type'].isin(valid_keys)]
    
    fill_cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
    df[fill_cols] = df[fill_cols].fillna(0).astype(int)
    
    print(f"   📊 Loaded {len(df)} rows.")

    # 3. 투수 아스날 구축
    print("   🧠 Building Pitcher Arsenals...")
    pitcher_arsenals = df.groupby('pitcher')['pitch_type'].unique().apply(lambda x: set(x)).to_dict()
    pitcher_arsenals_idx = {
        pid: {pitch_map[p] for p in pitches if p in pitch_map} 
        for pid, pitches in pitcher_arsenals.items()
    }

    # 4. Feature Engineering
    trend = era_trends.get(TARGET_YEAR, era_trends[max(era_trends.keys())])
    for k, v in trend.items(): df[k] = v

    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    df['prev_pitch_type'] = df.groupby('game_pk')['pitch_type'].shift(1).fillna('None')
    df['prev_result'] = df.groupby('game_pk')['result_type'].shift(1).fillna('None')
    df['count_advantage'] = df['strikes'] - df['balls']
    df['is_scoring_pos'] = ((df['on_2b'] == 1) | (df['on_3b'] == 1)).astype(int)

    # Encoding
    y_true_all = df['pitch_type'].map(pitch_map).astype(int).values
    pitcher_ids_all = df['pitcher'].values
    
    df['stand'] = (df['stand'] == 'L').astype(int)
    df['p_throws'] = (df['p_throws'] == 'L').astype(int)
    df['prev_pitch_type'] = df['prev_pitch_type'].map(feature_map).fillna(9).astype(int)
    res_map = {'S': 0, 'B': 1, 'X': 2, 'None': 3}
    df['prev_result'] = df['prev_result'].map(res_map).fillna(3).astype(int)

    features = [
        'balls', 'strikes', 'outs_when_up', 'score_diff', 'inning',
        'on_1b', 'on_2b', 'on_3b', 'is_scoring_pos', 'count_advantage',
        'stand', 'p_throws',
        'prev_pitch_type', 'prev_result',
        'era_ff_rate', 'era_sweeper_rate', 'era_swing_rate', 'era_avg_velo'
    ]
    
    X_all = df[features]
    
    # 메모리 정리
    del df
    gc.collect()

    # 5. 🌟 배치 처리 (Batch Processing) 🌟
    print(f"   🔥 Predicting in batches of {BATCH_SIZE}...")
    
    hits_top1 = 0
    hits_top2 = 0
    total_samples = len(X_all)
    final_preds_all = [] # 리포트용

    # tqdm으로 진행 상황 시각화
    num_batches = int(np.ceil(total_samples / BATCH_SIZE))
    
    for i in tqdm(range(num_batches), desc="Batch Progress"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_samples)
        
        # 배치 슬라이싱
        X_batch = X_all.iloc[start_idx:end_idx]
        y_batch = y_true_all[start_idx:end_idx]
        pids_batch = pitcher_ids_all[start_idx:end_idx]
        
        # 예측 (배치 단위라 빠름)
        raw_probs_batch = model.predict_proba(X_batch)
        
        # 마스킹 및 정확도 계산
        batch_preds = []
        
        for j in range(len(X_batch)):
            pid = pids_batch[j]
            true_idx = y_batch[j]
            prob_row = raw_probs_batch[j]
            
            # Masking
            valid_indices = pitcher_arsenals_idx.get(pid, set())
            if valid_indices:
                mask = np.zeros(num_classes)
                for idx in valid_indices: mask[idx] = 1.0
                masked_prob = prob_row * mask
                s = masked_prob.sum()
                if s > 0: masked_prob /= s
            else:
                masked_prob = prob_row
            
            # Top-1
            pred_top1 = np.argmax(masked_prob)
            if pred_top1 == true_idx:
                hits_top1 += 1
            
            # Top-2
            top2_args = np.argsort(masked_prob)[-2:]
            if true_idx in top2_args:
                hits_top2 += 1
                
            batch_preds.append(pred_top1)
            
        final_preds_all.extend(batch_preds)
        
        # 배치 끝날 때마다 가비지 컬렉션 (안전장치)
        if i % 10 == 0:
            gc.collect()

    # 6. 결과 리포트
    acc_top1 = hits_top1 / total_samples
    acc_top2 = hits_top2 / total_samples
    
    print("\n" + "="*50)
    print(f"   🏆 OPTIMIZED EVALUATION REPORT ({TARGET_YEAR})")
    print("="*50)
    print(f"   ✅ Top-1 Accuracy: {acc_top1:.4f}")
    print(f"   ✅ Top-2 Accuracy: {acc_top2:.4f}")
    print("-" * 50)
    
    target_names = [inv_pitch_map[i] for i in range(num_classes)]
    print("\n📊 Classification Report (After Masking):")
    print(classification_report(y_true_all, final_preds_all, target_names=target_names, digits=3))

if __name__ == "__main__":
    evaluate_optimized()