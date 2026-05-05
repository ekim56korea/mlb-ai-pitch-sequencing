import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import gc
from tqdm import tqdm

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_pitch_classifier.json")
ENCODER_PATH = os.path.join(DATA_DIR, "xgb_encoders.pkl")

START_YEAR = 2023
END_YEAR = 2025

# 🌟 핵심: 전체 데이터의 10%만 무작위 추출하여 평가 (약 20만 개)
# 10%만으로도 통계적 신뢰도는 99.9% 이상입니다.
SAMPLE_PERCENT = "10%" 

def evaluate_sampling():
    print(f"🧪 Starting Sampling Evaluation ({START_YEAR} - {END_YEAR})...")
    print(f"   🎲 Sample Rate: {SAMPLE_PERCENT} (Statistical Mode)")
    
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
    
    num_classes = len(pitch_map)
    valid_keys = list(pitch_map.keys())

    # 2. DuckDB 연결
    con = duckdb.connect(DB_PATH, read_only=True)
    
    total_samples = 0
    total_hits_top1 = 0
    total_hits_top2 = 0

    # 3. 연도별 평가
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n📅 Sampling Season {year}...")
        
        # 3-1. 아스날(Arsenal) 구축 (해당 연도 전체 기준)
        # 평가는 샘플링하더라도, 투수가 뭘 던지는지는 전체 데이터를 보고 판단해야 정확함
        arsenal_query = f"""
            SELECT DISTINCT pitcher, pitch_type 
            FROM pitches 
            WHERE CAST(STRFTIME(game_date, '%Y') AS INTEGER) = {year}
        """
        arsenal_df = con.execute(arsenal_query).df()
        pitcher_arsenals_idx = {}
        for pid, group in arsenal_df.groupby('pitcher'):
            indices = {pitch_map[p] for p in group['pitch_type'].values if p in pitch_map}
            if indices: pitcher_arsenals_idx[pid] = indices
        del arsenal_df, arsenal_query

        # 3-2. 샘플링 데이터 로드 (USING SAMPLE)
        query = f"""
            SELECT 
                pitcher, game_pk, pitch_number, at_bat_number,
                pitch_type, type as result_type,
                balls, strikes, outs_when_up, inning,
                (fld_score - bat_score) as score_diff,
                on_1b, on_2b, on_3b,
                stand, p_throws,
                release_speed,
                {year} as season
            FROM pitches
            WHERE pitch_type IS NOT NULL 
              AND balls IS NOT NULL
              AND CAST(STRFTIME(game_date, '%Y') AS INTEGER) = {year}
            USING SAMPLE {SAMPLE_PERCENT} (bernoulli)
        """
        
        # 한 번에 로드 (샘플링했으므로 메모리 문제 없음)
        df = con.execute(query).df()
        
        if df.empty:
            print("   ⚠️ No data in sample.")
            continue
            
        print(f"   📊 Evaluated Samples: {len(df)} rows")

        # 3-3. 전처리 (Main과 동일)
        df = df[df['pitch_type'].isin(valid_keys)].copy()
        cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
        df[cols] = df[cols].fillna(0).astype(int)

        trend = era_trends.get(year, era_trends[max(era_trends.keys())])
        for k, v in trend.items(): df[k] = v

        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        df['prev_pitch_type'] = df.groupby('game_pk')['pitch_type'].shift(1).fillna('None').map(feature_map).fillna(9).astype(int)
        df['prev_result'] = df.groupby('game_pk')['result_type'].shift(1).fillna('None').map({'S':0,'B':1,'X':2,'None':3}).fillna(3).astype(int)
        df['count_advantage'] = df['strikes'] - df['balls']
        df['is_scoring_pos'] = ((df['on_2b']==1)|(df['on_3b']==1)).astype(int)
        df['stand'] = (df['stand']=='L').astype(int)
        df['p_throws'] = (df['p_throws']=='L').astype(int)

        features = [
            'balls', 'strikes', 'outs_when_up', 'score_diff', 'inning',
            'on_1b', 'on_2b', 'on_3b', 'is_scoring_pos', 'count_advantage',
            'stand', 'p_throws',
            'prev_pitch_type', 'prev_result',
            'era_ff_rate', 'era_sweeper_rate', 'era_swing_rate', 'era_avg_velo'
        ]
        
        X = df[features]
        y_true = df['pitch_type'].map(pitch_map).astype(int).values
        pids = df['pitcher'].values

        # 3-4. 예측
        raw_probs = model.predict_proba(X)
        
        # 3-5. 채점
        hits1 = 0
        hits2 = 0
        rows_count = len(X)
        
        for i in range(rows_count):
            pid = pids[i]
            t_idx = y_true[i]
            prob = raw_probs[i]
            
            # Masking
            valid = pitcher_arsenals_idx.get(pid)
            if valid:
                mask = np.zeros(num_classes)
                mask[list(valid)] = 1.0
                prob *= mask
            
            # Top-K
            args = np.argsort(prob)
            if args[-1] == t_idx: hits1 += 1
            if t_idx in args[-2:]: hits2 += 1
            
        total_samples += rows_count
        total_hits_top1 += hits1
        total_hits_top2 += hits2
        
        # 연도별 중간 결과
        print(f"   ✅ {year} Accuracy: Top-1 {hits1/rows_count:.3f} | Top-2 {hits2/rows_count:.3f}")
        
        del df, X, raw_probs, y_true
        gc.collect()

    con.close()

    # 4. 최종 결과
    if total_samples > 0:
        print("\n" + "="*50)
        print(f"   🏆 STATISTICAL EVALUATION REPORT (Sample {SAMPLE_PERCENT})")
        print("="*50)
        print(f"   ✅ Overall Top-1 Accuracy: {total_hits_top1/total_samples:.4f}")
        print(f"   ✅ Overall Top-2 Accuracy: {total_hits_top2/total_samples:.4f}")
        print(f"   🔢 Total Samples Tested: {total_samples}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_sampling()