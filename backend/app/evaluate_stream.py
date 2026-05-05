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
# 🌟 [수정] 청크 사이즈 대폭 축소 (5만 -> 5천)
# 속도는 조금 느려지지만, 절대 죽지 않습니다.
CHUNK_SIZE = 5000 

def evaluate_stream_safe():
    print(f"🧪 Starting Safe Stream Evaluation ({START_YEAR} - {END_YEAR})...")
    print(f"   🛡️ Chunk Size: {CHUNK_SIZE} (Ultra-Safe Mode)")

    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found.")
        return

    # 1. 모델 로드 (전역 메모리 절약을 위해 함수 내 로드 권장)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    artifacts = joblib.load(ENCODER_PATH)
    pitch_map = artifacts['pitch_map']
    feature_map = artifacts['feature_map']
    era_trends = artifacts['era_trends']
    
    num_classes = len(pitch_map)
    valid_keys = list(pitch_map.keys())

    # 2. 아스날 사전 구축 (가장 가볍게)
    print("   🧠 Pre-scanning Arsenals...")
    con = duckdb.connect(DB_PATH, read_only=True)
    arsenal_query = f"""
        SELECT DISTINCT pitcher, pitch_type 
        FROM pitches 
        WHERE CAST(STRFTIME(game_date, '%Y') AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
    """
    arsenal_df = con.execute(arsenal_query).df()
    
    pitcher_arsenals_idx = {}
    for pid, group in arsenal_df.groupby('pitcher'):
        indices = {pitch_map[p] for p in group['pitch_type'].values if p in pitch_map}
        if indices: pitcher_arsenals_idx[pid] = indices
            
    del arsenal_df, arsenal_query
    gc.collect()

    total_samples = 0
    total_hits_top1 = 0
    total_hits_top2 = 0

    # 3. 스트리밍 시작
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n📅 Season {year}...")
        
        count_query = f"SELECT COUNT(*) FROM pitches WHERE CAST(STRFTIME(game_date, '%Y') AS INTEGER) = {year} AND pitch_type IS NOT NULL AND balls IS NOT NULL"
        total_rows = con.execute(count_query).fetchone()[0]
        
        # tqdm 진행바
        with tqdm(total=total_rows, desc=f"Evaluating {year}") as pbar:
            offset = 0
            while offset < total_rows:
                # 4. 아주 작은 단위로 데이터 로드
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
                    WHERE pitch_type IS NOT NULL AND balls IS NOT NULL
                      AND CAST(STRFTIME(game_date, '%Y') AS INTEGER) = {year}
                    LIMIT {CHUNK_SIZE} OFFSET {offset}
                """
                df = con.execute(query).df()
                
                if df.empty: break
                
                # 전처리
                df = df[df['pitch_type'].isin(valid_keys)].copy()
                if df.empty:
                    offset += CHUNK_SIZE
                    pbar.update(len(df)) # update with 0 or actual read count if filtered
                    continue

                rows_count = len(df)
                
                # 결측치
                cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
                df[cols] = df[cols].fillna(0).astype(int)

                # Era Context
                trend = era_trends.get(year, era_trends[2024])
                for k, v in trend.items(): df[k] = v

                # Encoding & Features
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

                # 예측
                raw_probs = model.predict_proba(X)
                
                # 채점 (벡터화 없이 순회 - 메모리 스파이크 방지)
                hits1 = 0
                hits2 = 0
                for i in range(rows_count):
                    pid = pids[i]
                    t_idx = y_true[i]
                    prob = raw_probs[i]
                    
                    # Masking
                    valid = pitcher_arsenals_idx.get(pid, set())
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
                
                # 🌟 [핵심] 메모리 강제 해제
                offset += CHUNK_SIZE
                pbar.update(rows_in_chunk)
                
                del df, X, raw_probs, y_true, pids
                gc.collect() # 매번 수행 (속도보다 안정성 우선)

    con.close()

    if total_samples > 0:
        print(f"\n✅ Final Top-1: {total_hits_top1/total_samples:.4f}")
        print(f"✅ Final Top-2: {total_hits_top2/total_samples:.4f}")

if __name__ == "__main__":
    evaluate_stream_safe()