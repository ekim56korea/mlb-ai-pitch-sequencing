import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import gc
from sklearn.utils.class_weight import compute_sample_weight # 🌟 핵심 추가

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_pitch_classifier.json")
ENCODER_PATH = os.path.join(DATA_DIR, "xgb_encoders.pkl")

# 유효 구종 (총 9개)
VALID_PITCHES = sorted(['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'ST', 'SV', 'FS'])

# 명시적 매핑
PITCH_MAP = {p: i for i, p in enumerate(VALID_PITCHES)}
FEATURE_PITCH_MAP = PITCH_MAP.copy()
FEATURE_PITCH_MAP['None'] = len(VALID_PITCHES)

def train_balanced():
    print(f"⚖️ Starting Balanced Training (Weighted for Rare Pitches)...")
    print(f"   Target DB: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print("❌ Database file not found!")
        return

    model = None 
    era_trends_archive = {} 

    # 2015 ~ 2025 연도별 루프
    for year in range(2015, 2026):
        print(f"\n📅 [Season {year}] Loading Data...")
        
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
                {year} as season
            FROM pitches
            WHERE pitch_type IS NOT NULL 
              AND balls IS NOT NULL
              AND CAST(STRFTIME(game_date, '%Y') AS INTEGER) = {year}
        """
        try:
            df = con.execute(query).df()
        except Exception as e:
            print(f"⚠️ Error loading {year}: {e}")
            con.close()
            continue
        con.close()
        
        if df.empty:
            print("⚠️ No data. Skipping.")
            continue

        # 필터링
        df = df[df['pitch_type'].isin(VALID_PITCHES)]
        
        # 결측치 처리
        fill_cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
        df[fill_cols] = df[fill_cols].fillna(0).astype(int)
        
        # ─── Era Context ───
        trend = {
            'era_ff_rate': (df['pitch_type'] == 'FF').mean(),
            'era_sweeper_rate': (df['pitch_type'] == 'ST').mean(),
            'era_avg_velo': df[df['pitch_type']=='FF']['release_speed'].mean(),
            'era_swing_rate': (df['result_type'] == 'X').mean()
        }
        if pd.isna(trend['era_avg_velo']): trend['era_avg_velo'] = 93.0
        for k, v in trend.items(): df[k] = v
        era_trends_archive[year] = trend

        # ─── Feature Engineering ───
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        df['prev_pitch_type'] = df.groupby('game_pk')['pitch_type'].shift(1).fillna('None')
        df['prev_result'] = df.groupby('game_pk')['result_type'].shift(1).fillna('None')
        df['count_advantage'] = df['strikes'] - df['balls']
        df['is_scoring_pos'] = ((df['on_2b'] == 1) | (df['on_3b'] == 1)).astype(int)

        # ─── Encoding ───
        y = df['pitch_type'].map(PITCH_MAP).astype(int)
        
        # 🌟 [핵심] 샘플 가중치 계산 (Class Balancing) 🌟
        # 빈도가 적은 구종(커브, 스플리터 등)에 더 높은 가중치 부여
        print("   ⚖️ Calculating Sample Weights...")
        weights = compute_sample_weight(class_weight='balanced', y=y)
        
        df['stand'] = (df['stand'] == 'L').astype(int)
        df['p_throws'] = (df['p_throws'] == 'L').astype(int)
        df['prev_pitch_type'] = df['prev_pitch_type'].map(FEATURE_PITCH_MAP).fillna(9).astype(int)
        res_map = {'S': 0, 'B': 1, 'X': 2, 'None': 3}
        df['prev_result'] = df['prev_result'].map(res_map).fillna(3).astype(int)

        features = [
            'balls', 'strikes', 'outs_when_up', 'score_diff', 'inning',
            'on_1b', 'on_2b', 'on_3b', 'is_scoring_pos', 'count_advantage',
            'stand', 'p_throws',
            'prev_pitch_type', 'prev_result',
            'era_ff_rate', 'era_sweeper_rate', 'era_swing_rate', 'era_avg_velo'
        ]
        X = df[features]
        
        # ─── 학습 ───
        print(f"   🔥 Training on {len(X)} samples with weights...")
        
        if model is None:
            model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=10,        # 깊이 약간 증가 (패턴 인식 강화)
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=len(VALID_PITCHES),
                n_jobs=-1,
                tree_method='hist'
            )
            # sample_weight 전달
            model.fit(X, y, sample_weight=weights, verbose=False)
        else:
            model.fit(X, y, sample_weight=weights, xgb_model=model, verbose=False)
            
        print(f"   ✅ Season {year} Finished.")
        del df, X, y, weights
        gc.collect()

    print("\n🎉 All Seasons Processed!")
    model.save_model(MODEL_PATH)
    
    final_artifacts = {
        'pitch_map': PITCH_MAP,
        'era_trends': era_trends_archive,
        'feature_map': FEATURE_PITCH_MAP
    }
    
    joblib.dump(final_artifacts, ENCODER_PATH)
    print(f"💾 Balanced Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_balanced()