import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import gc

# ─── 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_pitch_classifier.json")
ENCODER_PATH = os.path.join(DATA_DIR, "xgb_encoders.pkl")

VALID_PITCHES = sorted(['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'ST', 'SV', 'FS'])

# 명시적 매핑
PITCH_MAP = {p: i for i, p in enumerate(VALID_PITCHES)}
FEATURE_PITCH_MAP = PITCH_MAP.copy()
FEATURE_PITCH_MAP['None'] = len(VALID_PITCHES)

# 🌟 [최종 수정] 황금 밸런스 가중치 (Golden Ratio Weights)
# 이전 결과 분석: FF(저조), FC(저조) / CH(과다), ST(과다)
CUSTOM_WEIGHTS = {
    'FF': 1.25,        # ▲ 상향 (기본 1.0 -> 1.25) : 직구 추천 빈도 회복
    'SI': 1.5,         # - 유지
    'FC': 2.0,         # ▲ 상향 (기본 1.5 -> 2.0) : 커터 예측력 강화
    'SL': 2.0,         # - 유지 (현재 Recall 0.45로 양호)
    'CH': 2.2,         # ▼ 하향 (기본 3.0 -> 2.2) : 체인지업 과다 추천 억제
    'CU': 2.8,         # ▼ 소폭 하향
    'ST': 3.0,         # ▼ 하향 (기본 4.0 -> 3.0) : 스위퍼 과다 추천 억제
    'FS': 3.5,         # ▼ 소폭 하향
    'SV': 4.0          # ▼ 소폭 하향
}

def train_finetuned():
    print(f"⚖️ Starting Fine-Tuned Training (Golden Ratio)...")
    print(f"   Weights: {CUSTOM_WEIGHTS}")
    
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
        
        # 🌟 커스텀 가중치 적용
        weights = df['pitch_type'].map(CUSTOM_WEIGHTS).values
        
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
        print(f"   🔥 Training on {len(X)} samples...")
        
        if model is None:
            model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=9,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=len(VALID_PITCHES),
                n_jobs=-1,
                tree_method='hist'
            )
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
    print(f"💾 Finetuned Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_finetuned()