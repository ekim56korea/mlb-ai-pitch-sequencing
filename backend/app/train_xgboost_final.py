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

# 🌟 [Ultimate Weights] 최적의 균형점
# 직구(FF)를 중심에 두되, 변화구(CH, CU)의 가치를 충분히 인정하여 Top-2 방어
CUSTOM_WEIGHTS = {
    'FF': 1.6,         # ▲ 기준점 (너무 높지 않게 설정하여 유연성 확보)
    'SI': 1.4,         # ▼ 직구보다 약간 낮춰서 혼동 방지
    'FC': 1.4,         # ▼ 커터도 직구와 구분
    'SL': 2.0,         # - 슬라이더는 제2구종으로 매우 중요
    'CH': 2.5,         # ▲ 체인지업은 Top-2 정확도의 핵심
    'CU': 2.5,         # ▲ 커브도 동일
    'ST': 3.5,         # ▲ 희귀 구종은 높은 가중치 유지 (틀리면 손해 큼)
    'FS': 3.5,
    'SV': 3.5
}

def train_ultimate():
    print(f"🚀 Starting Ultimate Training (Balanced & Robust)...")
    print(f"   Weights: {CUSTOM_WEIGHTS}")
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found.")
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
        df = con.execute(query).df()
        con.close()
        
        if df.empty: continue

        # 필터링
        df = df[df['pitch_type'].isin(VALID_PITCHES)]
        
        # 결측치
        cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
        df[cols] = df[cols].fillna(0).astype(int)
        
        # Era Context
        trend = {
            'era_ff_rate': (df['pitch_type'] == 'FF').mean(),
            'era_sweeper_rate': (df['pitch_type'] == 'ST').mean(),
            'era_avg_velo': df[df['pitch_type']=='FF']['release_speed'].mean(),
            'era_swing_rate': (df['result_type'] == 'X').mean()
        }
        if pd.isna(trend['era_avg_velo']): trend['era_avg_velo'] = 93.0
        for k, v in trend.items(): df[k] = v
        era_trends_archive[year] = trend

        # Feature Engineering
        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        df['prev_pitch_type'] = df.groupby('game_pk')['pitch_type'].shift(1).fillna('None')
        df['prev_result'] = df.groupby('game_pk')['result_type'].shift(1).fillna('None')
        df['count_advantage'] = df['strikes'] - df['balls']
        df['is_scoring_pos'] = ((df['on_2b'] == 1) | (df['on_3b'] == 1)).astype(int)

        # Encoding
        y = df['pitch_type'].map(PITCH_MAP).astype(int)
        weights = df['pitch_type'].map(CUSTOM_WEIGHTS).values
        
        df['stand'] = (df['stand'] == 'L').astype(int)
        df['p_throws'] = (df['p_throws'] == 'L').astype(int)
        df['prev_pitch_type'] = df['prev_pitch_type'].map(FEATURE_PITCH_MAP).fillna(9).astype(int)
        df['prev_result'] = df['prev_result'].map({'S':0,'B':1,'X':2,'None':3}).fillna(3).astype(int)

        features = [
            'balls', 'strikes', 'outs_when_up', 'score_diff', 'inning',
            'on_1b', 'on_2b', 'on_3b', 'is_scoring_pos', 'count_advantage',
            'stand', 'p_throws',
            'prev_pitch_type', 'prev_result',
            'era_ff_rate', 'era_sweeper_rate', 'era_swing_rate', 'era_avg_velo'
        ]
        X = df[features]
        
        # 🌟 학습 파라미터 (안정성 최우선)
        print(f"   🔥 Training on {len(X)} samples...")
        
        if model is None:
            model = xgb.XGBClassifier(
                n_estimators=150,    # 과하지 않은 학습량
                learning_rate=0.05,  # 표준 학습 속도
                max_depth=8,         # 적절한 깊이
                min_child_weight=5,  # 노이즈에 휘둘리지 않도록 설정
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
            
        del df, X, y, weights
        gc.collect()

    print("\n🎉 Ultimate Training Complete!")
    model.save_model(MODEL_PATH)
    
    final_artifacts = {
        'pitch_map': PITCH_MAP,
        'era_trends': era_trends_archive,
        'feature_map': FEATURE_PITCH_MAP
    }
    
    joblib.dump(final_artifacts, ENCODER_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_ultimate()