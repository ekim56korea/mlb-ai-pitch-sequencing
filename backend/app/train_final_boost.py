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

# 🌟 [승부수] Aggressive Weights (공격적 가중치)
# 직구(FF)를 압도적인 1순위로 두고, 나머지는 확실한 근거가 있을 때만 튀어나오게 억제
CUSTOM_WEIGHTS = {
    'FF': 4.0,         # 🔥 초대폭 상향 (기존 1.8 -> 4.0) : "모르면 직구 던져!"
    'SI': 2.5,         # ▲ 상향 (싱커도 패스트볼 계열)
    'FC': 2.5,         # ▲ 상향 (커터도 패스트볼 계열)
    'SL': 2.0,         # - 유지 (슬라이더는 제2구종으로 중요)
    'CH': 1.0,         # ▼ 대폭 하향 (2.0 -> 1.0) : 체인지업 남발 금지
    'CU': 1.5,         # ▼ 하향 (커브 억제)
    'ST': 2.0,         # ▼ 하향 (스위퍼 억제)
    'FS': 2.0,         # - 유지
    'SV': 2.0          # - 유지
}

def train_final_boost():
    print(f"🚀 Starting Final Boost Training (Target: Top-1 45%+, Top-2 70%)...")
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
        
        # 🌟 학습 파라미터 강화 (Mac 성능 활용)
        print(f"   🔥 Training on {len(X)} samples (Boosted)...")
        
        if model is None:
            model = xgb.XGBClassifier(
                n_estimators=200,    # 🌟 학습량 2배 증가 (100 -> 200)
                learning_rate=0.03,  # 🌟 학습 속도 감소 (더 꼼꼼하게 학습)
                max_depth=8,         # 깊이는 적절하게 유지
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=len(VALID_PITCHES),
                n_jobs=-1,           # CPU Full Power
                tree_method='hist'
            )
            model.fit(X, y, sample_weight=weights, verbose=False)
        else:
            model.fit(X, y, sample_weight=weights, xgb_model=model, verbose=False)
            
        del df, X, y, weights
        gc.collect()

    print("\n🎉 Boost Training Complete!")
    model.save_model(MODEL_PATH)
    
    final_artifacts = {
        'pitch_map': PITCH_MAP,
        'era_trends': era_trends_archive,
        'feature_map': FEATURE_PITCH_MAP
    }
    
    joblib.dump(final_artifacts, ENCODER_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_final_boost()