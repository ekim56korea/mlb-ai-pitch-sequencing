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

# 평가 설정
START_YEAR = 2023
END_YEAR = 2025

def evaluate_fast():
    print(f"🚀 Starting High-Speed Evaluation ({START_YEAR} - {END_YEAR})...")
    print("   💻 Mode: Local/Vectorized (No Memory Limit)")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("❌ Model/Artifacts not found.")
        return

    # 1. 모델 및 아티팩트 로드
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    artifacts = joblib.load(ENCODER_PATH)
    pitch_map = artifacts['pitch_map']
    feature_map = artifacts['feature_map']
    era_trends = artifacts['era_trends']
    
    num_classes = len(pitch_map)
    valid_keys = list(pitch_map.keys())

    # 2. 아스날(구종 목록) 사전 구축
    # 전체 기간에 대해 한 번만 로드
    print("   🧠 Building Global Pitcher Arsenals...")
    con = duckdb.connect(DB_PATH, read_only=True)
    
    arsenal_query = f"""
        SELECT DISTINCT pitcher, pitch_type 
        FROM pitches 
        WHERE CAST(STRFTIME(game_date, '%Y') AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
          AND pitch_type IS NOT NULL
    """
    arsenal_df = con.execute(arsenal_query).df()
    
    # 딕셔너리 변환
    pitcher_arsenals_idx = {}
    for pid, group in arsenal_df.groupby('pitcher'):
        indices = [pitch_map[p] for p in group['pitch_type'].values if p in pitch_map]
        if indices:
            pitcher_arsenals_idx[pid] = indices
    
    del arsenal_df
    print(f"   ✅ Arsenals ready for {len(pitcher_arsenals_idx)} pitchers.")

    # 누적 변수
    total_samples = 0
    total_hits_top1 = 0
    total_hits_top2 = 0

    # 3. 연도별 고속 처리
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n⚡ Processing Season {year} (Full Load)...")
        
        # 3-1. 데이터 통째로 로드 (Chunking 없음)
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
        """
        df = con.execute(query).df()
        
        if df.empty:
            print(f"⚠️ No data for {year}")
            continue
            
        # 3-2. 벡터화 전처리 (Vectorized Preprocessing)
        # 판다스 전체 연산이므로 루프보다 훨씬 빠름
        df = df[df['pitch_type'].isin(valid_keys)].copy()
        
        fill_cols = ['on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'score_diff']
        df[fill_cols] = df[fill_cols].fillna(0).astype(int)

        trend = era_trends.get(year, era_trends[max(era_trends.keys())])
        for k, v in trend.items(): df[k] = v

        df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        # Lag Features (Vectorized)
        df['prev_pitch_type'] = df.groupby('game_pk')['pitch_type'].shift(1).fillna('None').map(feature_map).fillna(9).astype(int)
        df['prev_result'] = df.groupby('game_pk')['result_type'].shift(1).fillna('None').map({'S':0,'B':1,'X':2,'None':3}).fillna(3).astype(int)
        
        df['count_advantage'] = df['strikes'] - df['balls']
        df['is_scoring_pos'] = ((df['on_2b'] == 1) | (df['on_3b'] == 1)).astype(int)
        df['stand'] = (df['stand'] == 'L').astype(int)
        df['p_throws'] = (df['p_throws'] == 'L').astype(int)

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
        
        # 3-3. 초고속 예측 (Batch Prediction)
        print(f"   🔥 Predicting {len(X)} rows...")
        raw_probs = model.predict_proba(X) # (N, 9) Matrix
        
        # 3-4. 행렬 마스킹 (Matrix Masking) - 루프 제거의 핵심
        print(f"   🎭 Applying Arsenals (Vectorized)...")
        
        # 마스크 행렬 초기화 (모두 0)
        mask_matrix = np.zeros_like(raw_probs)
        
        # 투수별로 해당 투수가 던지는 구종 컬럼을 1로 설정
        # (row loop 대신 pitcher loop를 돌므로 훨씬 빠름: 70만번 -> 800번)
        unique_pids = np.unique(pids)
        
        # 투수 ID를 DataFrame 인덱스와 매핑하기 위해 정렬된 상태 활용 가능하지만
        # 여기서는 안전하게 boolean indexing 사용 (Numpy가 C로 최적화되어 있어 빠름)
        for pid in tqdm(unique_pids, desc="Masking"):
            valid_indices = pitcher_arsenals_idx.get(pid)
            if valid_indices:
                # 해당 투수의 row 위치 찾기
                rows = np.where(pids == pid)[0]
                # 해당 row들의 valid_indices 컬럼을 1.0으로 설정
                # np.ix_는 broadcast 불가능한 차원 처리를 도움, 하지만 여기선 단순 슬라이싱
                # mask_matrix[rows, valid_idx] 문법이 안되므로 아래와 같이 처리
                
                # 열 인덱스 브로드캐스팅
                mask_matrix[rows[:, None], valid_indices] = 1.0
            else:
                # 아스날 정보 없으면 모두 1 (패널티 없음)
                rows = np.where(pids == pid)[0]
                mask_matrix[rows, :] = 1.0

        # 마스킹 적용 (Element-wise multiplication)
        final_probs = raw_probs * mask_matrix
        
        # 3-5. 초고속 채점 (Numpy Functions)
        # Top-1: argmax
        pred_top1 = np.argmax(final_probs, axis=1)
        hits_1 = np.sum(pred_top1 == y_true)
        
        # Top-2: argsort (마지막 2개 컬럼에 정답이 있는지)
        # axis=1로 정렬하여 인덱스 반환
        top2_indices = np.argsort(final_probs, axis=1)[:, -2:]
        # y_true가 top2_indices의 각 행에 포함되는지 확인
        # Broadcasting: y_true를 (N, 1)로 만들어 비교
        hits_2 = np.sum(np.any(top2_indices == y_true[:, None], axis=1))
        
        acc_1 = hits_1 / len(X)
        acc_2 = hits_2 / len(X)
        
        print(f"   ✅ {year} Result -> Top-1: {acc_1:.4f} | Top-2: {acc_2:.4f}")
        
        total_samples += len(X)
        total_hits_top1 += hits_1
        total_hits_top2 += hits_2
        
        del df, X, raw_probs, mask_matrix, final_probs
        gc.collect()

    con.close()

    # 4. 최종 결과
    if total_samples > 0:
        final_acc_1 = total_hits_top1 / total_samples
        final_acc_2 = total_hits_top2 / total_samples
        
        print("\n" + "="*50)
        print(f"   🏆 FAST LOCAL REPORT ({START_YEAR}-{END_YEAR})")
        print("="*50)
        print(f"   ✅ Overall Top-1 Accuracy: {final_acc_1:.4f}")
        print(f"   ✅ Overall Top-2 Accuracy: {final_acc_2:.4f}")
        print(f"   🔢 Total Samples: {total_samples}")
        print("-" * 50)
    else:
        print("❌ No data.")

if __name__ == "__main__":
    evaluate_fast()