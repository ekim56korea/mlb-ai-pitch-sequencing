import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import os
import duckdb
import gc
import glob
from torch.utils.data import TensorDataset, DataLoader
# ⚠️ [REMOVED] from sklearn.model_selection import train_test_split
from app.model import PitchLSTM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# 🆕 [WEEK 1] Temporal validation utilities
from app.utils.validation import MLBTemporalValidator
from app.utils.metrics import MLBMetrics
# 🆕 [WEEK 2] Focal Loss for class imbalance
from app.losses.focal_loss import WeightedFocalLoss

# ─── 설정 및 경로 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
GLOBAL_MODEL_PATH = os.path.join(DATA_DIR, "pitch_lstm_global.pth")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")
DB_PATH = os.path.join(DATA_DIR, "savant.duckdb")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ⚙️ 하이퍼파라미터
DB_CHUNK_SIZE = 10000
BATCH_SIZE = 128       
TOTAL_EPOCHS = 5
INPUT_SIZE = 39        # 🔥 [WEEK 6] 43 → 39 (중복 피처 2개 제거: trajectory_div, inning_fatigue)

# 피처 순서 (모델 입력과 100% 일치해야 함)
FEATURES = [
    # [Group 1] 경기 상황 (9)
    'inning', 'balls', 'strikes', 'outs_when_up', 'score_diff', 
    'on_1b', 'on_2b', 'on_3b', 'stand_code',
    # [Group 2] 투수/타자 맥락 (4)
    'p_throws_code', 'pitch_number', 'tto', 'pitcher_pitch_count',
    # [Group 3] 타자 성향 (2)
    'batter_whiff_rate', 'batter_k_rate',
    # [Group 4] 투수 역량 및 시대 보정 (Z-Scores) (8)
    'z_vel', 'z_spin', 'z_hb', 'z_ivb', 'z_ext', 'z_rel_h', 'z_rel_s', 'z_league_whiff',
    # 🆕 [WEEK 4 - Group 5] Tunneling Features (7) - 🔥 [WEEK 6] trajectory_div 제거 (r=0.999 with tunnel_distance)
    'tunnel_distance', 'velocity_diff',
    'FF_count_last_5', 'SL_count_last_5', 'CH_count_last_5', 'CU_count_last_5',
    'sequence_entropy',
    # 🆕 [WEEK 4 - Group 6] Batter vs Pitcher Features (5)
    'bvp_ba', 'bvp_whiff_rate', 'bvp_k_rate', 'platoon_advantage', 'bvp_recent_ba',
    # 🆕 [WEEK 4 - Group 7] Contextual Features (4) - 🔥 [WEEK 6] inning_fatigue 제거 (r=1.0 with inning)
    'altitude_factor', 'rest_days', 'fatigue_index', 'pressure_index'
]

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB file not found at: {DB_PATH}")
    con = duckdb.connect(DB_PATH)
    con.execute("PRAGMA memory_limit='6GB'")
    return con

def prepare_advanced_views(con):
    """DuckDB View를 이용한 고속 Z-Score 변환 준비"""
    print("📊 Creating Advanced Statistical Views in DuckDB...", flush=True) # 🆕 flush=True
    
    # 1. 리그 연도별 평균/표준편차 (Baseline)
    con.execute("""
        CREATE OR REPLACE VIEW league_yearly_stats AS
        SELECT 
            EXTRACT(YEAR FROM game_date) as season,
            AVG(release_speed) as l_avg_vel, STDDEV(release_speed) as l_std_vel,
            AVG(release_spin_rate) as l_avg_spin, STDDEV(release_spin_rate) as l_std_spin,
            AVG(pfx_x) as l_avg_hb, STDDEV(pfx_x) as l_std_hb,
            AVG(pfx_z) as l_avg_ivb, STDDEV(pfx_z) as l_std_ivb,
            AVG(release_extension) as l_avg_ext, STDDEV(release_extension) as l_std_ext,
            AVG(release_pos_z) as l_avg_rel_h, STDDEV(release_pos_z) as l_std_rel_h,
            AVG(release_pos_x) as l_avg_rel_s, STDDEV(release_pos_x) as l_std_rel_s
        FROM pitches
        WHERE pitch_type IN ('FF','SI','FC') 
        GROUP BY 1
    """)


    # 2. 리그 연도별 헛스윙률
    con.execute("""
        CREATE OR REPLACE VIEW league_yearly_env AS
        SELECT
            EXTRACT(YEAR FROM game_date) as season,
            SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(COUNT(*), 0) as l_avg_whiff
        FROM pitches
        GROUP BY 1
    """)

    # 3. 투수별 연도별 평균 능력치
    con.execute("""
        CREATE OR REPLACE VIEW pitcher_yearly_stats AS
        SELECT 
            pitcher,
            EXTRACT(YEAR FROM game_date) as season,
            AVG(release_speed) as p_avg_vel,
            AVG(release_spin_rate) as p_avg_spin,
            AVG(pfx_x) as p_avg_hb,
            AVG(pfx_z) as p_avg_ivb,
            AVG(release_extension) as p_avg_ext,
            AVG(release_pos_z) as p_avg_rel_h,
            AVG(release_pos_x) as p_avg_rel_s
        FROM pitches
        WHERE pitch_type IN ('FF','SI','FC')
        GROUP BY 1, 2
    """)

    # 4. 최종 Z-Score 매핑 테이블
    con.execute("""
        CREATE OR REPLACE VIEW pitcher_context_z AS
        SELECT 
            p.pitcher,
            p.season,
            (p.p_avg_vel - l.l_avg_vel) / NULLIF(l.l_std_vel, 1) as z_vel,
            (p.p_avg_spin - l.l_avg_spin) / NULLIF(l.l_std_spin, 1) as z_spin,
            (p.p_avg_hb - l.l_avg_hb) / NULLIF(l.l_std_hb, 1) as z_hb,
            (p.p_avg_ivb - l.l_avg_ivb) / NULLIF(l.l_std_ivb, 1) as z_ivb,
            (p.p_avg_ext - l.l_avg_ext) / NULLIF(l.l_std_ext, 1) as z_ext,
            (p.p_avg_rel_h - l.l_avg_rel_h) / NULLIF(l.l_std_rel_h, 1) as z_rel_h,
            (p.p_avg_rel_s - l.l_avg_rel_s) / NULLIF(l.l_std_rel_s, 1) as z_rel_s,
            e.l_avg_whiff as z_league_whiff
        FROM pitcher_yearly_stats p
        JOIN league_yearly_stats l ON p.season = l.season
        JOIN league_yearly_env e ON p.season = e.season
    """)
    
    # 5. 타자 성향 뷰
    con.execute("""
        CREATE OR REPLACE VIEW batter_season_stats AS
        SELECT 
            batter,
            SUM(CASE WHEN description LIKE 'swinging_strike%' THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) as batter_whiff_rate,
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(DISTINCT at_bat_number), 0) as batter_k_rate
        FROM pitches
        GROUP BY batter
    """)
    
    # 🆕 [WEEK 4] 6. Batter vs Pitcher History View
    con.execute("""
        CREATE OR REPLACE VIEW batter_pitcher_history AS
        WITH ordered_pitches AS (
            SELECT 
                pitcher, batter, game_date, at_bat_number, pitch_number,
                pitch_type, events, description,
                ROW_NUMBER() OVER (PARTITION BY pitcher, batter ORDER BY game_date, at_bat_number, pitch_number) as rn
            FROM pitches
            WHERE pitch_type IS NOT NULL
        )
        SELECT 
            pitcher, batter,
            -- 누적 타수 (현재 타석 제외)
            COUNT(*) - 1 as bvp_ab,
            -- 누적 안타
            SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) as bvp_hits,
            -- 누적 헛스윙
            SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_blocked') THEN 1 ELSE 0 END) as bvp_whiffs,
            -- 누적 스윙
            SUM(CASE WHEN description LIKE '%swing%' OR description LIKE '%foul%' OR description IN ('swinging_strike','swinging_strike_blocked') THEN 1 ELSE 0 END) as bvp_swings,
            -- 누적 삼진
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) as bvp_strikeouts,
            -- 타석 수
            COUNT(DISTINCT at_bat_number) as bvp_pa
        FROM ordered_pitches
        GROUP BY pitcher, batter
    """)
    
    print("✅ Views Created!", flush=True)


def get_latest_checkpoint(model, optimizer):
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return 0
    latest_cp = max(checkpoints, key=os.path.getctime)
    print(f"🔄 Resuming from checkpoint: {latest_cp}")
    checkpoint = torch.load(latest_cp, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] + 1

def initialize_encoders(con):
    print("📏 Initializing Encoders...", flush=True) # 🆕 flush=True
    df = con.execute("SELECT * FROM pitches USING SAMPLE 10000").df()
    
    le_pitch = LabelEncoder().fit(df[df['pitch_type'].isin(['FF','SL','CH','CU','SI','FC','ST','FS','KC','KN'])]['pitch_type'])
    le_stand = LabelEncoder().fit(['L', 'R'])
    le_p_throws = LabelEncoder().fit(['L', 'R'])
    
    scaler = MinMaxScaler()
    scaler.fit(pd.DataFrame({
        'inning': [1, 9], 'score_diff': [-10, 10], 'pitch_number': [0, 100], 
        'at_bat_number': [0, 100], 'home_score': [0, 20], 'tto': [1, 4], 
        'pitcher_pitch_count': [0, 120]
    }))

    encoders = {
        'le_pitch': le_pitch, 'le_stand': le_stand, 'le_p_throws': le_p_throws,
        'scaler': scaler, 'input_size': INPUT_SIZE, 'num_classes': len(le_pitch.classes_)
    }
    joblib.dump(encoders, ENCODER_PATH)
    print("✅ Scalers Initialized & Saved!", flush=True) # 🆕 flush=True
    return encoders

def train_global_model(start_year=2015):
    """최종 학습 루프 - Focal Loss 적용"""
    print(f"🌍 Starting Global Training Process (Year: {start_year})...", flush=True) # 🆕

    con = get_db_connection()
    prepare_advanced_views(con)
    
    if os.path.exists(ENCODER_PATH):
        encoders = joblib.load(ENCODER_PATH)
    else:
        encoders = initialize_encoders(con)
    
    model = PitchLSTM(INPUT_SIZE, 128, 2, encoders['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 🆕 [WEEK 2] Step 1: Calculate class counts for weighted Focal Loss
    print("📊 Calculating class distribution for Focal Loss...", flush=True)
    class_query = """
        SELECT pitch_type, COUNT(*) as count 
        FROM pitches 
        WHERE pitch_type IN ('FF','SL','CH','CU','SI','FC','ST','FS','KC','KN')
        GROUP BY pitch_type
        ORDER BY pitch_type
    """
    class_df = con.execute(class_query).df()
    le_pitch = encoders['le_pitch']
    
    # Create class_counts array matching encoder order
    class_counts = np.zeros(len(le_pitch.classes_), dtype=int)
    for idx, pitch_type in enumerate(le_pitch.classes_):
        count_row = class_df[class_df['pitch_type'] == pitch_type]
        if not count_row.empty:
            class_counts[idx] = count_row['count'].iloc[0]
        else:
            class_counts[idx] = 1  # Avoid division by zero
    
    print(f"📊 Class Distribution:")
    for idx, (pitch, count) in enumerate(zip(le_pitch.classes_, class_counts)):
        print(f"   {pitch}: {count:,} pitches")
    
    # 🆕 [WEEK 2] Step 2: Use WeightedFocalLoss instead of CrossEntropyLoss
    criterion = WeightedFocalLoss(
        class_counts=class_counts,
        gamma=2.0,  # Standard focal loss parameter
        beta=0.999  # Effective number smoothing
    )
    print(f"✅ Focal Loss initialized (gamma=2.0, beta=0.999)", flush=True)
    
    start_epoch = get_latest_checkpoint(model, optimizer)
    
    total_rows_query = f"SELECT COUNT(*) FROM pitches WHERE game_date >= '{start_year}-01-01'"
    total_rows = con.execute(total_rows_query).fetchone()[0]
    print(f"🚀 Training on {total_rows:,} rows (2015-2025) | Input Features: {INPUT_SIZE}", flush=True) # 🆕

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\n🌟 Epoch {epoch+1} Start", flush=True) # 🆕
        
        for offset in range(0, total_rows, DB_CHUNK_SIZE):
            gc.collect()
            
            # 🔥 Main Query: 43개 피처 조립 (🆕 WEEK 4: +18개)
            query = f"""
                WITH base AS (
                    SELECT 
                        p.*,
                        DENSE_RANK() OVER (PARTITION BY game_pk, pitcher, batter ORDER BY at_bat_number) as tto,
                        ROW_NUMBER() OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as pitcher_pitch_count,
                        -- 🆕 Tunneling: 이전 투구 정보
                        LAG(release_pos_x) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_release_x,
                        LAG(release_pos_z) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_release_z,
                        LAG(pfx_x) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_pfx_x,
                        LAG(pfx_z) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_pfx_z,
                        LAG(release_speed) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_speed,
                        LAG(plate_x) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_plate_x,
                        LAG(plate_z) OVER (PARTITION BY game_pk, at_bat_number ORDER BY pitch_number) as prev_plate_z,
                        -- 🆕 Contextual: 휴식일
                        JULIANDAY(game_date) - LAG(JULIANDAY(game_date)) OVER (PARTITION BY pitcher ORDER BY game_date) as rest_days
                    FROM pitches p
                    WHERE p.game_date >= '{start_year}-01-01' AND p.pitch_type IS NOT NULL
                ),
                -- 🆕 최근 5투구 구종 카운트
                pitch_counts AS (
                    SELECT 
                        game_pk, at_bat_number, pitch_number,
                        SUM(CASE WHEN pitch_type = 'FF' THEN 1 ELSE 0 END) OVER (
                            PARTITION BY game_pk, at_bat_number 
                            ORDER BY pitch_number 
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                        ) as FF_count_last_5,
                        SUM(CASE WHEN pitch_type = 'SL' THEN 1 ELSE 0 END) OVER (
                            PARTITION BY game_pk, at_bat_number 
                            ORDER BY pitch_number 
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                        ) as SL_count_last_5,
                        SUM(CASE WHEN pitch_type = 'CH' THEN 1 ELSE 0 END) OVER (
                            PARTITION BY game_pk, at_bat_number 
                            ORDER BY pitch_number 
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                        ) as CH_count_last_5,
                        SUM(CASE WHEN pitch_type = 'CU' THEN 1 ELSE 0 END) OVER (
                            PARTITION BY game_pk, at_bat_number 
                            ORDER BY pitch_number 
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                        ) as CU_count_last_5
                    FROM pitches
                    WHERE game_date >= '{start_year}-01-01' AND pitch_type IS NOT NULL
                )
                SELECT 
                    -- 기존 피처 (25개)
                    b.inning, b.balls, b.strikes, b.outs_when_up, (b.fld_score - b.bat_score) as score_diff,
                    b.on_1b, b.on_2b, b.on_3b, b.stand, b.p_throws,
                    b.pitch_number, b.tto, b.pitcher_pitch_count, b.home_score,
                    COALESCE(bs.batter_whiff_rate, 0.25) as batter_whiff_rate,
                    COALESCE(bs.batter_k_rate, 0.20) as batter_k_rate,
                    -- Z-Scores
                    COALESCE(z.z_vel, 0) as z_vel, COALESCE(z.z_spin, 0) as z_spin,
                    COALESCE(z.z_hb, 0) as z_hb, COALESCE(z.z_ivb, 0) as z_ivb,
                    COALESCE(z.z_ext, 0) as z_ext, COALESCE(z.z_rel_h, 0) as z_rel_h,
                    COALESCE(z.z_rel_s, 0) as z_rel_s, COALESCE(z.z_league_whiff, 0.10) as z_league_whiff,
                    
                    -- 🆕 Tunneling Features (8개)
                    b.release_pos_x, b.release_pos_z, b.prev_release_x, b.prev_release_z,
                    b.pfx_x, b.pfx_z, b.prev_pfx_x, b.prev_pfx_z,
                    b.release_speed, b.prev_speed,
                    b.plate_x, b.plate_z, b.prev_plate_x, b.prev_plate_z,
                    COALESCE(pc.FF_count_last_5, 0) as FF_count_last_5,
                    COALESCE(pc.SL_count_last_5, 0) as SL_count_last_5,
                    COALESCE(pc.CH_count_last_5, 0) as CH_count_last_5,
                    COALESCE(pc.CU_count_last_5, 0) as CU_count_last_5,
                    
                    -- 🆕 BvP Features (5개)
                    COALESCE(bvp.bvp_hits::FLOAT / NULLIF(bvp.bvp_ab, 0), 0.250) as bvp_ba,
                    COALESCE(bvp.bvp_whiffs::FLOAT / NULLIF(bvp.bvp_swings, 0), 0.24) as bvp_whiff_rate,
                    COALESCE(bvp.bvp_strikeouts::FLOAT / NULLIF(bvp.bvp_pa, 0), 0.20) as bvp_k_rate,
                    CASE WHEN b.stand != b.p_throws THEN 1 ELSE -1 END as platoon_advantage,
                    COALESCE(bvp.bvp_hits::FLOAT / NULLIF(bvp.bvp_ab, 0), 0.250) as bvp_recent_ba,
                    
                    -- 🆕 Contextual Features (5개)
                    COALESCE(b.rest_days, 4) as rest_days,
                    b.game_date, b.pitcher, b.venue_name,
                    
                    b.pitch_type
                FROM base b
                LEFT JOIN batter_season_stats bs ON b.batter = bs.batter
                LEFT JOIN pitcher_context_z z ON b.pitcher = z.pitcher AND EXTRACT(YEAR FROM b.game_date) = z.season
                LEFT JOIN pitch_counts pc ON b.game_pk = pc.game_pk AND b.at_bat_number = pc.at_bat_number AND b.pitch_number = pc.pitch_number
                LEFT JOIN batter_pitcher_history bvp ON b.pitcher = bvp.pitcher AND b.batter = bvp.batter
                ORDER BY b.game_date ASC
                LIMIT {DB_CHUNK_SIZE} OFFSET {offset}
            """
            
            try:
                df = con.execute(query).df()
                if df.empty: break
                
                # Numpy Preprocessing (Fast)
                le_pitch = encoders['le_pitch']
                df = df[df['pitch_type'].isin(le_pitch.classes_)]
                if df.empty: continue
                
                X = np.zeros((len(df), INPUT_SIZE), dtype=np.float32)
                
                # Group 1: Situation Features (8)
                X[:, 0] = df['inning'] / 9.0  
                X[:, 1] = df['balls'] / 4.0
                X[:, 2] = df['strikes'] / 3.0
                X[:, 3] = df['outs_when_up'] / 3.0
                X[:, 4] = (df['score_diff'] + 10) / 20.0
                X[:, 5] = df['on_1b'].fillna(0)
                X[:, 6] = df['on_2b'].fillna(0)
                X[:, 7] = df['on_3b'].fillna(0)
                
                # Group 2: Pitcher/Batter Info (5)
                X[:, 8] = encoders['le_stand'].transform(df['stand'].fillna('R'))
                X[:, 9] = encoders['le_p_throws'].transform(df['p_throws'].fillna('R'))
                X[:, 10] = df['pitch_number'] / 100.0
                X[:, 11] = df['tto'] / 4.0
                X[:, 12] = df['pitcher_pitch_count'] / 100.0
                
                # Group 3: Batter Tendency (2)
                X[:, 13] = df['batter_whiff_rate']
                X[:, 14] = df['batter_k_rate']
                
                # Group 4: Z-Score Features (8)
                X[:, 15:23] = df[['z_vel', 'z_spin', 'z_hb', 'z_ivb', 'z_ext', 'z_rel_h', 'z_rel_s', 'z_league_whiff']].values
                
                # 🆕 Group 5: Tunneling Features (7) - 🔥 [WEEK 6] trajectory_div 제거
                # Calculate tunnel_distance from release points
                X[:, 23] = np.sqrt(
                    (df['release_pos_x'] - df['prev_release_x'].fillna(df['release_pos_x'])) ** 2 +
                    (df['release_pos_z'] - df['prev_release_z'].fillna(df['release_pos_z'])) ** 2
                )
                # Velocity difference (idx 24, was 25)
                X[:, 24] = np.abs(df['release_speed'] - df['prev_speed'].fillna(df['release_speed']))
                # Pitch type counts from last 5 pitches (idx 25-28, was 26-29)
                X[:, 25] = df['FF_count_last_5'].fillna(0) / 5.0
                X[:, 26] = df['SL_count_last_5'].fillna(0) / 5.0
                X[:, 27] = df['CH_count_last_5'].fillna(0) / 5.0
                X[:, 28] = df['CU_count_last_5'].fillna(0) / 5.0
                # Sequence entropy (idx 29, was 30) - TODO: implement Shannon entropy
                X[:, 29] = df['sequence_entropy'].fillna(0)
                
                # 🆕 Group 6: Batter vs Pitcher Features (5) - indices shift by 2
                X[:, 30] = df['bvp_ba']  # was 31
                X[:, 31] = df['bvp_whiff_rate']  # was 32
                X[:, 32] = df['bvp_k_rate']  # was 33
                X[:, 33] = df['platoon_advantage']  # was 34
                X[:, 34] = df['bvp_recent_ba']  # was 35
                
                # 🆕 Group 7: Contextual Features (4) - 🔥 [WEEK 6] inning_fatigue 제거
                # Altitude factor (idx 35, was 36)
                X[:, 35] = df['altitude_factor']
                # Rest days (idx 36, was 37)
                X[:, 36] = df['rest_days'].fillna(4) / 10.0
                # Fatigue index (idx 37, was 38)
                X[:, 37] = df['fatigue_index']
                # Pressure index (idx 38, was 39)
                runners_on = (df['on_1b'] + df['on_2b'] + df['on_3b']).clip(0, 3)
                late_inning = (df['inning'] >= 7).astype(float)
                close_game = (np.abs(df['score_diff']) <= 2).astype(float)
                X[:, 38] = (runners_on / 3.0 * 0.4 + late_inning * 0.3 + close_game * 0.3)
                
                # Target
                y = le_pitch.transform(df['pitch_type'])

                # Sequence
                seq_len = 5
                X_seq = np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])
                y_seq = y[seq_len:]
                
                if len(X_seq) == 0: continue

                tensor_x = torch.FloatTensor(X_seq).to(device)
                tensor_y = torch.LongTensor(y_seq).to(device)
                dataset = TensorDataset(tensor_x, tensor_y)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                
                print(f"\r   Processing {min((offset+DB_CHUNK_SIZE), total_rows)/total_rows*100:.1f}%", end="")

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

            progress = min((offset+DB_CHUNK_SIZE), total_rows)/total_rows*100
            print(f"\r   Processing {progress:.1f}%...", end="", flush=True) # 🆕 flush=True 필수

        # Save Checkpoint
        save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
        print(f"\n💾 Saved Epoch {epoch+1}", flush=True) # 🆕

    con.close()
    print("🎉 Grand Training Complete!", flush=True) # 🆕

def fine_tune_pitcher(pitcher_id: int, pitcher_name: str, target_date: str = None, seq_len: int = 5):
    """
    🆕 Fine-tuning 함수도 25개 피처를 사용하도록 업데이트 (로직 복원)
    """
    print(f"🎓 Fine-tuning for {pitcher_name}...")
    con = get_db_connection()
    prepare_advanced_views(con)
    encoders = joblib.load(ENCODER_PATH)
    
    # Global 모델 로드
    model = PitchLSTM(INPUT_SIZE, 128, 2, encoders['num_classes']).to(device)
    if os.path.exists(GLOBAL_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=device))
        except RuntimeError:
            print("⚠️ Model mismatch, retraining global model might be needed.")
            
    # 개인 데이터 조회 (Z-Score 포함 + 43 features)
    query = f"""
        WITH base AS (
            SELECT 
                p.*,
                DENSE_RANK() OVER (PARTITION BY game_pk, pitcher, batter ORDER BY at_bat_number) as tto,
                ROW_NUMBER() OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as pitcher_pitch_count,
                LAG(release_pos_x) OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as prev_release_x,
                LAG(release_pos_z) OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as prev_release_z,
                LAG(pfx_x) OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as prev_pfx_x,
                LAG(pfx_z) OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as prev_pfx_z,
                LAG(release_speed) OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as prev_speed,
                LAG(pitch_type) OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) as prev_pitch_type
            FROM pitches p
            WHERE pitcher = {pitcher_id} AND p.pitch_type IS NOT NULL
        ),
        pitch_counts AS (
            SELECT 
                *,
                SUM(CASE WHEN pitch_type = 'FF' THEN 1 ELSE 0 END) OVER (
                    PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number 
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as FF_count_last_5,
                SUM(CASE WHEN pitch_type = 'SL' THEN 1 ELSE 0 END) OVER (
                    PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number 
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as SL_count_last_5,
                SUM(CASE WHEN pitch_type = 'CH' THEN 1 ELSE 0 END) OVER (
                    PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number 
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as CH_count_last_5,
                SUM(CASE WHEN pitch_type = 'CU' THEN 1 ELSE 0 END) OVER (
                    PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number 
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as CU_count_last_5,
                0.0 as sequence_entropy
            FROM base
        )
        SELECT 
            b.inning, b.balls, b.strikes, b.outs_when_up, (b.fld_score - b.bat_score) as score_diff,
            b.on_1b, b.on_2b, b.on_3b, b.stand, b.p_throws,
            b.pitch_number, b.tto, b.pitcher_pitch_count, b.home_score,
            COALESCE(bs.batter_whiff_rate, 0.25) as batter_whiff_rate,
            COALESCE(bs.batter_k_rate, 0.20) as batter_k_rate,
            COALESCE(z.z_vel, 0) as z_vel, COALESCE(z.z_spin, 0) as z_spin,
            COALESCE(z.z_hb, 0) as z_hb, COALESCE(z.z_ivb, 0) as z_ivb,
            COALESCE(z.z_ext, 0) as z_ext, COALESCE(z.z_rel_h, 0) as z_rel_h,
            COALESCE(z.z_rel_s, 0) as z_rel_s, COALESCE(z.z_league_whiff, 0.10) as z_league_whiff,
            b.release_pos_x, b.prev_release_x, b.release_pos_z, b.prev_release_z,
            b.pfx_x, b.prev_pfx_x, b.pfx_z, b.prev_pfx_z,
            b.release_speed, b.prev_speed,
            b.FF_count_last_5, b.SL_count_last_5, b.CH_count_last_5, b.CU_count_last_5,
            b.sequence_entropy,
            COALESCE(bvp.bvp_hits::FLOAT / NULLIF(bvp.bvp_ab, 0), 0.250) as bvp_ba,
            COALESCE(bvp.bvp_whiffs::FLOAT / NULLIF(bvp.bvp_swings, 0), 0.30) as bvp_whiff_rate,
            COALESCE(bvp.bvp_strikeouts::FLOAT / NULLIF(bvp.bvp_pa, 0), 0.20) as bvp_k_rate,
            CASE WHEN b.stand != b.p_throws THEN 1.0 ELSE 0.0 END as platoon_advantage,
            0.250 as bvp_recent_ba,
            1.0 as altitude_factor,
            4 as rest_days,
            0.5 as fatigue_index,
            b.pitch_type
        FROM pitch_counts b
        LEFT JOIN batter_season_stats bs ON b.batter = bs.batter
        LEFT JOIN pitcher_context_z z ON b.pitcher = z.pitcher AND EXTRACT(YEAR FROM b.game_date) = z.season
        LEFT JOIN batter_pitcher_history bvp ON b.pitcher = bvp.pitcher AND b.batter = bvp.batter
        ORDER BY b.game_date ASC
    """
    try:
        df = con.execute(query).df()
    except Exception as e:
        return {"status": "error", "message": f"DB Error: {str(e)}"}
    
    if len(df) < 50:
        return {"status": "error", "message": "Not enough data"}

    # 전처리 및 학습 로직 복원
    le_pitch = encoders['le_pitch']
    df = df[df['pitch_type'].isin(le_pitch.classes_)]
    
    if df.empty:
        return {"status": "error", "message": "No valid pitch types found"}

    X = np.zeros((len(df), INPUT_SIZE), dtype=np.float32)
    
    # Group 1: Situation Features (8)
    X[:, 0] = df['inning'] / 9.0  
    X[:, 1] = df['balls'] / 4.0
    X[:, 2] = df['strikes'] / 3.0
    X[:, 3] = df['outs_when_up'] / 3.0
    X[:, 4] = (df['score_diff'] + 10) / 20.0
    X[:, 5] = df['on_1b'].fillna(0)
    X[:, 6] = df['on_2b'].fillna(0)
    X[:, 7] = df['on_3b'].fillna(0)
    
    # Group 2: Pitcher/Batter Info (5)
    X[:, 8] = encoders['le_stand'].transform(df['stand'].fillna('R'))
    X[:, 9] = encoders['le_p_throws'].transform(df['p_throws'].fillna('R'))
    X[:, 10] = df['pitch_number'] / 100.0
    X[:, 11] = df['tto'] / 4.0
    X[:, 12] = df['pitcher_pitch_count'] / 100.0
    
    # Group 3: Batter Tendency (2)
    X[:, 13] = df['batter_whiff_rate']
    X[:, 14] = df['batter_k_rate']
    
    # Group 4: Z-Score Features (8)
    X[:, 15:23] = df[['z_vel', 'z_spin', 'z_hb', 'z_ivb', 'z_ext', 'z_rel_h', 'z_rel_s', 'z_league_whiff']].values
    
    # 🔥 [WEEK 6] Group 5: Tunneling Features (7) - trajectory_div 제거
    X[:, 23] = np.sqrt((df['release_pos_x'] - df['prev_release_x'].fillna(df['release_pos_x'])) ** 2 +
                       (df['release_pos_z'] - df['prev_release_z'].fillna(df['release_pos_z'])) ** 2)
    X[:, 24] = np.abs(df['release_speed'] - df['prev_speed'].fillna(df['release_speed']))
    X[:, 25] = df['FF_count_last_5'].fillna(0) / 5.0
    X[:, 26] = df['SL_count_last_5'].fillna(0) / 5.0
    X[:, 27] = df['CH_count_last_5'].fillna(0) / 5.0
    X[:, 28] = df['CU_count_last_5'].fillna(0) / 5.0
    X[:, 29] = df['sequence_entropy'].fillna(0)
    
    # Group 6: Batter vs Pitcher Features (5)
    X[:, 30] = df['bvp_ba']
    X[:, 31] = df['bvp_whiff_rate']
    X[:, 32] = df['bvp_k_rate']
    X[:, 33] = df['platoon_advantage']
    X[:, 34] = df['bvp_recent_ba']
    
    # 🔥 [WEEK 6] Group 7: Contextual Features (4) - inning_fatigue 제거
    X[:, 35] = df['altitude_factor']
    X[:, 36] = df['rest_days'].fillna(4) / 10.0
    X[:, 37] = df['fatigue_index']
    runners_on = (df['on_1b'] + df['on_2b'] + df['on_3b']).clip(0, 3)
    late_inning = (df['inning'] >= 7).astype(float)
    close_game = (np.abs(df['score_diff']) <= 2).astype(float)
    X[:, 38] = (runners_on / 3.0 * 0.4 + late_inning * 0.3 + close_game * 0.3)
    
    y = le_pitch.transform(df['pitch_type'])
    
    X_seq = np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])
    y_seq = y[seq_len:]
    
    # 학습 수행
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # 낮은 학습률
    
    # 🆕 [WEEK 2] Fine-tuning도 Focal Loss 적용
    # Calculate class distribution for this pitcher
    pitch_counts = df['pitch_type'].value_counts()
    class_counts_ft = np.zeros(len(le_pitch.classes_), dtype=int)
    for idx, pitch_type in enumerate(le_pitch.classes_):
        if pitch_type in pitch_counts.index:
            class_counts_ft[idx] = pitch_counts[pitch_type]
        else:
            class_counts_ft[idx] = 1
    
    criterion = WeightedFocalLoss(class_counts=class_counts_ft, gamma=2.0, beta=0.999)
    
    tensor_x = torch.FloatTensor(X_seq).to(device)
    tensor_y = torch.LongTensor(y_seq).to(device)
    
    model.train()
    for i in range(10): # 10 Epochs
        optimizer.zero_grad()
        loss = criterion(model(tensor_x), tensor_y)
        loss.backward()
        optimizer.step()
        
    # 개인 모델 저장
    save_path = os.path.join(MODEL_DIR, f"lstm_{pitcher_id}.pth")
    torch.save(model.state_dict(), save_path)
    
    return {"status": "success", "accuracy": "updated"}