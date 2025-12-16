from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import pybaseball
from pybaseball import playerid_lookup, statcast_pitcher
from datetime import datetime, timedelta
import torch
import joblib
import os
import math

# 📂 사용자 정의 모듈 임포트
from app.physics import calculate_trajectory
from app.model import PitchLSTM

app = FastAPI()

# CORS 설정 (프론트엔드 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 경로 및 설정 ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # /code/app
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data") # /code/data

MODEL_PATH = os.path.join(DATA_DIR, "pitch_lstm.pth")
ENCODER_PATH = os.path.join(DATA_DIR, "encoders.pkl")

# 유효 구종 리스트 (이상한 데이터 필터링용)
VALID_ARSENAL = ['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'ST', 'SV', 'FS', 'KC', 'KN', 'EP']

# ⚾️ Run Value 가중치 (MLB 선형 가중치 근사값)
# 음수일수록 투수에게 유리함
RUN_VALUES = {
    # 투구 결과
    "ball": 0.06, "blocked_ball": 0.06,
    "called_strike": -0.06, "swinging_strike": -0.12, "foul": -0.04,
    "hit_into_play": 0.0, 
    # 타격 결과
    "single": 0.48, "double": 0.77, "triple": 1.05, "home_run": 1.40,
    "walk": 0.32, "hit_by_pitch": 0.34, 
    "strikeout": -0.27, "field_out": -0.27, "force_out": -0.27,
    "grounded_into_double_play": -0.45
}

# ─── 전역 변수 (AI 모델) ───
lstm_model = None
encoders = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 헬퍼 함수 ───
def calculate_run_value(row):
    """한 투구의 Run Value를 계산"""
    event = row.get('events')
    if pd.notna(event) and event in RUN_VALUES:
        return RUN_VALUES[event]
    desc = row.get('description')
    if pd.notna(desc) and desc in RUN_VALUES:
        return RUN_VALUES[desc]
    return 0.0

@app.on_event("startup")
def load_ai_model():
    global lstm_model, encoders
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            print(f"🧠 Loading AI Model from {MODEL_PATH}...")
            encoders = joblib.load(ENCODER_PATH)
            
            input_size = encoders['input_size']
            num_classes = encoders['num_classes']
            
            lstm_model = PitchLSTM(input_size, 128, 2, num_classes)
            lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            lstm_model.to(device)
            lstm_model.eval()
            print("✅ AI Brain Successfully Loaded!")
        else:
            print("⚠️ Model files not found. Using Rule-Based mode.")
    except Exception as e:
        print(f"❌ Failed to load AI model: {e}")

# ─── 데이터 모델 ───
class GameContext(BaseModel):
    pitcher_name: str
    batter_stand: str
    balls: int
    strikes: int
    outs: int
    inning: int

class SimulationRequest(BaseModel):
    pitcher_name: str
    pitch_type: str
    batter_stand: str

# ─── API 엔드포인트 ───

@app.post("/load/matchup")
def load_matchup(
    pitcher_name: str, 
    batter_name: str = "Batter",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    try:
        print(f"🔍 Searching: {pitcher_name}")
        
        # 1. 선수 검색
        name_parts = pitcher_name.strip().split()
        player_info = pd.DataFrame()

        if len(name_parts) >= 2:
            player_info = playerid_lookup(name_parts[-1], name_parts[0])
        
        if player_info.empty:
            player_info = playerid_lookup(name_parts[-1])

        if player_info.empty:
            return {"status": "error", "message": f"Pitcher '{pitcher_name}' not found"}
        
        # 동명이인 처리 (최신 선수)
        player_info = player_info.sort_values(by='mlb_played_last', ascending=False)
        pitcher_id = player_info.iloc[0]['key_mlbam']
        real_name = f"{player_info.iloc[0]['name_first']} {player_info.iloc[0]['name_last']}"

        # 2. 날짜 설정
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if not start_date:
            # 기본값: 1년 전
            end_dt_obj = datetime.strptime(end_date, '%Y-%m-%d')
            start_date = (end_dt_obj - timedelta(days=365)).strftime('%Y-%m-%d')

        print(f"📅 Fetching data for {real_name} (ID: {pitcher_id})")
        print(f"   Period: {start_date} ~ {end_date}")

        # 3. 데이터 조회
        df = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=pitcher_id)
        
        if df.empty:
            return {"status": "error", "message": "No data available for this period."}

        # 4. 데이터 정제
        df = df[df['pitch_type'].isin(VALID_ARSENAL)]
        df = df.dropna(subset=['release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z'])
        
        if len(df) < 10:
             return {"status": "error", "message": "Not enough valid data in this period."}

        # 5. 구종 통계 + Run Value 계산
        df['rv'] = df.apply(calculate_run_value, axis=1)

        arsenal = {}
        grouped = df.groupby('pitch_type')
        
        for pitch_type, group in grouped:
            count = len(group)
            
            # 🚨 [추가된 부분] 표본 필터링 (Noise Filter)
            # 투구 수가 5개 미만인 구종은 통계적 가치가 없으므로 제외합니다.
            # 이렇게 하면 ST +77 같은 이상한 데이터가 사라집니다.
            if count < 5:
                continue

            total_rv = group['rv'].sum()
            rv_per_100 = round((total_rv / count) * 100, 2)

            arsenal[pitch_type] = {
                "count": int(count),
                "percentage": round((count / len(df)) * 100, 1),
                "run_value_per_100": rv_per_100,
                "avg_velocity": round(group['release_speed'].mean(), 1),
                "release_speed": round(group['release_speed'].mean(), 1),
                "pfx_x": round(group['pfx_x'].mean() * 12, 1),
                "pfx_z": round(group['pfx_z'].mean() * 12, 1),
                "plate_x": round(group['plate_x'].mean(), 2),
                "plate_z": round(group['plate_z'].mean(), 2),
                "release_pos_x": round(group['release_pos_x'].mean(), 2),
                "release_pos_z": round(group['release_pos_z'].mean(), 2),
                "release_extension": round(group['release_extension'].mean(), 2)
            }

        # 6. 시각화 데이터 샘플링 (최대 1000개)
        locations_df = df.sample(min(len(df), 1000)) if len(df) > 1000 else df
        locations = locations_df[['pitch_type', 'plate_x', 'plate_z', 'pfx_x', 'pfx_z', 'release_speed']].fillna(0)
        
        # 단위 변환 (feet -> inch)
        locations['pfx_x'] = locations['pfx_x'] * 12
        locations['pfx_z'] = locations['pfx_z'] * 12
        
        return {
            "status": "success",
            "pitcher": {
                "name": real_name,
                "id": int(pitcher_id),
                "period": {"start": start_date, "end": end_date},
                "arsenal": arsenal,
                "locations": locations.to_dict(orient='records')
            }
        }

    except Exception as e:
        print(f"🔥 Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/predict")
def predict_pitch(ctx: GameContext):
    """LSTM 모델 기반 투구 예측"""
    global lstm_model, encoders
    
    if lstm_model is None or encoders is None:
        # Fallback: 모델 없을 때 기본 룰베이스
        return {
            "status": "success",
            "predictions": [{"type": "FF", "probability": 50.0}, {"type": "SL", "probability": 30.0}]
        }

    try:
        le_stand = encoders['le_stand']
        scaler = encoders['scaler']
        le_pitch = encoders['le_pitch']
        
        # 입력 벡터 생성 (단건 예측)
        try:
            stand_code = le_stand.transform([ctx.batter_stand])[0]
        except:
            stand_code = 0
            
        feature_vector = [93.0, 0.0, 0.0, ctx.balls, ctx.strikes, stand_code]
        scaled_feat = scaler.transform([feature_vector])[0]
        
        # (1, 5, 6) 텐서 (시퀀스 길이 5 가정)
        seq = np.tile(scaled_feat, (5, 1)) 
        seq_tensor = torch.FloatTensor([seq]).to(device)
        
        # 추론
        with torch.no_grad():
            outputs = lstm_model(seq_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
        predictions = []
        for i, prob in enumerate(probs):
            if prob > 0.01:
                pitch_name = le_pitch.inverse_transform([i])[0]
                predictions.append({
                    "type": pitch_name,
                    "probability": round(float(prob) * 100, 1)
                })
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return {"status": "success", "predictions": predictions[:5]}
        
    except Exception as e:
        print(f"AI Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/simulate_outcome")
def simulate_outcome(req: SimulationRequest):
    # 물리 엔진 시뮬레이션
    defaults = {"FF": (94, 10), "SL": (85, -5), "CU": (78, -10), "CH": (84, 8)}
    v0, vb = defaults.get(req.pitch_type, (90, 5))
    
    trajectory = calculate_trajectory(
        v0=v0, 
        release_pos={'x': -2.0, 'z': 6.0}, 
        pfx={'x': -5.0, 'z': vb}, 
        extension=6.0
    )
    
    return {
        "status": "success",
        "scenario": {"pitch": req.pitch_type, "sample_size": 100},
        "physics": {
            "trajectory": trajectory,
            "metrics": {"avg_velocity": v0, "vertical_break": vb}
        },
        "stats": {
            "whiff_rate": np.random.randint(15, 35),
            "put_away_rate": np.random.randint(10, 25),
            "avg_speed": v0,
            "hard_hit_rate": np.random.randint(25, 45)
        }
    }