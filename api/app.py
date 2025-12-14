from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 데이터 & 모델 로드 ───
CSV_FILE = "savant_data.csv"
MODEL_FILE = "pitch_predictor.pkl"

global_df = pd.DataFrame()
ai_model = None

# 1. 데이터 로드
if os.path.exists(CSV_FILE):
    print(f"📊 Loading data from {CSV_FILE}...")
    try:
        global_df = pd.read_csv(CSV_FILE, usecols=[
            'player_name', 'pitch_type', 'release_speed', 
            'balls', 'strikes', 'pfx_x', 'pfx_z', 
            'plate_x', 'plate_z',
            'release_pos_x', 'release_pos_z', 'release_extension',
            'stand', 'outs_when_up', 'inning',
            'description', 'events', 'launch_speed' # 👈 시뮬레이션용 컬럼 필수!
        ])
        global_df = global_df.dropna(subset=['pitch_type', 'player_name'])
        print("✅ Data Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")

# 2. AI 모델 로드
if os.path.exists(MODEL_FILE):
    try:
        ai_model = joblib.load(MODEL_FILE)
        print("✅ AI Brain Connected!")
    except Exception as e:
        print(f"❌ AI Load Failed: {e}")
else:
    print("⚠️ No AI model found.")


# ─── 입력 모델 정의 ───
class GameContext(BaseModel):
    pitcher_name: str
    batter_stand: str 
    balls: int
    strikes: int
    outs: int
    inning: int

# 🌟 [New] 시뮬레이션 요청 모델
class SimulationRequest(BaseModel):
    pitcher_name: str
    pitch_type: str
    batter_stand: str


# ─── API 엔드포인트 ───

@app.post("/load/matchup")
def load_matchup(pitcher_name: str, batter_name: str):
    if global_df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded")

    pitcher_data = global_df[global_df['player_name'].str.contains(pitcher_name, case=False, na=False)]
    
    if pitcher_data.empty:
        return {"status": "error", "message": "Pitcher not found"}

    total_pitches = len(pitcher_data)
    
    stats = pitcher_data.groupby('pitch_type').agg({
        'release_speed': 'mean',
        'pfx_x': 'mean', 'pfx_z': 'mean',
        'plate_x': 'mean', 'plate_z': 'mean',
        'release_pos_x': 'mean', 'release_pos_z': 'mean', 'release_extension': 'mean',
        'pitch_type': 'count'
    }).rename(columns={'pitch_type': 'count'})
    
    arsenal = {}
    for pitch_type, row in stats.iterrows():
        arsenal[pitch_type] = {
            "count": int(row['count']),
            "release_speed": round(row['release_speed'], 1),
            "pfx_x": round(row['pfx_x'], 2) if not pd.isna(row['pfx_x']) else 0,
            "pfx_z": round(row['pfx_z'], 2) if not pd.isna(row['pfx_z']) else 0,
            "plate_x": round(row['plate_x'], 2) if not pd.isna(row['plate_x']) else 0,
            "plate_z": round(row['plate_z'], 2) if not pd.isna(row['plate_z']) else 2.5,
            "release_pos_x": round(row['release_pos_x'], 2) if not pd.isna(row['release_pos_x']) else -1.5,
            "release_pos_z": round(row['release_pos_z'], 2) if not pd.isna(row['release_pos_z']) else 6.0,
            "release_extension": round(row['release_extension'], 1) if not pd.isna(row['release_extension']) else 6.0,
            "percentage": round((row['count'] / total_pitches) * 100, 1)
        }

    # 차트 및 히트맵용 데이터
    locations = pitcher_data[[
        'pitch_type', 'plate_x', 'plate_z', 'release_speed', 'pfx_x', 'pfx_z'
    ]].dropna().to_dict(orient='records')

    return {
        "status": "success",
        "pitcher": {
            "name": pitcher_data['player_name'].iloc[0],
            "data_count": total_pitches,
            "arsenal": arsenal,
            "locations": locations
        }
    }

@app.post("/predict")
def predict_pitch(ctx: GameContext):
    if ai_model is None:
        return {"error": "AI Model not loaded"}

    input_data = pd.DataFrame([{
        'player_name': ctx.pitcher_name,
        'stand': ctx.batter_stand,
        'balls': ctx.balls,
        'strikes': ctx.strikes,
        'outs_when_up': ctx.outs,
        'inning': ctx.inning
    }])

    try:
        probs = ai_model.predict_proba(input_data)[0]
        classes = ai_model.classes_

        predictions = []
        for pitch, prob in zip(classes, probs):
            if prob > 0.05:
                predictions.append({"type": pitch, "probability": round(prob * 100, 1)})
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return {"status": "success", "predictions": predictions[:3]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 🌟 [New] 시뮬레이션 엔드포인트
@app.post("/simulate_outcome")
def simulate_outcome(req: SimulationRequest):
    if global_df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # 이름 검색 (부분 일치 허용)
    filtered = global_df[
        (global_df['player_name'].str.contains(req.pitcher_name, case=False, na=False)) & 
        (global_df['pitch_type'] == req.pitch_type) &
        (global_df['stand'] == req.batter_stand)
    ]

    if filtered.empty:
        return {"status": "error", "message": "No historical data"}

    total = len(filtered)
    
    # 헛스윙 비율
    whiffs = filtered[filtered['description'].isin(['swinging_strike', 'swinging_strike_blocked'])].shape[0]
    swings = filtered[filtered['description'].isin(['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked'])].shape[0]
    whiff_rate = round((whiffs / swings * 100), 1) if swings > 0 else 0
    
    # 결정구(삼진) 비율
    two_strikes = filtered[filtered['strikes'] == 2]
    strikeouts = two_strikes[two_strikes['events'].isin(['strikeout', 'strikeout_double_play'])].shape[0]
    put_away_rate = round((strikeouts / len(two_strikes) * 100), 1) if len(two_strikes) > 0 else 0

    # 강한 타구 비율
    hard_hits = filtered[filtered['launch_speed'] >= 95].shape[0]
    batted_balls = filtered[filtered['description'] == 'hit_into_play'].shape[0]
    hard_hit_rate = round((hard_hits / batted_balls * 100), 1) if batted_balls > 0 else 0

    return {
        "status": "success",
        "scenario": {
            "pitcher": req.pitcher_name,
            "pitch": req.pitch_type,
            "batter": "Righty" if req.batter_stand == 'R' else "Lefty",
            "sample_size": total
        },
        "stats": {
            "whiff_rate": whiff_rate,
            "put_away_rate": put_away_rate,
            "hard_hit_rate": hard_hit_rate,
            "avg_speed": round(filtered['release_speed'].mean(), 1)
        }
    }