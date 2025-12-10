from fastapi import FastAPI, HTTPException, Query
from typing import List
from collections import deque
import pandas as pd
import joblib
import os
import numpy as np

# --- [V3.0] 엔진 및 스키마 임포트 ---
from api.engine.physics_v3 import HyperPhysicsEngine
from api.engine.metrics import MetricsEngine
from api.engine.strategy import StrategyEngine
from api.engine.data_loader import PlayerDataLoader
from api.schemas import PitchInput, TrajectoryResponse, MetricResponse, BatterInput, BatterAnalysisResponse, GameContext

# 앱 초기화
app = FastAPI(title="Pitch Commander Pro V4.0", version="4.0 (Tactical)")

# --- 엔진 초기화 ---
physics_engine = HyperPhysicsEngine()
metrics_engine = MetricsEngine()
strategy_engine = StrategyEngine()
data_loader = PlayerDataLoader()

# 전역 변수
live_pitch_buffer = deque(maxlen=10)
current_context = {"pitcher_df": None, "batter_df": None}

# AI 모델 로드
MODEL_PATH = os.path.join("api", "engine", "batter_cluster_model.pkl")
cluster_model = None
scaler = None

if os.path.exists(MODEL_PATH):
    try:
        loaded_data = joblib.load(MODEL_PATH)
        cluster_model = loaded_data['model']
        scaler = loaded_data['scaler']
        print("✅ AI 모델 로드 완료 (Batter Clustering)")
    except Exception as e:
        print(f"⚠️ 모델 로드 중 오류 발생: {e}")
else:
    print("⚠️ AI 모델 파일이 없습니다.")

# --- API 엔드포인트 ---

@app.get("/")
def health_check():
    return {"status": "active", "version": "v4.0"}

# [1] 궤적 시뮬레이션
@app.post("/simulate/trajectory", response_model=TrajectoryResponse)
def simulate_pitch(pitch: PitchInput):
    try:
        pitch_data = pitch.dict()
        traj, vaa, haa = physics_engine.calculate_trajectory(pitch_data, pitch.env)
        
        if len(traj) == 0:
            raise HTTPException(status_code=400, detail="궤적 계산 실패")

        return {
            "x": traj[:, 0].tolist(),
            "y": traj[:, 1].tolist(),
            "z": traj[:, 2].tolist(),
            "final_x": traj[-1][0],
            "final_z": traj[-1][2],
            "approach_angle_v": vaa,
            "approach_angle_h": haa
        }
    except Exception as e:
        print(f"Simulation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# [2] 구위 평가
@app.post("/analyze/metrics", response_model=MetricResponse)
def analyze_metrics(pitch: PitchInput):
    try:
        score = metrics_engine.calculate_stuff_plus(pitch.dict())
        return {
            "stuff_plus": score,
            "location_plus": 0.0,
            "xRV": -0.05
        }
    except Exception as e:
        print(f"Metrics Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# [3] 타자 분석
@app.post("/analyze/batter", response_model=BatterAnalysisResponse)
def analyze_batter(batter: BatterInput):
    if cluster_model is None:
        raise HTTPException(status_code=500, detail="AI 모델 로드 실패")
    
    features = np.array([[batter.swing_rate, batter.whiff_rate, batter.chase_rate]])
    scaled_features = scaler.transform(features)
    cluster_id = int(cluster_model.predict(scaled_features)[0])
    
    types = {
        0: ("공격적 컨택터", "존 안쪽 승부 유효"),
        1: ("신중형/수동적", "카운트 선점 필수"),
        2: ("선구안 마스터", "유인구 자제, 구위 승부"),
        3: ("공풍기", "하이 패스트볼 또는 떨어지는 변화구"),
        4: ("배드볼 히터", "존 바깥 유인구 적극 활용")
    }
    b_type, strat = types.get(cluster_id, ("알 수 없음", "데이터 부족"))
    return {"cluster_id": cluster_id, "batter_type": b_type, "strategy": strat}

# [4] 전략 추천 (수정된 부분: Query 파라미터 명시)
@app.post("/recommend/context")
def recommend_context(
    context: GameContext, 
    arsenal: List[str] = Query(None)  # [수정] 명시적 쿼리 파라미터 선언
):
    """
    [Strategy V4] 경기 상황을 입력받아 최적의 투구를 추천
    """
    if not arsenal:
        arsenal = ["FF", "SL", "CH", "CB"]
        
    recommendation = strategy_engine.recommend_pitch(arsenal, context.dict())
    return recommendation

# [5] 데이터 로더 (선수 실데이터)
@app.post("/load/matchup")
def load_matchup_data(pitcher_name: str, batter_name: str, start_dt: str = None, end_dt: str = None):
    try:
        p_last, p_first = pitcher_name.split()
        b_last, b_first = batter_name.split()
    except:
        raise HTTPException(status_code=400, detail="이름 포맷 오류 (성 이름)")

    # 1. 투수 로드
    p_id = data_loader.find_player_id(p_last, p_first)
    if not p_id:
        return {"status": "error", "message": f"투수 {pitcher_name} 못 찾음"}
    
    p_df = data_loader.load_pitcher_data(p_id, start_dt, end_dt)
    current_context["pitcher_df"] = p_df

    # 2. 타자 로드
    b_id = data_loader.find_player_id(b_last, b_first)
    if not b_id:
        return {"status": "error", "message": f"타자 {batter_name} 못 찾음"}
        
    b_df = data_loader.load_batter_data(b_id, start_dt, end_dt)
    current_context["batter_df"] = b_df

    # 3. 구종 통계
    arsenal = {}
    if not p_df.empty and 'pitch_type' in p_df.columns:
        summary = p_df.groupby('pitch_type').agg({
            'release_speed': 'mean',
            'release_spin_rate': 'mean',
            'pfx_x': 'mean',
            'pfx_z': 'mean',
            'release_extension': 'mean'
        }).to_dict('index')
        arsenal = summary

    return {
        "status": "success",
        "pitcher": {"name": pitcher_name, "id": int(p_id), "data_count": len(p_df), "arsenal": arsenal},
        "batter": {"name": batter_name, "id": int(b_id), "data_count": len(b_df)}
    }

# [6] 라이브 데이터 (기존)
@app.post("/live/ingest")
def ingest_live_data(pitch: PitchInput):
    live_pitch_buffer.append(pitch)
    return {"status": "received"}

@app.get("/live/latest")
def get_latest_pitch():
    if not live_pitch_buffer:
        return {"status": "empty"}
    return live_pitch_buffer[-1]