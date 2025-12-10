from fastapi import FastAPI, HTTPException
from api.engine.physics import PhysicsEngine
from api.schemas import PitchInput, TrajectoryResponse, BatterInput, BatterAnalysisResponse
import pandas as pd
import joblib
import os
import numpy as np
from api.engine.strategy import StrategyEngine

# 1. 앱(서버) 초기화
app = FastAPI(title="Pitch Commander Pro API", version="1.0")
strategy_engine = StrategyEngine()

# 2. 엔진 및 모델 로드 (서버 켜질 때 한 번만 로딩)
physics_engine = PhysicsEngine()

# 저장된 AI 모델 불러오기
MODEL_PATH = os.path.join("api", "engine", "batter_cluster_model.pkl")
if os.path.exists(MODEL_PATH):
    loaded_data = joblib.load(MODEL_PATH)
    cluster_model = loaded_data['model']
    scaler = loaded_data['scaler']
    print("✅ AI 모델 로드 완료")
else:
    print("⚠️ AI 모델 파일이 없습니다. (학습 스크립트를 먼저 실행하세요)")
    cluster_model = None

# --- API 엔드포인트 (기능 정의) ---

@app.get("/")
def health_check():
    """서버 상태 확인용"""
    return {"status": "active", "system": "Pitch Commander Pro"}

@app.post("/simulate/trajectory", response_model=TrajectoryResponse)
def simulate_pitch(pitch: PitchInput):
    """
    [물리 엔진] 투구 정보를 받아 궤적을 계산합니다.
    """
    # Pydantic 모델을 Pandas Series나 딕셔너리로 변환
    # (간단한 시뮬레이션을 위해 9-param 중 일부만 가상으로 생성하거나 입력받음)
    # 여기서는 입력받은 값 외에 물리 엔진에 필요한 값들을 기본값으로 채웁니다.
    
    # 가상의 물리 파라미터 생성 (실제로는 더 정교한 변환 필요)
    mock_data = {
        'vx0': 0, # 정면 승부 가정
        'vy0': -pitch.release_speed * 1.467, # mph -> ft/s 변환
        'vz0': -5.0,
        'ax': 0,
        'ay': 15.0, # 공기 저항
        'az': -32.174,
        'release_pos_x': 0,
        'release_pos_z': 6.0
    }
    
    row = pd.Series(mock_data)
    traj = physics_engine.calculate_trajectory(row)
    
    if len(traj) == 0:
        raise HTTPException(status_code=400, detail="궤적 계산 실패")

    return {
        "x": traj[:, 0].tolist(),
        "y": traj[:, 1].tolist(),
        "z": traj[:, 2].tolist(),
        "final_x": traj[-1][0],
        "final_z": traj[-1][2]
    }

@app.post("/analyze/batter", response_model=BatterAnalysisResponse)
def analyze_batter(batter: BatterInput):
    """
    [AI 엔진] 타자 스탯을 입력받아 성향(Cluster)을 분석합니다.
    """
    if cluster_model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    # 1. 입력 데이터 전처리 (스케일링)
    features = np.array([[batter.swing_rate, batter.whiff_rate, batter.chase_rate]])
    scaled_features = scaler.transform(features)
    
    # 2. 예측
    cluster_id = int(cluster_model.predict(scaled_features)[0])
    
    # 3. 결과 해석 (룰 기반 매핑)
    types = {
        0: ("공격적 컨택터", "존 안쪽 승부"),
        1: ("신중형/수동적", "카운트 잡기 쉬움"),
        2: ("선구안 마스터 (까다로움)", "유인구 자제, 구위로 압박"),
        3: ("공풍기 (헛스윙 머신)", "변화구 적극 활용"),
        4: ("배드볼 히터 (프리 스윙어)", "스트라이크 같은 볼 던지기")
    }
    
    batter_type, strategy = types.get(cluster_id, ("알 수 없음", "데이터 부족"))

    return {
        "cluster_id": cluster_id,
        "batter_type": batter_type,
        "strategy": strategy
    }

@app.post("/recommend/strategy")
def get_pitch_recommendation(cluster_id: int, ball_count: str):
    """
    [전략 엔진] 타자 클러스터와 볼카운트를 기반으로 다음 공을 추천합니다.
    """
    recommendation = strategy_engine.recommend_pitch(cluster_id, ball_count)
    return recommendation