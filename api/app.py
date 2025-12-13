from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import pandas as pd
import asyncio
import numpy as np

# [Engines]
from api.engine.physics_v3 import HyperPhysicsEngine     # Phase 2: Math-based Physics
from api.engine.metrics import MetricsEngine             # Phase 3: Metrics & Tunneling
from api.engine.strategy_rl import RLStrategyEngine      # Phase 2-4: Deep Intelligence
from api.engine.data_loader import PlayerDataLoader      # Phase 1: Optimized Loader
from api.engine.strategy_transfer import StrategyTransferEngine
from api.engine.analytics import VolumetricAnalytics     # Phase 3: Volumetric Lab

# [Schemas]
from api.schemas import PitchInput, TrajectoryResponse, MetricResponse, GameContext

app = FastAPI(title="Pitch Commander Pro v7.0", version="7.0 (Zero-Cost Edition)")

# ==================== [Initialize Engines] ====================
physics_engine = HyperPhysicsEngine()
metrics_engine = MetricsEngine()
rl_engine = RLStrategyEngine()
data_loader = PlayerDataLoader()
transfer_engine = StrategyTransferEngine()
analytics_engine = VolumetricAnalytics()

# ==================== [Endpoints] ====================

@app.post("/simulate/trajectory")
def simulate_pitch(pitch: PitchInput):
    """
    [Phase 2] 궤적 시뮬레이션 및 물리 파라미터 역산
    - Trajectory Calculation (3-DOF)
    - Spin Parameter Estimation (Efficiency, Gyro, Axis)
    - Contact Physics (Exit Velocity, Distance)
    """
    try:
        pitch_data = pitch.dict()
        
        # 1. 궤적 계산
        traj, vaa, haa = physics_engine.calculate_trajectory(pitch_data, pitch.env)
        if len(traj) == 0: 
            raise HTTPException(status_code=400, detail="Trajectory calculation failed")
        
        # 2. [Phase 2] 스핀 파라미터 역산 (Reverse Engineering)
        eff, gyro, axis = physics_engine.estimate_spin_parameters(
            pitch.release_speed, pitch.release_spin_rate, pitch.pfx_x, pitch.pfx_z
        )
        
        # 3. [Phase 2] 타격 결과 시뮬레이션 (Collision Physics)
        # 타겟 지점에 공이 맞았다고 가정
        contact_res = physics_engine.calculate_contact_outcome(
            pitch.release_speed, pitch.plate_x or 0.0, pitch.plate_z or 2.5
        )
        
        return {
            "x": traj[:, 0].tolist(), 
            "y": traj[:, 1].tolist(), 
            "z": traj[:, 2].tolist(),
            "final_x": traj[-1][0], 
            "final_z": traj[-1][2],
            "approach_angle_v": vaa, 
            "approach_angle_h": haa,
            # [New] Physics Lab Data
            "physics_est": {
                "efficiency": eff,
                "gyro_degree": gyro,
                "spin_axis": axis
            },
            "contact_est": contact_res
        }
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/metrics", response_model=MetricResponse)
def analyze_metrics(pitch: PitchInput):
    """
    [Phase 3] Pitching+ 평가 및 터널링 분석
    - Stuff+ / Location+ / Pitching+
    - Tunneling Bonus (Ghost Trail Comparison)
    """
    try:
        ctx = pitch.context.dict() if pitch.context else {'balls': 0, 'strikes': 0}
        ctx['stand'] = 'R' # 기본값
        pitch_data = pitch.dict()

        # 1. Physics Calc (위치 보정용)
        if pitch.plate_x is not None and pitch.plate_z is not None:
            pitch_data['plate_x'] = pitch.plate_x
            pitch_data['plate_z'] = pitch.plate_z
        else:
            traj, _, _ = physics_engine.calculate_trajectory(pitch_data, pitch.env)
            if len(traj) > 0:
                pitch_data['plate_x'] = traj[-1][0]
                pitch_data['plate_z'] = traj[-1][2]
            else: 
                pitch_data['plate_x'], pitch_data['plate_z'] = 0.0, 2.5

        # 2. [Phase 3] 터널링 거리 계산
        tunnel_bonus = 0.0
        if pitch.prev_pitch:
            COMMIT_Y = 23.8 # 타자가 결심하는 거리 (ft)
            
            # 현재 투구 위치
            curr_x, curr_z = physics_engine.get_position_at_y(pitch_data, COMMIT_Y, pitch.env)
            
            # 직전 투구 위치 (prev_pitch 데이터 기반 시뮬레이션)
            prev_data = pitch.prev_pitch.dict()
            prev_x, prev_z = physics_engine.get_position_at_y(prev_data, COMMIT_Y, pitch.env)
            
            # 보너스 산출
            tunnel_bonus = metrics_engine.calculate_tunneling_bonus((curr_x, curr_z), (prev_x, prev_z))
        
        pitch_data['tunneling_bonus'] = tunnel_bonus
        
        # 3. 종합 점수 계산
        result = metrics_engine.calculate_pitching_plus(pitch_data, ctx)
        return result
        
    except Exception as e: 
        print(f"Metrics Error: {e}")
        return {"stuff_plus": 100.0, "location_plus": 100.0, "pitching_plus": 100.0, "xRV": 0.0}

@app.post("/recommend/context")
def recommend_context(context: GameContext, arsenal: List[str] = Query(None)):
    """
    [Phase 2-4] 심층 지능 전략 추천
    - Phase 2: Guess Hitting (노림수 예측)
    - Phase 3: Swing Probability (반응성 예측)
    - Phase 4: Leverage Index (상황 중요도)
    """
    if not arsenal: arsenal = ["FF", "SL", "CH", "CB"]
    
    # RL 엔진을 통해 시퀀스 예측 (Context-Aware)
    sequence = rl_engine.predict_sequence(arsenal, context.dict())
    
    # UI 호환성을 위한 좌표 매핑
    setup = sequence['recommended_pitch']
    setup_loc_desc = sequence['location']
    finish = sequence['next_pitch']
    
    loc_map = {"High-In": (-0.8, 3.5), "High-Out": (0.8, 3.5), 
               "Low-In": (-0.8, 1.5), "Low-Out": (0.8, 1.5), 
               "Middle": (0.0, 2.5), "Middle-Low": (0.0, 2.0)}
    
    tx, tz = loc_map.get(setup_loc_desc.split(' ')[0], (0.0, 2.5))
    fx, fz = loc_map.get(finish['location'].split(' ')[0], (0.0, 1.5))
    
    return {
        "strategy_name": sequence['strategy_name'],
        "recommended_pitch": setup,
        "location_desc": setup_loc_desc,
        "target_x": tx, 
        "target_z": tz,
        "reasoning": sequence['reasoning'],
        # [New Data Points]
        "guess_probs": sequence['guess_probs'],      # Phase 2
        "swing_prob": sequence['swing_prob'],        # Phase 3
        "leverage_index": sequence.get('leverage_index', 1.0), # Phase 4
        "next_pitch": {
            "pitch": finish['pitch'], 
            "location": finish['location'], 
            "target_x": fx, 
            "target_z": fz
        }
    }

@app.post("/load/matchup")
async def load_matchup_data(pitcher_name: str, batter_name: str, start_dt: str = None, end_dt: str = None):
    """
    [Phase 1] 비동기 데이터 로딩 (Async IO)
    - ThreadPool을 사용하여 무거운 I/O 작업을 메인 루프에서 분리
    - Parquet 캐싱 활용으로 속도 최적화
    """
    try:
        # 1. Pitcher Search & Load (Async)
        p_id = await asyncio.to_thread(data_loader.find_player_id, pitcher_name)
        if not p_id: 
            return {"status": "error", "message": f"Pitcher '{pitcher_name}' not found."}
        
        p_df = await asyncio.to_thread(data_loader.load_pitcher_data, p_id, start_dt, end_dt)
        
        # 2. Batter Search & Load (Async)
        b_id = await asyncio.to_thread(data_loader.find_player_id, batter_name)
        b_df = pd.DataFrame()
        if b_id:
            b_df = await asyncio.to_thread(data_loader.load_batter_data, b_id, start_dt, end_dt)

        # 3. Arsenal Analysis & Archetype (CPU Bound)
        arsenal = {}
        similar_pitcher = None
        
        if not p_df.empty and 'pitch_type' in p_df.columns:
            summary = p_df.groupby('pitch_type').agg({
                'release_speed': 'mean', 'release_spin_rate': 'mean',
                'pfx_x': 'mean', 'pfx_z': 'mean', 'release_extension': 'mean',
                'pitch_type': 'count'
            }).rename(columns={'pitch_type': 'count'}).to_dict('index')
            arsenal = summary
            
            # Archetype Finding
            try:
                main_pitch = max(summary.items(), key=lambda x: x[1]['count'])
                stats = main_pitch[1]
                similar_pitcher = transfer_engine.find_similar_pitcher(
                    stats['release_speed'], stats['pfx_x'], stats['pfx_z']
                )
            except: pass

        return {
            "status": "success",
            "pitcher": {
                "name": pitcher_name, "id": int(p_id), 
                "data_count": len(p_df), "arsenal": arsenal, 
                "archetype": similar_pitcher
            },
            "batter": {
                "name": batter_name, "id": int(b_id) if b_id else 0, 
                "data_count": len(b_df)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze/volumetric")
def analyze_volumetric(batter_name: str, start_dt: str = None, end_dt: str = None):
    """
    [Phase 3] 3D 볼류메트릭 핫존 분석 요청
    """
    try:
        b_id = data_loader.find_player_id(batter_name)
        if not b_id:
            return {"status": "error", "data": []}
            
        b_df = data_loader.load_batter_data(b_id, start_dt, end_dt)
        
        # 3D 핫존 생성
        volumetric_data = analytics_engine.generate_volumetric_hotzone(b_df)
        
        return {"status": "success", "data": volumetric_data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}