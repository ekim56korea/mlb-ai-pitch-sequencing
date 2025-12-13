from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# [Updated] 구장 및 환경 정보 (바람 추가)
class StadiumContext(BaseModel):
    temperature: float = 70.0  # 화씨 (F)
    elevation: float = 0.0     # 해발 고도 (ft)
    humidity: float = 50.0     # 습도 (%)
    wind_speed: float = 0.0    # 풍속 (mph) [v6.0]
    wind_direction: float = 0.0 # 풍향 (0~360도) [v6.0]

class GameContext(BaseModel):
    inning: int = 1
    balls: int = 0
    strikes: int = 0
    outs: int = 0
    runner_on_1b: bool = False
    runner_on_2b: bool = False
    runner_on_3b: bool = False
    score_diff: int = 0

# [New] 직전 투구 정보 (터널링 분석용)
class PreviousPitch(BaseModel):
    pitch_type: str
    release_speed: float
    pfx_x: float
    pfx_z: float
    plate_x: float
    plate_z: float

class PitchInput(BaseModel):
    pitch_type: str
    release_speed: float
    release_spin_rate: float
    pfx_x: float = 0.0
    pfx_z: float = 0.0
    extension: float = 6.0
    
    # [Phase 1 New] Advanced Physics Params
    spin_efficiency: float = 100.0 # 스핀 효율 (%)
    seam_lat: float = 0.0          # 솔기 위도 (Seam Latitude, -90~90도)
    seam_long: float = 0.0         # 솔기 경도 (Seam Longitude, 0~360도)
    
    # Target (Auto-Aiming)
    plate_x: Optional[float] = None
    plate_z: Optional[float] = None
    
    # Dynamics
    pitch_count: int = 0
    prev_pitch: Optional[PreviousPitch] = None
    
    env: Optional['StadiumContext'] = None
    context: Optional['GameContext'] = None

class TrajectoryResponse(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    final_x: float
    final_z: float
    approach_angle_v: float
    approach_angle_h: float

class MetricResponse(BaseModel):
    stuff_plus: float
    location_plus: float
    pitching_plus: float
    xRV: float
    details: Optional[Dict[str, Any]] = None # 상세 점수 내역

class BatterInput(BaseModel):
    swing_rate: float
    whiff_rate: float
    chase_rate: float

class BatterAnalysisResponse(BaseModel):
    cluster_id: int
    batter_type: str
    strategy: str