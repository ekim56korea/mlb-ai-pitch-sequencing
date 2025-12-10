from pydantic import BaseModel
from typing import List, Optional

# [New] 구장 및 환경 정보
class GameContext(BaseModel):
    inning: int = 1
    balls: int = 0
    strikes: int = 0
    outs: int = 0
    runner_on_1b: bool = False
    runner_on_2b: bool = False
    runner_on_3b: bool = False
    score_diff: int = 0 # 점수차 (양수면 이기고 있음)

class PitchInput(BaseModel):
    pitch_type: str
    release_speed: float
    release_spin_rate: float
    pfx_x: float = 0.0
    pfx_z: float = 0.0
    extension: float = 6.0
    env: Optional['StadiumContext'] = None
    
    # [New] 시뮬레이션 시 상황 정보도 함께 전달
    context: Optional[GameContext] = None
    
class StadiumContext(BaseModel):
    temperature: float = 70.0  # 화씨 (F)
    elevation: float = 0.0     # 해발 고도 (ft) - 예: 쿠어스필드 5200
    humidity: float = 50.0     # 습도 (%)

class PitchInput(BaseModel):
    pitch_type: str       # 구종
    release_speed: float  # 구속
    release_spin_rate: float
    pfx_x: float = 0.0    # 수평 무브먼트 (인치)
    pfx_z: float = 0.0    # 수직 무브먼트 (인치)
    extension: float = 6.0 # 익스텐션
    
    # [New] 환경 변수 (Optional)
    env: Optional[StadiumContext] = None

class BatterInput(BaseModel):
    swing_rate: float
    whiff_rate: float
    chase_rate: float

class TrajectoryResponse(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    final_x: float
    final_z: float
    # [New] 물리적 수치
    approach_angle_v: float # VAA
    approach_angle_h: float # HAA

class BatterAnalysisResponse(BaseModel):
    cluster_id: int
    batter_type: str
    strategy: str

class MetricResponse(BaseModel):
    stuff_plus: float     # 구위 점수 (100점 평균)
    location_plus: float  # 제구 점수 (100점 평균) - 추후 구현
    xRV: float            # 예상 득점 가치