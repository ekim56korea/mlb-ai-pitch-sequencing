from pydantic import BaseModel
from typing import List, Optional

# 입력 데이터 (질문)
class PitchInput(BaseModel):
    pitch_type: str       # 구종 (FF, SL 등)
    release_speed: float  # 구속
    release_spin_rate: float # 회전수
    # 필요한 경우 9-param 전체 추가 가능

class BatterInput(BaseModel):
    # 타자의 시즌 기록을 입력받아 실시간으로 클러스터링할 때 사용
    swing_rate: float
    whiff_rate: float
    chase_rate: float

# 출력 데이터 (대답)
class TrajectoryResponse(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    final_x: float # 홈플레이트 도달 시 x (좌우)
    final_z: float # 홈플레이트 도달 시 z (상하)

class BatterAnalysisResponse(BaseModel):
    cluster_id: int
    batter_type: str # "선구안형", "배드볼히터" 등 설명
    strategy: str    # 간단한 공략 팁