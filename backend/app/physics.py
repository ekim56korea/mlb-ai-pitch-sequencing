import numpy as np

def calculate_trajectory(v0, release_pos, pfx, extension, env_params=None):
    """
    Constant Acceleration Model (단순화된 물리 모델)
    - v0: 초구 속도 (mph)
    - release_pos: {'x': ft, 'z': ft} 릴리스 포인트
    - pfx: {'x': inch, 'z': inch} 무브먼트
    - extension: 익스텐션 (ft)
    """
    # 1. 단위 변환 및 초기값 설정
    v0_fps = v0 * 1.467  # mph -> fps (feet per second)
    start_y = 60.5 - extension  # 릴리스 지점 (포수까지의 거리)
    flight_time = start_y / (v0_fps * 0.96)  # 비행 시간 추정 (공기저항 고려 감속)
    
    # 가속도 계산 (무브먼트 기반)
    # pfx는 inch 단위이므로 feet로 변환 (/12)
    # 중력 가속도: -32.174 ft/s^2
    acc_x = (2 * ((pfx['x'] / 12) * -1)) / (flight_time ** 2)
    acc_z = -32.174 + (2 * (pfx['z'] / 12) / (flight_time ** 2))
    
    # 2. 궤적 포인트 생성 (40등분)
    points = []
    steps = 40
    
    for i in range(steps + 1):
        t = (i / steps) * flight_time
        
        # 등가속도 운동 방정식: s = ut + 0.5at^2
        # Y축 (거리): 등속 운동에 가깝지만 공기저항으로 약간 감속
        curr_y = start_y - (v0_fps * t) + (0.5 * -5.0 * t * t) # -5.0은 드래그 계수 근사치
        
        # X축 (좌우): 초기 속도(직선 가정) + 무브먼트 가속도
        # 시작점(rx)에서 타겟(px) 방향으로 쏜다고 가정하고 무브먼트 추가
        curr_x = (release_pos['x'] * -1) + (0.5 * acc_x * t * t) # * -1은 포수 시점 보정
        
        # Z축 (상하): 초기 속도 + 중력/무브먼트 가속도
        # 보통 투수는 아래로 던지므로 초기 수직 속도(vy0)가 음수임
        vy0 = (release_pos['z'] - 2.5) / flight_time # 스트라이크존(2.5ft)을 향해 던짐
        curr_z = release_pos['z'] - (vy0 * t) + (0.5 * acc_z * t * t)
        
        # 바닥 뚫기 방지
        if curr_z < 0: curr_z = 0.05
        
        points.append({"x": curr_x, "y": curr_y, "z": curr_z})
        
    return points