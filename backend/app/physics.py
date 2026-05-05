import numpy as np

def calculate_trajectory(v0, release_pos, pfx, extension):
    """
    [Phase 4 Final] MLB급 정밀 물리 엔진 (Quadratic Drag Model)
    공기 저항을 속도의 제곱에 비례하게 계산하여 실제와 같은 종속(V-end)을 구현합니다.
    
    :param v0: 릴리스 구속 (mph)
    :param release_pos: {'x': float, 'z': float} (ft) - 릴리스 포인트
    :param pfx: {'x': float, 'z': float} (인치) - 무브먼트 (Statcast pfx)
    :param extension: 익스텐션 (ft)
    :return: 정밀 3D 궤적 리스트
    """
    
    # ─── 1. 물리 상수 및 초기 조건 설정 ───
    
    # 단위 변환: mph -> ft/s
    v0_fts = v0 * 1.467
    
    # 릴리스 포인트 (포수 시점 기준 좌표계 변환)
    # x: 좌우 (투수판 중심 0), y: 홈까지 거리, z: 높이
    r0 = np.array([release_pos['x'], 60.5 - extension, release_pos['z']])
    
    # 초기 속도 벡터 (Initial Velocity Vector)
    # 타겟(존 중심)을 향해 던진다고 가정하고 초기 발사각 역산
    target_y = 0 # 홈 플레이트
    flight_distance = r0[1] - target_y
    flight_time_approx = flight_distance / (v0_fts * 0.92) # 1차 추정
    
    # 초기 속도 성분 (vx, vy, vz)
    # vy는 투수->포수 방향이므로 음수 (-)
    vy0 = -v0_fts 
    # x, z 성분은 pfx(무브먼트)를 고려하지 않은 '직선' 기준 발사각
    vx0 = (0 - r0[0]) / flight_time_approx 
    vz0 = (2.5 - r0[2]) / flight_time_approx # 스트라이크 존 높이(2.5ft) 타겟팅 가정
    
    velocity = np.array([vx0, vy0, vz0])
    position = np.array(r0)
    
    # ─── 2. 외력(Force) 파라미터 산출 ───
    
    # 중력 가속도 (ft/s^2)
    GRAVITY = np.array([0, 0, -32.174])
    
    # 마그누스 가속도 (Magnus Acceleration)
    # pfx는 '중력 없는 상태'에서의 순수 휘어짐(인치)
    # 가속도 a = 2 * d / t^2 공식을 이용해 역산 (인치 -> 피트 변환: / 12)
    t_sq = flight_time_approx ** 2
    a_magnus_x = 2 * (pfx['x'] / 12) / t_sq
    a_magnus_z = 2 * (pfx['z'] / 12) / t_sq
    ACCEL_MAGNUS = np.array([a_magnus_x, 0, a_magnus_z])
    
    # 공기 저항 상수 (Drag Factor)
    # F_drag = -C * v * |v|
    # MLB 평균 감속도(8~9mph loss)를 재현하는 실험적 상수 (ft/s 단위)
    DRAG_FACTOR = 5.0e-4 

    # ─── 3. 정밀 시뮬레이션 루프 (Euler Method) ───
    
    trajectory = []
    dt = 0.001 # 0.001초 단위 (1ms) - 기존보다 10배 정밀
    t = 0
    
    # 데이터 포인트 샘플링 간격 (프론트엔드 과부하 방지)
    sample_rate = 10 
    step_count = 0
    
    while position[1] > 0: # 홈 플레이트(y=0) 도달 전까지 반복
        
        # 현재 속력 (Scalar Speed)
        speed = np.linalg.norm(velocity)
        
        # 1) 공기 저항 가속도 (Quadratic Drag)
        # a_drag = -k * v * |v| (방향은 속도의 반대)
        a_drag = -DRAG_FACTOR * speed * velocity
        
        # 2) 전체 가속도 = 중력 + 마그누스 + 공기저항
        total_accel = GRAVITY + ACCEL_MAGNUS + a_drag
        
        # 3) 위치 및 속도 업데이트
        # 위치: r(t+dt) = r(t) + v(t)dt + 0.5a(t)dt^2
        position += velocity * dt + 0.5 * total_accel * (dt ** 2)
        # 속도: v(t+dt) = v(t) + a(t)dt
        velocity += total_accel * dt
        
        t += dt
        step_count += 1
        
        # 시각화용 데이터 저장 (샘플링)
        if step_count % sample_rate == 0:
            trajectory.append({
                "x": round(position[0], 3),
                "y": round(position[1], 3), # 깊이
                "z": round(position[2], 3), # 높이
                "time": round(t, 3),
                "speed_mph": round(speed / 1.467, 1) # 실시간 구속 정보 (UI 표시용)
            })
            
    # 마지막 지점(홈 플레이트) 강제 추가
    trajectory.append({
        "x": round(position[0], 3),
        "y": 0.0,
        "z": round(position[2], 3),
        "time": round(t, 3),
        "speed_mph": round(np.linalg.norm(velocity) / 1.467, 1)
    })

    return trajectory