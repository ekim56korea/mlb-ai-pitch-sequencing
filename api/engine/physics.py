import numpy as np
import pandas as pd

class PhysicsEngine:
    """
    Statcast 데이터를 기반으로 야구공의 3D 궤적을 계산하는 엔진입니다.
    SRS 요구사항 REQ-PHY-01(Trajectory Simulation)을 준수합니다.
    """
    
    def __init__(self):
        # 야구장 좌표계 설정
        # y=50ft(약 15m) 지점에서 투구 시작 -> y=0(홈플레이트) 도달
        self.MOUND_Y = 54.5  # 투수판 거리 (피트)
        self.HOME_PLATE_Y = 1.417 # 홈플레이트 앞쪽 끝 (피트, PITCHf/x 기준)

    def calculate_trajectory(self, pitch_row: pd.Series, time_step: float = 0.01):
        """
        투구 데이터 1개(row)를 받아서 시간별 (x, y, z) 위치를 계산합니다.
        
        Args:
            pitch_row: Statcast 데이터의 한 행(Row)
            time_step: 시간 간격 (초 단위, 작을수록 정교함)
        
        Returns:
            궤적 데이터 (List of x, y, z)
        """
        
        # 1. 필요한 물리 파라미터 가져오기 (데이터가 비어있으면 0으로 처리)
        # s: 위치(Space), v: 속도(Velocity), a: 가속도(Acceleration)
        try:
            x0 = pitch_row.get('release_pos_x', 0)
            y0 = 50.0  # Statcast 기준 릴리스 포인트 y는 보통 50으로 가정
            z0 = pitch_row.get('release_pos_z', 0)
            
            vx0 = pitch_row.get('vx0', 0)
            vy0 = pitch_row.get('vy0', 0) # 투수 -> 포수 방향이라 음수(-) 값임
            vz0 = pitch_row.get('vz0', 0)
            
            ax = pitch_row.get('ax', 0)
            ay = pitch_row.get('ay', 0)
            az = pitch_row.get('az', -32.174) # 데이터 없으면 중력가속도 기본값
            
        except Exception as e:
            print(f"⚠️ 데이터 읽기 오류: {e}")
            return []

        # 2. 이동 시간(Flight Time) 계산
        # 공식: 거리 = 속도 * 시간  => 시간 = 거리 / 속도
        # y축 거리: (50 - 1.417)
        # 간단한 근사치 계산 (등가속도 운동 공식 적용 전 근사값)
        if vy0 == 0: return [] # 속도가 0이면 계산 불가
        
        start_y = y0
        end_y = self.HOME_PLATE_Y
        
        # 3. 궤적 시뮬레이션 루프
        trajectory = []
        t = 0.0
        
        current_x, current_y, current_z = x0, y0, z0
        
        # 공이 홈플레이트(y <= 1.417)를 지날 때까지 반복
        while current_y > end_y:
            # 물리 공식 (등가속도 운동): 위치 = 초기위치 + 속도*t + 0.5*가속도*t^2
            # x(t)
            current_x = x0 + vx0 * t + 0.5 * ax * (t**2)
            # y(t)
            current_y = y0 + vy0 * t + 0.5 * ay * (t**2)
            # z(t)
            current_z = z0 + vz0 * t + 0.5 * az * (t**2)
            
            trajectory.append([current_x, current_y, current_z])
            t += time_step # 시간 흐름
            
        return np.array(trajectory)

    def get_pitch_location_at_plate(self, pitch_row):
        """
        공이 홈플레이트에 도달했을 때의 (x, z) 위치, 즉 스트라이크 존 위치를 반환
        """
        # Statcast 데이터에 이미 'plate_x', 'plate_z'가 있지만, 
        # 물리 엔진 검증을 위해 직접 계산한 값과 비교할 수도 있습니다.
        # 여기서는 간단히 제공된 데이터를 반환합니다.
        return pitch_row.get('plate_x'), pitch_row.get('plate_z')