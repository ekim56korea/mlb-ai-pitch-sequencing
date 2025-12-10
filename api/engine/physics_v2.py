import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

class AdvancedPhysicsEngine:
    """
    [V2.0] 공기 역학(Aerodynamics)을 반영한 고정밀 물리 엔진
    - Alan Nathan 교수의 야구 물리학 모델 적용
    - 공기 저항(Drag) 및 마그누스 효과(Magnus Force) 계산
    - 미분방정식 솔버(Runge-Kutta) 사용
    """
    def __init__(self):
        # 환경 상수 (구장별로 다를 수 있음, 여기선 표준값)
        self.RHO = 0.0765     # 공기 밀도 (lb/ft^3) - 해수면 기준 15도
        self.MASS = 5.125 / 16 # 야구공 무게 (lb) -> oz를 lb로 변환
        self.CIRCUMFERENCE = 9.125 / 12 # 야구공 둘레 (ft) -> inch를 ft로 변환
        self.AREA = (self.CIRCUMFERENCE / (2 * np.pi))**2 * np.pi # 단면적 (ft^2)
        self.GRAVITY = 32.174 # 중력 가속도 (ft/s^2)
        
        # 상수 C0 계산 (항력/양력 계산용 계수)
        # Force = 0.5 * rho * Area * C * v^2
        # a = Force / mass = (0.5 * rho * Area / mass) * C * v^2
        self.CONST_K = 0.5 * self.RHO * self.AREA / self.MASS

    def get_drag_coefficient(self, v):
        """
        항력 계수 (Cd): 속도에 따라 변함 (Adair 모델)
        """
        # 간단화된 모델: 100mph 근처에서는 약 0.30 ~ 0.35
        # 더 정교한 모델은 스핀 효율(Spin Efficiency)을 고려해야 함
        return 0.30

    def get_lift_coefficient(self, spin_rate, v):
        """
        양력 계수 (Cl): 회전수(Spin)와 속도(v)의 비율인 'Spin Factor'에 비례
        """
        if v == 0: return 0
        # Spin Factor (S) = (R * omega) / v
        # R: 반지름, omega: 각속도(rad/s), v: 속도(ft/s)
        radius = (self.CIRCUMFERENCE / (2 * np.pi))
        omega = spin_rate * 2 * np.pi / 60 # rpm -> rad/s
        s = (radius * omega) / v
        
        # Nathan's fit for Cl
        cl = 0.87 * s
        return cl

    def equations_of_motion(self, t, state, spin_components):
        """
        미분방정식: dy/dt = v, dv/dt = a (F/m)
        state: [x, y, z, vx, vy, vz]
        """
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # 1. 항력 (Drag Force) 방향: 속도의 반대 방향
        # F_drag = -C_d * v * vector(v)
        cd = self.get_drag_coefficient(v)
        a_drag_x = -self.CONST_K * cd * v * vx
        a_drag_y = -self.CONST_K * cd * v * vy
        a_drag_z = -self.CONST_K * cd * v * vz
        
        # 2. 마그누스 힘 (Magnus Force) 방향: 회전축 x 속도벡터 (Cross Product)
        # 야구공의 회전축(wx, wy, wz)은 입력받아야 함
        wx, wy, wz = spin_components
        spin_rate = np.sqrt(wx**2 + wy**2 + wz**2) # rpm unit (approx)
        
        cl = self.get_lift_coefficient(spin_rate, v)
        
        # 마그누스 가속도 벡터 방향 계산 (Spin X Velocity)
        # 주의: 실제로는 Spin Efficiency 고려 필요. 여기서는 100% 효율 가정.
        # 벡터 외적 (Cross Product)
        magnus_x = wy*vz - wz*vy
        magnus_y = wz*vx - wx*vz
        magnus_z = wx*vy - wy*vx
        
        # 정규화 (Normalize)
        magnus_mag = np.sqrt(magnus_x**2 + magnus_y**2 + magnus_z**2)
        if magnus_mag > 0:
            a_lift_abs = self.CONST_K * cl * v**2
            a_lift_x = a_lift_abs * (magnus_x / magnus_mag)
            a_lift_y = a_lift_abs * (magnus_y / magnus_mag)
            a_lift_z = a_lift_abs * (magnus_z / magnus_mag)
        else:
            a_lift_x, a_lift_y, a_lift_z = 0, 0, 0

        # 3. 중력 (Gravity)
        a_grav_x = 0
        a_grav_y = 0
        a_grav_z = -self.GRAVITY

        # 최종 가속도 합산
        ax_total = a_drag_x + a_lift_x + a_grav_x
        ay_total = a_drag_y + a_lift_y + a_grav_y
        az_total = a_drag_z + a_lift_z + a_grav_z
        
        return [vx, vy, vz, ax_total, ay_total, az_total]

    def calculate_trajectory(self, pitch_row, time_step=0.01):
        """
        Runge-Kutta (RK45) 메서드를 사용하여 궤적 적분
        """
        # 초기 상태 추출
        x0 = pitch_row.get('release_pos_x', 0)
        y0 = 50.0
        z0 = pitch_row.get('release_pos_z', 6.0)
        
        vx0 = pitch_row.get('vx0', 0)
        vy0 = pitch_row.get('vy0', -130) # ft/s approx
        vz0 = pitch_row.get('vz0', -5)
        
        # 회전 성분 추정 (Statcast 데이터에 spin_axis가 없다면 vx, vz로 추정하거나 기본값 사용)
        # 여기서는 간단히:
        # - 포심(FF): Backspin (wx=큰값)
        # - 커브(CB): Topspin (wx=음수)
        # - 슬라이더(SL): Gyro/Side spin (wz, wy)
        spin_rate = pitch_row.get('release_spin_rate', 2200)
        p_type = pitch_row.get('pitch_type', 'FF')
        
        # [약식 회전축 설정] 실제로는 ax, az 데이터로 역산해야 함
        if p_type == 'FF':
            spin_vec = (-spin_rate, 0, 0) # 순수 백스핀 (x축 회전)
        elif p_type == 'CB':
            spin_vec = (spin_rate * 0.8, 0, 0) # 탑스핀 (부호 반대여야 하나 좌표계 따라 다름, 여기선 양력 반대 테스트)
        else:
            spin_vec = (-spin_rate * 0.5, 0, spin_rate * 0.5) # 슬라이더 (횡회전 포함)

        # 초기 상태 벡터 [x, y, z, vx, vy, vz]
        initial_state = [x0, y0, z0, vx0, vy0, vz0]
        
        # 적분 시간 범위 (0초 ~ 1초)
        t_span = (0, 1.0)
        
        # 이벤트 함수: 홈플레이트(y=1.417) 도달 시 중단
        def reach_home(t, y): return y[1] - 1.417
        reach_home.terminal = True
        reach_home.direction = -1

        # 미분방정식 풀기 (solve_ivp)
        sol = solve_ivp(
            fun=lambda t, y: self.equations_of_motion(t, y, spin_vec),
            t_span=t_span,
            y0=initial_state,
            events=reach_home,
            max_step=time_step,
            method='RK45'
        )
        
        # 결과 변환 (N, 3) -> x, y, z
        # sol.y 는 (6, N) 형태이므로 전치(Transpose) 필요
        trajectory = sol.y[:3, :].T 
        return trajectory