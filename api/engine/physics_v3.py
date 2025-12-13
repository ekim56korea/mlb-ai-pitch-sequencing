import numpy as np
from scipy.integrate import solve_ivp

class HyperPhysicsEngine:
    """
    [v7.0 Phase 2] Hyper-Physics Engine (Zero-Cost Math Edition)
    - Spin Axis & Efficiency Estimation (Math-based Reverse Engineering)
    - Collision Physics (Bat-Ball Interaction)
    - 3-DOF Trajectory with Environmental Factors
    """
    def __init__(self):
        self.GRAVITY = 32.174
        self.MASS = 5.125 / 16
        self.CIRCUMFERENCE = 9.125 / 12
        self.AREA = (self.CIRCUMFERENCE / (2 * np.pi))**2 * np.pi
        
        # [Collision Constants] 앨런 네이선 충돌 모델 상수
        self.BAT_MASS_OZ = 31.0
        self.COR = 0.5  # 반발 계수 (나무 배트 기준)
        self.BAT_SPEED_AVG = 72.0 # MLB 평균 배트 스피드 (mph)

    def calculate_air_density(self, temp_f, elevation_ft, humidity):
        """환경 변수에 따른 공기 밀도 계산"""
        temp_r = temp_f + 459.67
        pressure_hg = 29.92 * np.exp(-0.0000366 * elevation_ft)
        svp = 10 ** (0.7859 + 0.03477 * temp_f) / (1 + 0.00412 * temp_f)
        vp = (humidity / 100.0) * svp
        rho = 1.2929 * (273.15 / (5/9 * (temp_f - 32) + 273.15)) * (pressure_hg * 25.4) / 760
        return rho * 0.062428

    def estimate_spin_parameters(self, v_mph, spin_rate, pfx_x, pfx_z):
        """
        [Phase 2.1] 회전 효율(Efficiency) 및 자이로 각도 수학적 추정
        - 원리: 실제 무브먼트(PFX) vs 이론상 최대 무브먼트(Max Magnus) 비율 계산
        """
        if spin_rate == 0: return 0.0, 0.0, 0.0

        # 1. 이론상 최대 양력 계수 (Pure Backspin일 때)
        # Cl ~ 0.87 * (rw/v)
        radius = self.CIRCUMFERENCE / (2 * np.pi)
        v_ft_s = v_mph * 1.467
        omega = spin_rate * 2 * np.pi / 60
        spin_ratio = (radius * omega) / v_ft_s
        cl_max = 1 / (2.32 + (0.4 / spin_ratio)) # Adair Model

        # 2. 실제 발생한 가속도 (PFX 기반 역산)
        # pfx는 인치 단위, 0.4초 기준 무브먼트라고 가정 (간소화)
        # a = 2 * d / t^2
        t_ref = 0.4
        acc_x = (pfx_x / 12.0) * 2 / (t_ref**2)
        acc_z = (pfx_z / 12.0) * 2 / (t_ref**2) - self.GRAVITY # 중력 보정 제거
        
        acc_total = np.sqrt(acc_x**2 + acc_z**2)
        
        # 3. 실제 양력 계수
        # F = ma = 0.5 * rho * A * Cl * v^2
        rho_std = 0.074
        const_k = 0.5 * rho_std * self.AREA / self.MASS
        cl_actual = acc_total / (const_k * v_ft_s**2)

        # 4. 효율 계산 (Actual / Max)
        efficiency = min(1.0, cl_actual / (cl_max + 1e-9))
        
        # 5. 자이로 각도 (Gyro Degree)
        # Efficiency = cos(theta)
        gyro_deg = np.degrees(np.arccos(efficiency))
        
        # 6. 회전축 (Spin Axis - 2D Clock)
        # arctan2(z, x) -> deg conversion
        axis_rad = np.arctan2(pfx_x, pfx_z) # 야구는 12시가 0도 기준이라 z, x 순서 유의
        axis_deg = np.degrees(axis_rad)
        
        return round(efficiency * 100, 1), round(gyro_deg, 1), round(axis_deg, 1)

    def calculate_contact_outcome(self, pitch_speed_mph, plate_x, plate_z):
        """
        [Phase 2.2] 충돌 물리학 (Collision Physics)
        - 투구 속도와 위치에 따른 예상 타구 속도(EV) 및 비거리 시뮬레이션
        - 정타(Sweet Spot) 가정 시의 물리적 한계치 계산
        """
        # 1. 충돌 효율 (Collision Efficiency 'q')
        # q approx 0.2 for wood bat
        q = 0.2 
        
        # 2. 배트 스피드 보정 (존 중앙일수록 빠름, 바깥쪽은 느림 가정)
        dist_from_center = np.sqrt(plate_x**2 + (plate_z - 2.5)**2)
        bat_speed = self.BAT_SPEED_AVG * (1 - (dist_from_center * 0.05)) # 중심에서 멀어지면 배트 속도 감소
        
        # 3. 타구 속도 (Exit Velocity)
        # V_exit = q * V_pitch + (1+q) * V_bat
        # 투구 속도가 빠를수록 반발력 증가 (Pitch Speed 1mph 당 EV 0.2mph 증가)
        ev_mph = (q * pitch_speed_mph) + ((1 + q) * bat_speed)
        
        # 4. 발사각 (Launch Angle) - 위치에 따른 가정
        # 낮으면 땅볼(Low LA), 높으면 뜬공(High LA) 경향
        la_deg = (plate_z - 2.5) * 15 + 15 # 2.5ft -> 15도, 1.5ft -> 0도
        
        # 5. 비거리 (Simple Projectile with Drag)
        # d = (v^2 * sin(2theta)) / g ... but with heavy drag
        # 야구공 근사식: Distance ~ 2.5 * EV_mph * sin(2*LA) (매우 단순화)
        distance_ft = 0
        if 0 < la_deg < 50:
             distance_ft = 2.0 * ev_mph * np.sin(np.radians(2 * la_deg)) * (1 + (ev_mph/100))
        
        return {
            "exit_velocity": round(ev_mph, 1),
            "launch_angle": round(la_deg, 1),
            "est_distance": round(distance_ft, 1)
        }

    # ... (equations, calculate_trajectory 등 기존 궤적 계산 로직은 유지하되, 
    #      필요시 estimate_spin_parameters 결과를 활용하도록 연결 가능)
    
    def equations(self, t, state, spin_vec, rho, wind_vec):
        # (기존 Phase 1 로직과 동일하게 유지 - 3DOF Trajectory)
        x, y, z, vx, vy, vz = state
        wx, wy, wz = wind_vec
        v_rel_x = vx - wx; v_rel_y = vy - wy; v_rel_z = vz - wz
        v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
        
        const_k = 0.5 * rho * self.AREA / self.MASS
        
        # Basic Cd (Drag)
        v_mph = v_rel / 1.467
        cd = 0.30 + (100 - v_mph) * 0.002 if v_mph < 100 else 0.30
        
        ax_d = -const_k * cd * v_rel * v_rel_x
        ay_d = -const_k * cd * v_rel * v_rel_y
        az_d = -const_k * cd * v_rel * v_rel_z
        
        # Basic Magnus (Lift)
        # 스핀 벡터가 주어졌다고 가정
        sp_x, sp_y, sp_z = spin_vec
        cl = 0.3 # Average estimation
        
        mag_x = sp_y*v_rel_z - sp_z*v_rel_y
        mag_y = sp_z*v_rel_x - sp_x*v_rel_z
        mag_z = sp_x*v_rel_y - sp_y*v_rel_x
        mag_norm = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        
        if mag_norm > 0:
            lift = const_k * cl * v_rel**2
            ax_m = lift * (mag_x / mag_norm)
            ay_m = lift * (mag_y / mag_norm)
            az_m = lift * (mag_z / mag_norm)
        else: ax_m, ay_m, az_m = 0,0,0

        return [vx, vy, vz, ax_d + ax_m, ay_d + ay_m, az_d + az_m - self.GRAVITY]

    def calculate_trajectory(self, pitch_row, env_params=None):
        # (기존 로직 유지 - 코드 길이상 생략하나 핵심은 기존과 동일)
        # ...
        # 단, equations 호출 시 필요한 파라미터 전달
        # 여기서는 Zero-Cost Phase 2의 핵심인 "Estimation" 기능 추가에 집중
        
        # Trajectory 계산을 위한 최소한의 구현 (기존 코드 복원)
        temp = env_params.temperature if env_params else 70.0
        elev = env_params.elevation if env_params else 0.0
        humid = env_params.humidity if env_params else 50.0
        rho = self.calculate_air_density(temp, elev, humid)
        
        y0 = 60.5 - pitch_row.get('extension', 6.0)
        x0 = pitch_row.get('release_pos_x', 0.0)
        z0 = pitch_row.get('release_pos_z', 6.0)
        v0_mph = pitch_row.get('release_speed', 90)
        vy0 = -v0_mph * 1.467
        vx0, vz0 = 0, 0 # Simplified
        
        # 타겟 Auto-Aim (간소화)
        target_x = pitch_row.get('plate_x')
        target_z = pitch_row.get('plate_z')
        if target_x is not None:
            t = (1.417 - y0) / vy0
            vx0 = (target_x - x0) / t
            vz0 = (target_z - z0 + 0.5 * self.GRAVITY * t**2) / t

        spin = pitch_row.get('release_spin_rate', 2200)
        pfx_x = pitch_row.get('pfx_x', 0)
        pfx_z = pitch_row.get('pfx_z', 0)
        
        # Spin Vector estimation from PFX
        total_pfx = np.sqrt(pfx_x**2 + pfx_z**2) + 1e-9
        wx = -spin * (pfx_z / total_pfx)
        wz = spin * (pfx_x / total_pfx)
        spin_vec = [wx, 0, wz]

        sol = solve_ivp(
            fun=lambda t, y: self.equations(t, y, spin_vec, rho, [0,0,0]),
            t_span=(0, 1.0),
            y0=[x0, y0, z0, vx0, vy0, vz0],
            events=lambda t, y: y[1] - 1.417,
            method='RK45'
        )
        
        traj = sol.y[:3, :].T
        final_v = sol.y[3:, -1]
        vaa = -np.degrees(np.arctan(final_v[2] / final_v[1]))
        haa = -np.degrees(np.arctan(final_v[0] / final_v[1]))
        
        return traj, vaa, haa
        
    def get_position_at_y(self, pitch_row, target_y, env_params=None):
        traj, _, _ = self.calculate_trajectory(pitch_row, env_params)
        distances = np.abs(traj[:, 1] - target_y)
        idx = np.argmin(distances)
        return traj[idx, 0], traj[idx, 2]